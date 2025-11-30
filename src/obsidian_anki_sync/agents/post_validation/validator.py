"""Post-validator agent for card quality validation."""

import time

from obsidian_anki_sync.agents.llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)
from obsidian_anki_sync.agents.models import GeneratedCard, PostValidationResult
from obsidian_anki_sync.agents.repair_learning import get_repair_learning_system
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

from .auto_fix import AggressiveFixer, DeterministicFixer, RuleBasedHeaderFixer
from .error_categories import ErrorCategory
from .prompts import AUTOFIX_SYSTEM_PROMPT, build_autofix_prompt
from .semantic_validator import semantic_validation
from .syntax_validator import syntax_validation
from .validation_models import ValidationError

logger = get_logger(__name__)


class PostValidatorAgent:
    """Agent for post-validation of generated cards.

    Uses medium model (qwen3:14b) for quality validation with thinking mode.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:14b",
        temperature: float = 0.0,
    ):
        """Initialize post-validator agent.

        Args:
            ollama_client: LLM provider instance (BaseLLMProvider)
            model: Model to use for validation
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.repair_learning = get_repair_learning_system()
        logger.info("post_validator_agent_initialized", model=model)

    def validate(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Validate generated cards.

        Args:
            cards: List of generated cards to validate
            metadata: Note metadata for context
            strict_mode: Enable strict validation

        Returns:
            PostValidationResult with validation outcome
        """
        start_time = time.time()

        logger.info(
            "post_validation_start",
            cards_count=len(cards),
            strict_mode=strict_mode,
        )

        # Step 1: Syntax validation (fast, deterministic)
        syntax_errors = syntax_validation(cards)

        if syntax_errors:
            validation_time = time.time() - start_time

            # Categorize errors by type for better diagnostics
            error_by_type: dict[str, int] = {}
            error_by_category: dict[str, int] = {}
            for error in syntax_errors:
                error_key = error.code
                error_by_type[error_key] = error_by_type.get(error_key, 0) + 1
                # Categorize error
                category = error.category.value
                error_by_category[category] = error_by_category.get(category, 0) + 1

            # Log error summary
            logger.warning(
                "post_validation_syntax_failed",
                errors_count=len(syntax_errors),
                error_breakdown=error_by_type,
                error_by_category=error_by_category,
            )

            # Log each error individually (up to 20 to avoid spam)
            for i, error in enumerate(syntax_errors[:20]):
                logger.warning(
                    "validation_error_detail", error_num=i + 1, error=str(error)
                )

            if len(syntax_errors) > 20:
                logger.warning(
                    "validation_errors_truncated",
                    shown=20,
                    total=len(syntax_errors),
                    additional=len(syntax_errors) - 20,
                )

            return PostValidationResult(
                is_valid=False,
                error_type="syntax",
                error_details="; ".join([e.message for e in syntax_errors]),
                corrected_cards=None,
                validation_time=validation_time,
                structured_errors=[e.__dict__ for e in syntax_errors],
            )

        # Step 2: Semantic validation (AI-powered)
        try:
            semantic_result = semantic_validation(
                cards=cards,
                metadata=metadata,
                strict_mode=strict_mode,
                ollama_client=self.ollama_client,
                model=self.model,
                temperature=self.temperature,
            )

            validation_time = time.time() - start_time

            logger.info(
                "post_validation_complete",
                is_valid=semantic_result.is_valid,
                error_type=semantic_result.error_type,
                time=validation_time,
            )

            return PostValidationResult(
                is_valid=semantic_result.is_valid,
                error_type=semantic_result.error_type,
                error_details=semantic_result.error_details,
                corrected_cards=semantic_result.corrected_cards,
                validation_time=validation_time,
                structured_errors=semantic_result.structured_errors,
            )

        except Exception as e:
            validation_time = time.time() - start_time

            # Categorize and log the error properly
            llm_error = categorize_llm_error(
                error=e,
                model=self.model,
                operation="post-validation",
                duration=validation_time,
            )

            log_llm_error(
                llm_error,
                cards_count=len(cards),
                strict_mode=strict_mode,
            )

            logger.error(
                "post_validation_llm_error",
                error_type=llm_error.error_type.value,
                error=str(llm_error),
                user_message=format_llm_error_for_user(llm_error),
                time=validation_time,
            )

            return PostValidationResult(
                is_valid=False,
                error_type="semantic",
                error_details=f"Semantic validation failed: {format_llm_error_for_user(llm_error)}",
                corrected_cards=None,
                validation_time=validation_time,
            )

    def attempt_auto_fix(
        self,
        cards: list[GeneratedCard],
        error_details: str,
        structured_errors: list[dict] | None = None,
    ) -> list[GeneratedCard] | None:
        """Attempt to auto-fix validation errors using intelligent recovery strategy.

        Args:
            cards: Original cards with errors
            error_details: Description of errors
            structured_errors: Optional list of structured error dicts

        Returns:
            Corrected cards if successful, None otherwise
        """
        try:
            # Reconstruct ValidationError objects if available
            validation_errors = []
            if structured_errors:
                for err_dict in structured_errors:
                    try:
                        # Handle enum reconstruction
                        if isinstance(err_dict.get("category"), str):
                            err_dict["category"] = ErrorCategory(err_dict["category"])
                        validation_errors.append(ValidationError(**err_dict))
                    except Exception as e:
                        logger.warning("failed_to_reconstruct_error", error=str(e))

            # Determine primary error category/type for strategy lookup
            primary_error = validation_errors[0] if validation_errors else None
            error_category = (
                primary_error.category.value if primary_error else "unknown"
            )
            error_type = primary_error.code if primary_error else "unknown"

            # Ask learning system for best strategy
            suggested_strategy = self.repair_learning.suggest_strategy(
                error_category, error_type, error_details
            )

            strategies = [
                ("deterministic", self._apply_deterministic_fix),
                ("rule_based", self._apply_rule_based_fix),
                ("llm", self._apply_llm_fix),
                ("aggressive", self._apply_aggressive_fix),
            ]

            # Reorder strategies based on suggestion
            if suggested_strategy:
                # Move suggested strategy to front
                strategies.sort(
                    key=lambda x: 0 if x[0] == suggested_strategy else 1
                )
                logger.info(
                    "using_suggested_repair_strategy",
                    strategy=suggested_strategy,
                    error_category=error_category,
                )

            # Try strategies in order
            for strategy_name, strategy_func in strategies:
                logger.debug(f"auto_fix_attempting_{strategy_name}")
                fixed_cards = strategy_func(cards, error_details, validation_errors)

                if fixed_cards:
                    logger.info(
                        f"auto_fix_{strategy_name}_success",
                        cards_fixed=len(fixed_cards),
                    )
                    # Learn from success
                    self.repair_learning.learn_from_success(
                        error_category=error_category,
                        error_type=error_type,
                        error_message=error_details,
                        strategy_used=strategy_name,
                    )
                    return fixed_cards

            logger.warning(
                "auto_fix_all_strategies_failed",
                error_category=error_category,
                strategies_attempted=[s[0] for s in strategies],
            )
            return None

        except Exception as e:
            logger.error("auto_fix_failed", error=str(e))
            return None

    def _apply_deterministic_fix(
        self,
        cards: list[GeneratedCard],
        error_details: str,
        errors: list[ValidationError],
    ) -> list[GeneratedCard] | None:
        """Apply deterministic fixes."""
        # Check if errors seem fixable by deterministic fixer
        if errors:
            if any(
                e.category
                in [ErrorCategory.APF_FORMAT, ErrorCategory.HTML, ErrorCategory.SYNTAX]
                for e in errors
            ):
                return DeterministicFixer.apply_fixes(cards, error_details)
        # Fallback to string check if no structured errors
        elif "Missing" in error_details or "Tag" in error_details:
            return DeterministicFixer.apply_fixes(cards, error_details)
        return None

    def _apply_rule_based_fix(
        self,
        cards: list[GeneratedCard],
        error_details: str,
        errors: list[ValidationError],
    ) -> list[GeneratedCard] | None:
        """Apply rule-based fixes."""
        if errors:
            if any(e.category == ErrorCategory.APF_FORMAT for e in errors):
                return RuleBasedHeaderFixer.apply_fixes(cards)
        elif "header" in error_details.lower():
            return RuleBasedHeaderFixer.apply_fixes(cards)
        return None

    def _apply_llm_fix(
        self,
        cards: list[GeneratedCard],
        error_details: str,
        errors: list[ValidationError],
    ) -> list[GeneratedCard] | None:
        """Apply LLM-based fix with progressive temperature."""
        return self._llm_based_fix(cards, error_details)

    def _apply_aggressive_fix(
        self,
        cards: list[GeneratedCard],
        error_details: str,
        errors: list[ValidationError],
    ) -> list[GeneratedCard] | None:
        """Apply aggressive deterministic fixes."""
        return AggressiveFixer.apply_fixes(cards, error_details)

    def _llm_based_fix(
        self, cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Attempt LLM-based fix for complex issues with progressive temperature.

        Args:
            cards: Cards to fix
            error_details: Error description

        Returns:
            Fixed cards if successful, None otherwise
        """
        try:
            llm_start_time = time.time()

            # Build prompt
            prompt = build_autofix_prompt(cards, error_details)

            # Progressive temperature strategy
            temperatures = [0.1, 0.3, 0.5]
            base_temp = self.temperature if self.temperature is not None else 0.0
            if base_temp > 0.1:
                temperatures = [base_temp, base_temp + 0.2, base_temp + 0.4]

            for temp in temperatures:
                logger.info(
                    "attempting_llm_auto_fix",
                    model=self.model,
                    temperature=temp,
                    cards_to_fix=len(cards[:10]),
                )

                try:
                    result = self.ollama_client.generate_json(
                        model=self.model,
                        prompt=prompt,
                        system=AUTOFIX_SYSTEM_PROMPT,
                        temperature=temp,
                    )

                    if result and isinstance(result, dict):
                        corrected_data = result.get("corrected_cards", [])
                        if corrected_data:
                            llm_duration = time.time() - llm_start_time
                            logger.info(
                                "auto_fix_llm_success",
                                duration=round(llm_duration, 2),
                                temperature=temp,
                                cards_fixed=len(corrected_data),
                            )
                            return [GeneratedCard(**card) for card in corrected_data]

                except Exception as e:
                    logger.warning(
                        "auto_fix_llm_attempt_failed", temperature=temp, error=str(e)
                    )
                    continue

            logger.warning("auto_fix_llm_all_attempts_failed")
            return None

        except Exception as llm_error:
            llm_duration = time.time() - llm_start_time

            # Categorize and log the error
            categorized_error = categorize_llm_error(
                error=llm_error,
                model=self.model,
                operation="auto-fix",
                duration=llm_duration,
            )

            log_llm_error(
                categorized_error,
                cards_to_fix=len(cards[:10]),
                error_details_length=len(error_details),
            )

            return None
