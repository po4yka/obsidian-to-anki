"""Post-validator agent for card quality validation."""

import time

from ...models import NoteMetadata
from ...providers.base import BaseLLMProvider
from ...utils.logging import get_logger
from ..llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)
from ..models import GeneratedCard, PostValidationResult
from .auto_fix import AggressiveFixer, DeterministicFixer, RuleBasedHeaderFixer
from .error_categories import ErrorCategory
from .prompts import AUTOFIX_SYSTEM_PROMPT, build_autofix_prompt
from .semantic_validator import semantic_validation
from .syntax_validator import syntax_validation

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
                # Extract error type from error message
                if "APF format:" in error:
                    error_type = error.split("APF format:")[1].strip().split()[0:3]
                    error_key = " ".join(error_type) if error_type else "unknown"
                elif "HTML:" in error:
                    error_key = "HTML validation"
                else:
                    error_key = "other"
                error_by_type[error_key] = error_by_type.get(error_key, 0) + 1
                # Categorize error
                category = ErrorCategory.from_error_string(error)
                error_by_category[category.value] = (
                    error_by_category.get(category.value, 0) + 1
                )

            # Log error summary
            logger.warning(
                "post_validation_syntax_failed",
                errors_count=len(syntax_errors),
                error_breakdown=error_by_type,
                error_by_category=error_by_category,
            )

            # Log each error individually (up to 20 to avoid spam)
            for i, error in enumerate(syntax_errors[:20]):
                logger.warning("validation_error_detail", error_num=i + 1, error=error)

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
                error_details="; ".join(syntax_errors),
                corrected_cards=None,
                validation_time=validation_time,
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
        self, cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Attempt to auto-fix validation errors using progressive recovery strategy.

        Recovery strategy (in order):
        1. Deterministic fixes (fast, no LLM)
        2. Rule-based header fixes
        3. LLM-based fixes with comprehensive prompts
        4. Aggressive deterministic fixes (last resort)

        Args:
            cards: Original cards with errors
            error_details: Description of errors

        Returns:
            Corrected cards if successful, None otherwise
        """
        try:
            # Strategy 1: Try deterministic fixes first (fastest, most reliable)
            logger.debug(
                "auto_fix_strategy_1_deterministic", error_preview=error_details[:100]
            )
            fixed_cards = DeterministicFixer.apply_fixes(cards, error_details)
            if fixed_cards:
                logger.info(
                    "auto_fix_deterministic_success", cards_fixed=len(fixed_cards)
                )
                return fixed_cards

            # Strategy 2: Try rule-based header fixes
            if (
                "Invalid card header format" in error_details
                or "card header" in error_details.lower()
            ):
                logger.debug("auto_fix_strategy_2_rule_based")
                fixed_cards = RuleBasedHeaderFixer.apply_fixes(cards)
                if fixed_cards:
                    logger.info(
                        "auto_fix_rule_based_success", cards_fixed=len(fixed_cards)
                    )
                    return fixed_cards

            # Strategy 3: LLM-based fix for complex issues
            logger.debug("auto_fix_strategy_3_llm")
            fixed_cards = self._llm_based_fix(cards, error_details)
            if fixed_cards:
                logger.info("auto_fix_llm_success", cards_fixed=len(fixed_cards))
                return fixed_cards

            # Strategy 4: Aggressive deterministic fixes (last resort)
            logger.debug("auto_fix_strategy_4_aggressive_deterministic")
            fixed_cards = AggressiveFixer.apply_fixes(cards, error_details)
            if fixed_cards:
                logger.info(
                    "auto_fix_aggressive_deterministic_success",
                    cards_fixed=len(fixed_cards),
                )
                return fixed_cards

            logger.warning(
                "auto_fix_all_strategies_failed",
                strategies_attempted=[
                    "deterministic",
                    "rule_based",
                    "llm",
                    "aggressive_deterministic",
                ],
            )
            return None

        except Exception as e:
            logger.error("auto_fix_failed", error=str(e))
            return None

    def _llm_based_fix(
        self, cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Attempt LLM-based fix for complex issues.

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

            # Use slightly higher temperature for auto-fix to allow creativity
            # but keep it low for consistency
            base_temperature = self.temperature if self.temperature is not None else 0.0
            auto_fix_temperature = max(0.1, base_temperature)

            logger.info(
                "attempting_manual_auto_fix",
                model=self.model,
                temperature=auto_fix_temperature,
                cards_to_fix=len(cards[:10]),
            )

            result = self.ollama_client.generate_json(
                model=self.model,
                prompt=prompt,
                system=AUTOFIX_SYSTEM_PROMPT,
                temperature=auto_fix_temperature,
            )

            llm_duration = time.time() - llm_start_time

            logger.info(
                "auto_fix_llm_complete",
                duration=round(llm_duration, 2),
                result_type=type(result).__name__,
            )

            # Handle case where model returns empty or invalid JSON
            if not result or not isinstance(result, dict):
                logger.warning(
                    "auto_fix_invalid_response",
                    response_type=type(result).__name__,
                    response=str(result)[:200],
                )
                return None

            corrected_data = result.get("corrected_cards", [])
            if not corrected_data:
                logger.warning(
                    "auto_fix_no_corrections",
                    result_keys=list(result.keys()),
                    result_preview=str(result)[:200],
                )
                return None

            logger.info("auto_fix_success", corrected_cards_count=len(corrected_data))
            return [GeneratedCard(**card) for card in corrected_data]

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

            logger.error(
                "auto_fix_llm_error",
                error_type=categorized_error.error_type.value,
                error=str(categorized_error),
                user_message=format_llm_error_for_user(categorized_error),
            )

            return None
