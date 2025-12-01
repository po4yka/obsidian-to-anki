"""Post-validation agent using PydanticAI.

Validates generated cards for quality, syntax, and accuracy.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.agents.exceptions import (
    ModelError,
    PostValidationError,
    StructuredOutputError,
)
from obsidian_anki_sync.agents.improved_prompts import POST_VALIDATION_SYSTEM_PROMPT
from obsidian_anki_sync.agents.models import GeneratedCard, PostValidationResult
from obsidian_anki_sync.agents.output_fixing import OutputFixingParser
from obsidian_anki_sync.agents.patch_applicator import apply_corrections
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.utils.logging import get_logger

from .models import PostValidationDeps, PostValidationOutput

logger = get_logger(__name__)


class PostValidatorAgentAI:
    """PydanticAI-based post-validation agent.

    Validates generated cards for quality, syntax, and accuracy.
    Uses structured outputs to identify and suggest corrections.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize post-validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        # PostValidationOutput has nested CardCorrection objects which LLMs sometimes
        # struggle to produce correctly on first attempt - give more attempts via output_retries
        self.agent: Agent[PostValidationDeps, PostValidationOutput] = Agent(
            model=self.model,
            output_type=PostValidationOutput,
            system_prompt=self._get_system_prompt(),
            output_retries=5,  # PydanticAI output validation retries
        )
        # OutputFixingParser handles retries at the prompt improvement level
        self.fixing_parser = OutputFixingParser(
            agent=self.agent,
            max_fix_attempts=2,
            fix_temperature=temperature,
        )

        logger.info("pydantic_ai_post_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for post-validation with few-shot examples."""
        return POST_VALIDATION_SYSTEM_PROMPT

    async def validate(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Validate generated cards.

        Args:
            cards: Generated cards to validate
            metadata: Note metadata for context
            strict_mode: Enable strict validation

        Returns:
            PostValidationResult with validation outcome
        """
        logger.info("pydantic_ai_post_validation_start",
                    cards_count=len(cards))

        # Create dependencies
        deps = PostValidationDeps(
            cards=cards, metadata=metadata, strict_mode=strict_mode
        )

        # Build validation prompt with FULL card content
        prompt = f"""Validate these {len(cards)} generated APF cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {", ".join(metadata.language_tags)}

Cards to validate:
"""
        # Include full HTML content for proper validation (up to 3 cards to avoid token limits)
        for card in cards[:3]:
            prompt += f"\n--- Card {card.card_index} ({card.lang}): {card.slug} ---\n"
            prompt += f"```html\n{card.apf_html}\n```\n"

        if len(cards) > 3:
            prompt += f"\n(Plus {len(cards) - 3} more cards with similar structure)\n"

        prompt += (
            f"\nValidate all {len(cards)} cards for APF v2.1 compliance, correctness, and quality."
        )

        try:
            # Run agent
            output = await self.fixing_parser.run(prompt, deps=deps)

            # Apply suggested corrections if any
            corrected_cards = None
            applied_changes: list[str] = []

            if output.suggested_corrections:
                # Log detailed info about each suggested correction
                for i, corr in enumerate(output.suggested_corrections):
                    suggested_preview = corr.suggested_value[:200] if corr.suggested_value else ""
                    has_html_entities = "&lt;" in suggested_preview or "&gt;" in suggested_preview
                    logger.info(
                        "post_validation_correction_suggested",
                        correction_index=i,
                        card_index=corr.card_index,
                        field_name=corr.field_name,
                        rationale=corr.rationale[:100] if corr.rationale else "",
                        has_html_entities=has_html_entities,
                        suggested_preview=suggested_preview,
                    )

                corrected_cards, applied_changes = apply_corrections(
                    cards, output.suggested_corrections
                )
                logger.info(
                    "post_validation_corrections_applied",
                    suggested_count=len(output.suggested_corrections),
                    applied_count=len(applied_changes),
                    skipped_count=len(output.suggested_corrections) - len(applied_changes),
                )

            validation_result = PostValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                suggested_corrections=output.suggested_corrections,
                corrected_cards=corrected_cards,
                applied_changes=applied_changes,
                validation_time=0.0,
            )

            logger.info(
                "pydantic_ai_post_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
                issues_found=len(output.card_issues),
                corrections_applied=len(applied_changes),
            )

            return validation_result

        except ValueError as e:
            logger.error(
                "pydantic_ai_post_validation_parse_error",
                error=str(e),
                model=str(self.model),
                cards_count=len(cards),
            )
            msg = "Failed to parse post-validation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "cards_count": len(cards), "model": str(self.model)},
            ) from e
        except StructuredOutputError as e:
            logger.error(
                "pydantic_ai_post_validation_structured_error",
                error=str(e),
                model=str(self.model),
                cards_count=len(cards),
            )
            msg = "Post-validation structured output failed"
            raise PostValidationError(
                msg, details={"cards_count": len(cards), "error": str(e), "model": str(self.model)}
            ) from e
        except TimeoutError as e:
            logger.error(
                "pydantic_ai_post_validation_timeout",
                error=str(e),
                model=str(self.model),
                cards_count=len(cards),
            )
            msg = "Post-validation timed out"
            raise ModelError(msg, details={"cards_count": len(cards), "model": str(self.model)}) from e
        except Exception as e:
            logger.error(
                "pydantic_ai_post_validation_failed",
                error=str(e),
                error_type=type(e).__name__,
                model=str(self.model),
                cards_count=len(cards),
            )
            msg = f"Post-validation failed: {e!s}"
            raise PostValidationError(
                msg, details={"cards_count": len(cards), "model": str(self.model)}) from e
