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

        # Create PydanticAI agent with increased retry limit for output validation
        self.agent: Agent[PostValidationDeps, PostValidationOutput] = Agent(
            model=self.model,
            output_type=PostValidationOutput,
            system_prompt=self._get_system_prompt(),
            retries=3,  # Allow more retries for output validation failures
        )
        self.fixing_parser = OutputFixingParser(
            agent=self.agent,
            max_fix_attempts=1,
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

        # Build validation prompt
        prompt = f"""Validate these {len(cards)} generated APF cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {", ".join(metadata.language_tags)}

Cards to validate:
"""
        for card in cards[:3]:
            prompt += f"\nCard {card.card_index} ({card.lang}): {card.slug}\n"
            prompt += f"HTML Preview: {card.apf_html[:200]}...\n"

        prompt += (
            f"\nValidate all {len(cards)} cards for correctness, accuracy, and quality."
        )

        try:
            # Run agent
            output = await self.fixing_parser.run(prompt, deps=deps)

            # Convert suggested corrections to GeneratedCard list
            corrected_cards: list[GeneratedCard] | None = None
            if output.suggested_corrections:
                corrected_cards = []
                for correction in output.suggested_corrections:
                    try:
                        corrected_card = GeneratedCard(**correction)
                        corrected_cards.append(corrected_card)
                    except (KeyError, ValueError) as e:
                        logger.warning("invalid_correction", error=str(e))
                        continue

            validation_result = PostValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                corrected_cards=corrected_cards,
                validation_time=0.0,
            )

            logger.info(
                "pydantic_ai_post_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
                issues_found=len(output.card_issues),
            )

            return validation_result

        except ValueError as e:
            logger.error(
                "pydantic_ai_post_validation_parse_error", error=str(e))
            msg = "Failed to parse post-validation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "cards_count": len(cards)},
            ) from e
        except StructuredOutputError as e:
            logger.error(
                "pydantic_ai_post_validation_structured_error", error=str(e))
            msg = "Post-validation structured output failed"
            raise PostValidationError(
                msg, details={"cards_count": len(cards), "error": str(e)}
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_post_validation_timeout", error=str(e))
            msg = "Post-validation timed out"
            raise ModelError(msg, details={"cards_count": len(cards)}) from e
        except Exception as e:
            logger.error("pydantic_ai_post_validation_failed", error=str(e))
            msg = f"Post-validation failed: {e!s}"
            raise PostValidationError(
                msg, details={"cards_count": len(cards)}) from e
