"""Pre-validation agent using PydanticAI.

Validates note structure, formatting, and frontmatter before generation.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from ...models import NoteMetadata, QAPair
from ...utils.logging import get_logger
from ..exceptions import (
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from ..improved_prompts import PRE_VALIDATION_SYSTEM_PROMPT
from ..models import PreValidationResult
from .models import PreValidationDeps, PreValidationOutput

logger = get_logger(__name__)


class PreValidatorAgentAI:
    """PydanticAI-based pre-validation agent.

    Validates note structure, formatting, and frontmatter before generation.
    Uses structured outputs for type-safe validation results.
    """

    def __init__(self, model: OpenAIModel, temperature: float = 0.0):
        """Initialize pre-validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent with structured output
        self.agent: Agent[PreValidationDeps, PreValidationOutput] = Agent(
            model=self.model,
            result_type=PreValidationOutput,
            system_prompt=self._get_system_prompt(),
        )

        logger.info("pydantic_ai_pre_validator_initialized", model=str(model))

    def _get_system_prompt(self) -> str:
        """Get system prompt for pre-validation with few-shot examples."""
        return PRE_VALIDATION_SYSTEM_PROMPT

    async def validate(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> PreValidationResult:
        """Validate note before card generation.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path

        Returns:
            PreValidationResult with validation outcome
        """
        logger.info("pydantic_ai_pre_validation_start", title=metadata.title)

        # Create dependencies
        deps = PreValidationDeps(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            file_path=file_path,
        )

        # Build validation prompt
        prompt = f"""Validate this note for card generation:

Title: {metadata.title}
Topic: {metadata.topic}
Tags: {', '.join(metadata.tags)}
Language Tags: {', '.join(metadata.language_tags)}
Q&A Pairs: {len(qa_pairs)}

Note Content Preview:
{note_content[:500]}...

Validate the structure, frontmatter, and content quality."""

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)

            # Convert to PreValidationResult
            output: PreValidationOutput = result.data
            validation_result = PreValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=0.0,
            )

            logger.info(
                "pydantic_ai_pre_validation_complete",
                is_valid=output.is_valid,
                confidence=output.confidence,
            )

            return validation_result

        except ValueError as e:
            logger.error(
                "pydantic_ai_pre_validation_parse_error", error=str(e))
            raise StructuredOutputError(
                "Failed to parse pre-validation output",
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_pre_validation_timeout", error=str(e))
            raise ModelError(
                "Pre-validation timed out", details={"title": metadata.title}
            ) from e
        except Exception as e:
            logger.error("pydantic_ai_pre_validation_failed", error=str(e))
            raise PreValidationError(
                f"Pre-validation failed: {str(e)}", details={"title": metadata.title}
            ) from e
