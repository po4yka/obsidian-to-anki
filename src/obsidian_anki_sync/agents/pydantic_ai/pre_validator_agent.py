"""PydanticAI-based pre-validation agent."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger

from ..exceptions import (
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from ..improved_prompts import PRE_VALIDATION_SYSTEM_PROMPT
from ..models import PreValidationResult
from .deps import PreValidationDeps
from .outputs import PreValidationOutput
from .streaming import run_agent_with_streaming

logger = get_logger(__name__)


class PreValidatorAgentAI:
    """PydanticAI-based pre-validation agent.

    Validates note structure, formatting, and frontmatter before generation.
    Uses structured outputs for type-safe validation results.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
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
            output_type=PreValidationOutput,
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
Tags: {", ".join(metadata.tags)}
Language Tags: {", ".join(metadata.language_tags)}
Q&A Pairs: {len(qa_pairs)}

Note Content Preview:
{note_content[:500]}...

Validate the structure, frontmatter, and content quality."""

        try:
            # Run agent with streaming to prevent connection timeouts
            output: PreValidationOutput = await run_agent_with_streaming(
                self.agent,
                prompt,
                deps,
                operation_name="pre_validation",
            )
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
            logger.error("pydantic_ai_pre_validation_parse_error", error=str(e))
            msg = "Failed to parse pre-validation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_pre_validation_timeout", error=str(e))
            msg = "Pre-validation timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_pre_validation_failed", error=str(e))
            msg = f"Pre-validation failed: {e!s}"
            raise PreValidationError(msg, details={"title": metadata.title}) from e
