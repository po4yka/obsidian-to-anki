"""Pre-validation agent using PydanticAI.

Validates note structure, formatting, and frontmatter before generation.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.agents.autofix.registry import AutoFixRegistry
from obsidian_anki_sync.agents.exceptions import (
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from obsidian_anki_sync.agents.improved_prompts import PRE_VALIDATION_SYSTEM_PROMPT
from obsidian_anki_sync.agents.models import PreValidationResult
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.validation.ai_fixer import AIFixer

from .models import PreValidationDeps, PreValidationOutput

logger = get_logger(__name__)


class PreValidatorAgentAI:
    """PydanticAI-based pre-validation agent.

    Validates note structure, formatting, and frontmatter before generation.
    Uses structured outputs for type-safe validation results.
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        temperature: float = 0.0,
        enable_autofix: bool = False,
        ai_fixer: AIFixer | None = None,
    ):
        """Initialize pre-validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
            enable_autofix: Whether to enable auto-fixing loop
            ai_fixer: Optional AIFixer instance for AI fixes
        """
        self.model = model
        self.temperature = temperature
        self.enable_autofix = enable_autofix
        self.ai_fixer = ai_fixer

        # Initialize fix registry if enabled
        self.fix_registry = (
            AutoFixRegistry(ai_fixer=ai_fixer) if enable_autofix else None
        )

        # Create PydanticAI agent with structured output
        # PreValidationOutput schema can be complex - use output_retries
        self.agent: Agent[PreValidationDeps, PreValidationOutput] = Agent(
            model=self.model,
            output_type=PreValidationOutput,
            system_prompt=self._get_system_prompt(),
            output_retries=5,  # PydanticAI output validation retries
        )

        logger.info(
            "pydantic_ai_pre_validator_initialized",
            model=str(model),
            autofix=enable_autofix,
        )

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

        # Run auto-fix loop first if enabled
        fixed_content = None
        auto_fix_applied = False

        if self.enable_autofix and self.fix_registry:
            fix_result = self.fix_registry.fix_until_valid(
                note_content, max_retries=3, file_path=file_path
            )
            if fix_result.file_modified:
                note_content = fix_result.fixed_content or note_content
                fixed_content = note_content
                auto_fix_applied = True
                logger.info(
                    "pre_validation_autofix_applied",
                    issues_fixed=fix_result.issues_fixed,
                )

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
            # Run agent (output_retries configured in Agent constructor)
            result = await self.agent.run(prompt, deps=deps)

            # Convert to PreValidationResult
            output: PreValidationOutput = result.output
            validation_result = PreValidationResult(
                is_valid=output.is_valid,
                error_type=output.error_type,
                error_details=output.error_details,
                auto_fix_applied=auto_fix_applied,
                fixed_content=fixed_content,
                validation_time=0.0,
            )

            if output.is_valid:
                logger.info(
                    "pydantic_ai_pre_validation_complete",
                    is_valid=output.is_valid,
                    confidence=output.confidence,
                )
            else:
                logger.warning(
                    "pydantic_ai_pre_validation_failed",
                    is_valid=output.is_valid,
                    confidence=output.confidence,
                    error_type=output.error_type,
                    error_details=output.error_details,
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
