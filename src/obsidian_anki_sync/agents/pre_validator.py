"""Pre-validator agent for note structure and format validation.

This agent performs fast structural checks before expensive card generation:
- Markdown syntax correctness
- Frontmatter validation
- Required fields presence
- Heading structure
- Q/A pair format
"""

import json
import time
from pathlib import Path

from ..models import NoteMetadata, QAPair
from ..utils.logging import get_logger
from .models import PreValidationResult
from .ollama_client import OllamaClient

logger = get_logger(__name__)


class PreValidatorAgent:
    """Agent for pre-validation of note structure and formatting.

    Uses lightweight model (qwen3:8b) for fast structural validation.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "qwen3:8b",
        temperature: float = 0.0,
    ):
        """Initialize pre-validator agent.

        Args:
            ollama_client: Ollama client instance
            model: Model to use for validation
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        logger.info("pre_validator_agent_initialized", model=model)

    def _build_validation_prompt(
        self, note_content: str, metadata: NoteMetadata, qa_pairs: list[QAPair]
    ) -> str:
        """Build validation prompt for the LLM.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs

        Returns:
            Formatted prompt string
        """
        return f"""Analyze this Obsidian note for structural and formatting issues.

NOTE METADATA:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {', '.join(metadata.language_tags)}
- Q/A Pairs Found: {len(qa_pairs)}

NOTE CONTENT (first 2000 chars):
{note_content[:2000]}

VALIDATION REQUIREMENTS:
1. Frontmatter must include: id, title, topic, language_tags, created, updated
2. At least one Q/A pair must be present
3. Language tags must be either 'en' or 'ru' (or both)
4. Q/A pairs must have questions and answers in specified languages
5. Note should start with 'q-' filename pattern
6. Markdown syntax should be valid

ANALYZE AND RESPOND IN JSON FORMAT:
{{
    "is_valid": true/false,
    "error_type": "format" | "structure" | "frontmatter" | "content" | "none",
    "error_details": "Detailed description of any errors",
    "auto_fix_applied": false,
    "fixed_content": null
}}

If you can suggest a fix, set auto_fix_applied to true and provide fixed_content.
Be specific about errors. If everything is valid, set error_type to "none"."""

    def validate(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> PreValidationResult:
        """Validate note structure and formatting.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path for filename validation

        Returns:
            PreValidationResult with validation outcome
        """
        start_time = time.time()

        logger.info(
            "pre_validation_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
            content_length=len(note_content),
        )

        # First, perform basic structural checks (fast)
        basic_errors = self._basic_validation(metadata, qa_pairs, file_path)

        if basic_errors:
            # Try auto-fix for simple issues
            fixed_content = self._attempt_auto_fix(note_content, basic_errors)

            validation_time = time.time() - start_time
            logger.warning(
                "pre_validation_basic_failed",
                errors=basic_errors,
                auto_fix=bool(fixed_content),
            )

            return PreValidationResult(
                is_valid=False,
                error_type="structure",
                error_details="; ".join(basic_errors),
                auto_fix_applied=bool(fixed_content),
                fixed_content=fixed_content,
                validation_time=validation_time,
            )

        # Advanced AI validation using LLM
        try:
            prompt = self._build_validation_prompt(note_content, metadata, qa_pairs)

            system_prompt = """You are a validation agent for Obsidian notes.
Your job is to check note structure, formatting, and completeness.
Always respond in valid JSON format.
Be strict but helpful - suggest fixes when possible."""

            result = self.ollama_client.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
            )

            validation_time = time.time() - start_time

            # Parse LLM response into PreValidationResult
            validation_result = PreValidationResult(
                is_valid=result.get("is_valid", False),
                error_type=result.get("error_type", "none"),
                error_details=result.get("error_details", ""),
                auto_fix_applied=result.get("auto_fix_applied", False),
                fixed_content=result.get("fixed_content"),
                validation_time=validation_time,
            )

            logger.info(
                "pre_validation_complete",
                is_valid=validation_result.is_valid,
                error_type=validation_result.error_type,
                auto_fix=validation_result.auto_fix_applied,
                time=validation_time,
            )

            return validation_result

        except Exception as e:
            validation_time = time.time() - start_time
            logger.error("pre_validation_llm_error", error=str(e), time=validation_time)

            # Fall back to basic validation result
            return PreValidationResult(
                is_valid=False,
                error_type="format",
                error_details=f"LLM validation failed: {str(e)}",
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=validation_time,
            )

    def _basic_validation(
        self, metadata: NoteMetadata, qa_pairs: list[QAPair], file_path: Path | None
    ) -> list[str]:
        """Perform basic structural validation without LLM.

        Args:
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required metadata fields
        if not metadata.id:
            errors.append("Missing required field: id")
        if not metadata.title:
            errors.append("Missing required field: title")
        if not metadata.topic:
            errors.append("Missing required field: topic")

        # Check language tags
        if not metadata.language_tags:
            errors.append("No language_tags specified")
        else:
            allowed_langs = {"en", "ru"}
            invalid = set(metadata.language_tags) - allowed_langs
            if invalid:
                errors.append(f"Invalid language tags: {invalid}")

        # Check Q/A pairs
        if not qa_pairs:
            errors.append("No Q/A pairs found in note")

        # Check filename pattern if file_path provided
        if file_path and not file_path.name.startswith("q-"):
            errors.append(f"File should start with 'q-': {file_path.name}")

        return errors

    def _attempt_auto_fix(self, note_content: str, errors: list[str]) -> str | None:
        """Attempt to auto-fix simple structural errors.

        Args:
            note_content: Original note content
            errors: List of validation errors

        Returns:
            Fixed content if successful, None otherwise
        """
        # For now, we don't implement auto-fix at the basic level
        # The LLM-based validation can suggest fixes
        # This could be extended in the future
        return None
