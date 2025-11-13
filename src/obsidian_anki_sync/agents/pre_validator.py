"""Pre-validator agent for note structure and format validation.

This agent performs fast structural checks before expensive card generation:
- Markdown syntax correctness
- Frontmatter validation
- Required fields presence
- Heading structure
- Q/A pair format
"""

import time
from pathlib import Path

from ..models import NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .json_schemas import get_pre_validation_schema
from .llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)
from .models import PreValidationResult

logger = get_logger(__name__)


class PreValidatorAgent:
    """Agent for pre-validation of note structure and formatting.

    Uses lightweight model (qwen3:8b) for fast structural validation.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:8b",
        temperature: float = 0.0,
    ):
        """Initialize pre-validator agent.

        Args:
            ollama_client: LLM provider instance (BaseLLMProvider)
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
        return f"""<task>
Perform structural validation of an Obsidian note. Think step by step to identify any formatting, structure, or content issues.
</task>

<input>
<metadata>
Title: {metadata.title}
Topic: {metadata.topic}
Languages: {", ".join(metadata.language_tags)}
Q/A Pairs Found: {len(qa_pairs)}
</metadata>

<note_content_preview>
{note_content[:2000]}
</note_content_preview>
</input>

<validation_steps>
Step 1: Check frontmatter completeness
- Verify presence of required fields: id, title, topic, language_tags, created, updated
- Validate YAML syntax is correct

Step 2: Validate language tags
- Ensure language_tags contains only 'en' or 'ru' (or both)
- Verify no invalid language codes are present

Step 3: Check content presence
- Confirm at least one Q/A pair exists
- Verify Q/A pairs have content in all specified languages

Step 4: Validate markdown structure
- Check section headers are properly formatted
- Verify Q/A section ordering is logical
- Ensure no malformed markdown syntax

Step 5: Check filename pattern compliance
- Verify note filename follows 'q-' prefix convention (if applicable)
</validation_steps>

<validation_rules>
REQUIRED frontmatter fields:
- id: Unique note identifier
- title: Note title
- topic: Primary topic classification
- language_tags: Array containing 'en', 'ru', or both
- created: Creation date
- updated: Last update date

REQUIRED content:
- At least one complete Q/A pair
- Questions and answers in ALL specified languages

ALLOWED language tags:
- en (English)
- ru (Russian)

DO NOT allow:
- Empty or missing language_tags
- Invalid language codes
- Missing frontmatter fields
- Notes with zero Q/A pairs
</validation_rules>

<output_format>
Respond with valid JSON matching this structure:

{{
    "is_valid": true/false,
    "error_type": "format" | "structure" | "frontmatter" | "content" | "none",
    "error_details": "Specific description of errors found, or empty string if valid",
    "auto_fix_applied": false,
    "fixed_content": null
}}

error_type values:
- "format": Markdown syntax errors
- "structure": Section ordering or heading issues
- "frontmatter": Missing or invalid frontmatter fields
- "content": Missing or incomplete Q/A pairs
- "none": No errors found (is_valid = true)

If you can auto-fix the issue:
- Set auto_fix_applied to true
- Provide the complete corrected note content in fixed_content
</output_format>

<examples>
<example_1>
Input: Note missing 'created' field in frontmatter

Output:
{{
    "is_valid": false,
    "error_type": "frontmatter",
    "error_details": "Missing required frontmatter field: created",
    "auto_fix_applied": false,
    "fixed_content": null
}}
</example_1>

<example_2>
Input: Note with invalid language tag 'es'

Output:
{{
    "is_valid": false,
    "error_type": "frontmatter",
    "error_details": "Invalid language tag 'es'. Only 'en' and 'ru' are allowed.",
    "auto_fix_applied": false,
    "fixed_content": null
}}
</example_2>

<example_3>
Input: Valid note with all requirements met

Output:
{{
    "is_valid": true,
    "error_type": "none",
    "error_details": "",
    "auto_fix_applied": false,
    "fixed_content": null
}}
</example_3>
</examples>"""

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

            system_prompt = """<role>
You are a structural validation agent for Obsidian educational notes. Your expertise is in identifying formatting errors, missing required fields, and content completeness issues before expensive card generation.
</role>

<approach>
Think step by step through each validation requirement:
1. Analyze frontmatter structure and completeness
2. Verify language tag validity
3. Assess content presence and quality
4. Check markdown syntax correctness
5. Evaluate overall note structure

Be systematic and thorough in your analysis.
</approach>

<requirements>
- Always respond in valid JSON format matching the provided schema
- Be strict about required fields and valid values
- Be helpful by suggesting auto-fixes when issues are simple and deterministic
- Provide specific, actionable error details
- Never approve notes that fail critical requirements
</requirements>

<constraints>
DO NOT approve:
- Notes missing required frontmatter fields
- Notes with invalid language tags (only 'en' and 'ru' allowed)
- Notes with zero Q/A pairs
- Notes with malformed YAML frontmatter

DO suggest auto-fixes for:
- Simple formatting issues
- Correctable YAML syntax errors
- Minor structural improvements
</constraints>"""

            llm_start_time = time.time()

            logger.info(
                "pre_validation_llm_start",
                model=self.model,
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                content_length=len(note_content),
                prompt_length=len(prompt),
                qa_pairs_count=len(qa_pairs),
            )

            # Get JSON schema for structured output
            json_schema = get_pre_validation_schema()

            result = self.ollama_client.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
                json_schema=json_schema,
            )

            llm_duration = time.time() - llm_start_time
            validation_time = time.time() - start_time

            logger.info(
                "pre_validation_llm_complete",
                llm_duration=round(llm_duration, 2),
                total_duration=round(validation_time, 2),
            )

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
            llm_duration = time.time() - llm_start_time
            validation_time = time.time() - start_time

            # Categorize and log the error
            llm_error = categorize_llm_error(
                error=e,
                model=self.model,
                operation="pre-validation",
                duration=llm_duration,
            )

            log_llm_error(
                llm_error,
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                qa_pairs_count=len(qa_pairs),
                content_length=len(note_content),
                prompt_length=len(prompt) if "prompt" in locals() else 0,
            )

            logger.error(
                "pre_validation_llm_error",
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                model=self.model,
                error_type=llm_error.error_type.value,
                error=str(llm_error),
                user_message=format_llm_error_for_user(llm_error),
                time=validation_time,
            )

            # Fall back to basic validation result
            return PreValidationResult(
                is_valid=False,
                error_type="format",
                error_details=f"LLM validation failed: {format_llm_error_for_user(llm_error)}",
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
