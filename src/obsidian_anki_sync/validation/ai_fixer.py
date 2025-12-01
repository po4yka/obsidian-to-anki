"""AI-powered auto-fix system for validation issues.

This module provides AI-driven fixes for common validation issues:
- Code block language detection
- Bilingual title generation
- List formatting fixes
"""

import re
import time
import json
from typing import Any

from pydantic import BaseModel, Field

from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseValidator, Severity, ValidationIssue

logger = get_logger(__name__)


# Pydantic models for structured outputs


class CodeLanguageResult(BaseModel):
    """Result of code language detection."""

    language: str = Field(
        description="Detected programming language (lowercase, e.g., 'kotlin', 'python', 'java', 'swift', 'xml', 'json', 'yaml', 'bash', 'sql', 'text')"
    )
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class BilingualTitleResult(BaseModel):
    """Result of bilingual title generation."""

    en_title: str = Field(description="English title (concise, technical)")
    ru_title: str = Field(description="Russian title (accurate translation)")


class StructureFixResult(BaseModel):
    """Result of structure fixing."""

    fixed_content: str = Field(description="Reorganized note content")
    changes_made: list[str] = Field(description="List of changes applied")


class AIFixer:
    """AI-powered fixer using existing provider system."""

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        model: str = "qwen/qwen-2.5-14b-instruct",
        temperature: float = 0.1,
    ):
        """Initialize AI fixer.

        Args:
            provider: LLM provider instance. If None, AI features are disabled.
            model: Model name to use for AI operations
            temperature: Sampling temperature for generation
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.available = provider is not None

        if self.available and provider is not None:
            logger.info(
                "ai_fixer_initialized",
                provider=provider.get_provider_name(),
                model=model,
            )
        else:
            logger.info("ai_fixer_disabled", reason="no_provider")

    @classmethod
    def from_config(cls, config: Any) -> "AIFixer":
        """Create AIFixer from configuration.

        Args:
            config: Configuration object with provider settings

        Returns:
            Initialized AIFixer instance
        """
        enable_ai = getattr(config, "enable_ai_validation", False)

        if not enable_ai:
            logger.info("ai_fixer_disabled_by_config")
            return cls(provider=None)

        try:
            provider = ProviderFactory.create_from_config(config)
            model = getattr(config, "ai_validation_model", "qwen/qwen-2.5-14b-instruct")
            temperature = getattr(config, "ai_validation_temperature", 0.1)

            logger.info(
                "ai_fixer_created_from_config",
                provider=provider.get_provider_name(),
                model=model,
            )
            return cls(provider=provider, model=model, temperature=temperature)

        except Exception as e:
            logger.error(
                "ai_fixer_creation_failed",
                error=str(e),
                fallback="disabled",
            )
            return cls(provider=None)

    def detect_code_language(self, code_block: str) -> str | None:
        """Detect the programming language of a code block using AI.

        Args:
            code_block: The code content to analyze

        Returns:
            Detected language name (lowercase) or None if detection failed
        """
        if not self.available or not code_block.strip() or self.provider is None:
            return None

        system_prompt = """You are a code language detector. Analyze the code and identify the programming language.
Common languages: kotlin, java, python, swift, javascript, typescript, xml, json, yaml, bash, sql, groovy, gradle, text.
For Android development, prefer 'kotlin' over 'java' when uncertain.
For build files, use 'groovy' for build.gradle or 'kotlin' for build.gradle.kts.
If truly uncertain, use 'text'."""

        user_prompt = f"""Identify the programming language of this code:

```
{code_block[:2000]}
```

Respond with JSON: {{"language": "<language>", "confidence": "<high|medium|low>"}}"""

        try:
            # Use provider's JSON generation with schema
            schema = {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["language", "confidence"],
            }

            # Retry logic for JSON errors
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    result = self.provider.generate_json(
                        model=self.model,
                        prompt=user_prompt,
                        system=system_prompt,
                        temperature=self.temperature,
                        json_schema=schema,
                    )
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    logger.warning(
                        "code_language_detection_json_error",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    if attempt == max_retries - 1:
                        raise last_error
                    time.sleep(1)  # Brief pause before retry

            # Validate result matches our expected structure
            parsed = CodeLanguageResult(**result)

            if parsed.confidence in ("high", "medium"):
                logger.debug(
                    "code_language_detected",
                    language=parsed.language,
                    confidence=parsed.confidence,
                )
                return parsed.language.lower()

            logger.debug(
                "code_language_low_confidence",
                language=parsed.language,
                confidence=parsed.confidence,
            )
            return None

        except Exception as e:
            logger.error("code_language_detection_failed", error=str(e), exc_info=True)
            return None

    def generate_bilingual_title(
        self, content: str, current_title: str
    ) -> tuple[str, str] | None:
        """Generate bilingual title (EN / RU) from content or translate existing title.

        Args:
            content: Full note content for context
            current_title: Current title (may be missing bilingual format)

        Returns:
            Tuple of (english_title, russian_title) or None if generation failed
        """
        if not self.available or self.provider is None:
            return None

        # Extract key information from content
        question_match = re.search(
            r"# Question \(EN\)\s*\n(.*?)(?=\n#|\n##|$)", content, re.DOTALL
        )
        ru_question_match = re.search(
            r"# Вопрос \(RU\)\s*\n(.*?)(?=\n#|\n##|$)", content, re.DOTALL
        )

        context = ""
        if question_match:
            context = question_match.group(1).strip()[:500]
        elif ru_question_match:
            context = ru_question_match.group(1).strip()[:500]

        system_prompt = """You are a technical documentation specialist. Generate concise bilingual titles for technical interview questions.
Rules:
- English title should be technical and concise (3-8 words)
- Russian title should be an accurate translation
- Use proper technical terminology in both languages
- Format: "English Title / Russian Title" """

        user_prompt = f"""Current title: {current_title}

Question context:
{context}

Generate a bilingual title. Respond with JSON: {{"en_title": "<English>", "ru_title": "<Russian>"}}"""

        try:
            # Use provider's JSON generation with schema
            schema = {
                "type": "object",
                "properties": {
                    "en_title": {"type": "string"},
                    "ru_title": {"type": "string"},
                },
                "required": ["en_title", "ru_title"],
            }

            # Retry logic for JSON errors
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    result = self.provider.generate_json(
                        model=self.model,
                        prompt=user_prompt,
                        system=system_prompt,
                        temperature=self.temperature,
                        json_schema=schema,
                    )
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    logger.warning(
                        "bilingual_title_generation_json_error",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    if attempt == max_retries - 1:
                        raise last_error
                    time.sleep(1)  # Brief pause before retry

            # Validate result matches our expected structure
            parsed = BilingualTitleResult(**result)

            logger.debug(
                "bilingual_title_generated",
                en_title=parsed.en_title,
                ru_title=parsed.ru_title,
            )
            return (parsed.en_title, parsed.ru_title)

        except Exception as e:
            logger.error("bilingual_title_generation_failed", error=str(e), exc_info=True)
            return None

    def fix_note_structure(self, content: str) -> tuple[str, list[str]]:
        """Fix note structure (headers, order, missing sections) using AI.

        Args:
            content: Full note content

        Returns:
            Tuple of (fixed_content, list_of_changes)
        """
        if not self.available or self.provider is None:
            return content, []

        system_prompt = """You are an Obsidian note structure fixer.
Your goal is to reorganize and fix the note to match the expected format strictly.
Do not change the content (text, code), only the structure, headers, and order.

Expected Structure:
1. YAML Frontmatter (preserve existing)
2. # Question (EN)
3. # Вопрос (RU)
4. ## Answer (EN)
5. ## Ответ (RU)
6. ## References (optional)
7. ## Ссылки (RU) (optional)
8. ## Related Questions (optional)

Rules:
- Ensure all headers match exactly as above.
- Move content to appropriate sections.
- If a section is missing but content exists, add the header.
- Maintain all code blocks and text exactly as is.
- Ensure 2 blank lines before H1/H2 headers.
"""

        user_prompt = f"""Fix the structure of this note:

```markdown
{content}
```

Respond with JSON: {{"fixed_content": "<full markdown content>", "changes_made": ["<change 1>", "<change 2>"]}}"""

        try:
            schema = {
                "type": "object",
                "properties": {
                    "fixed_content": {"type": "string"},
                    "changes_made": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fixed_content", "changes_made"],
            }

            # Retry logic for JSON errors
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    result = self.provider.generate_json(
                        model=self.model,
                        prompt=user_prompt,
                        system=system_prompt,
                        temperature=self.temperature,
                        json_schema=schema,
                    )
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    logger.warning(
                        "structure_fix_json_error",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    if attempt == max_retries - 1:
                        raise last_error
                    time.sleep(1)  # Brief pause before retry

            parsed = StructureFixResult(**result)
            return parsed.fixed_content, parsed.changes_made

        except Exception as e:
            logger.error("structure_fix_failed", error=str(e), exc_info=True)
            return content, []


class AIFixerValidator(BaseValidator):
    """Validator that provides AI-powered auto-fixes.

    This validator checks for common issues that can be automatically fixed
    using AI:
    - Code blocks without language specification
    - Missing bilingual titles
    - List formatting issues
    """

    def __init__(
        self,
        content: str,
        frontmatter: dict[str, Any],
        filepath: str,
        ai_fixer: AIFixer | None = None,
        enable_ai_fixes: bool = True,
    ):
        """Initialize the AI fixer validator.

        Args:
            content: Full note content including frontmatter
            frontmatter: Parsed YAML frontmatter as dict
            filepath: Path to the note file
            ai_fixer: Optional AIFixer instance (created if not provided)
            enable_ai_fixes: Enable AI-powered fixes (only works if ai_fixer is available)
        """
        super().__init__(content, frontmatter, filepath)
        self.ai_fixer = ai_fixer
        self.enable_ai_fixes = enable_ai_fixes and (
            ai_fixer is not None and ai_fixer.available
        )
        self._code_blocks_without_lang: list[tuple[int, str]] = []
        self._needs_bilingual_title = False

        if self.enable_ai_fixes:
            logger.debug("ai_fixer_validator_enabled", filepath=filepath)
        else:
            logger.debug("ai_fixer_validator_disabled", filepath=filepath)

    def validate(self) -> list[ValidationIssue]:
        """Perform validation and register AI-powered fixes.

        Returns:
            List of validation issues found
        """
        self._check_code_blocks_for_ai_fix()
        self._check_bilingual_title()
        self._check_list_formatting()
        return self.issues

    def _check_code_blocks_for_ai_fix(self) -> None:
        """Find code blocks without language specification and offer AI fix."""
        lines = self.content.split("\n")
        in_code_block = False
        code_block_start = None
        code_block_content: list[str] = []

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Opening code block
                    in_code_block = True
                    code_block_start = i
                    lang_spec = line.strip()[3:].strip()
                    if not lang_spec:
                        code_block_content = []
                    else:
                        code_block_start = None  # Has language, skip
                else:
                    # Closing code block
                    if code_block_start is not None:
                        content = "\n".join(code_block_content)
                        self._code_blocks_without_lang.append(
                            (code_block_start, content)
                        )
                    in_code_block = False
                    code_block_start = None
                    code_block_content = []
            elif in_code_block and code_block_start is not None:
                code_block_content.append(line)

        if self._code_blocks_without_lang:
            if self.enable_ai_fixes:
                self.add_fix(
                    f"AI-detect language for {len(self._code_blocks_without_lang)} code block(s)",
                    lambda: self._fix_code_block_languages(),
                    Severity.WARNING,
                    safe=True,
                )
            else:
                # Add issue without fix if AI is disabled
                self.add_issue(
                    Severity.WARNING,
                    f"Found {len(self._code_blocks_without_lang)} code block(s) without language specification",
                )

    def _fix_code_block_languages(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Detect and add language to code blocks using AI.

        Returns:
            Tuple of (new_content, frontmatter)
        """
        lines = self.content.split("\n")

        # Process in reverse order to maintain line indices
        for line_idx, code_content in reversed(self._code_blocks_without_lang):
            if self.ai_fixer is None:
                continue
            detected_lang = self.ai_fixer.detect_code_language(code_content)
            if detected_lang:
                # Replace ``` with ```<language>
                original_line = lines[line_idx]
                indent = len(original_line) - len(original_line.lstrip())
                lines[line_idx] = " " * indent + "```" + detected_lang
                logger.info(
                    "code_language_fixed",
                    line=line_idx + 1,
                    language=detected_lang,
                    filepath=self.filepath,
                )

        new_content = "\n".join(lines)
        return new_content, self.frontmatter

    def _check_bilingual_title(self) -> None:
        """Check if title needs bilingual format and offer AI fix."""
        title = self.frontmatter.get("title", "")
        if " / " not in title:
            self._needs_bilingual_title = True
            if self.enable_ai_fixes:
                self.add_fix(
                    "AI-generate bilingual title (EN / RU)",
                    lambda: self._fix_bilingual_title(),
                    Severity.WARNING,
                    safe=False,  # Changing metadata
                )
            else:
                # Add issue without fix if AI is disabled
                self.add_issue(
                    Severity.WARNING,
                    "Title missing bilingual format (EN / RU)",
                )

    def _fix_bilingual_title(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Generate bilingual title using AI.

        Returns:
            Tuple of (content, new_frontmatter)
        """
        if self.ai_fixer is None:
            return self.content, self.frontmatter

        current_title = self.frontmatter.get("title", "")
        result = self.ai_fixer.generate_bilingual_title(self.content, current_title)

        if result:
            en_title, ru_title = result
            new_title = f"{en_title} / {ru_title}"
            new_frontmatter = dict(self.frontmatter)
            new_frontmatter["title"] = new_title
            logger.info(
                "bilingual_title_fixed",
                old_title=current_title,
                new_title=new_title,
                filepath=self.filepath,
            )
            return self.content, new_frontmatter

        logger.warning(
            "bilingual_title_generation_failed_fallback",
            filepath=self.filepath,
        )
        return self.content, self.frontmatter

    def _check_list_formatting(self) -> None:
        """Check list formatting and offer simple auto-fix."""
        lines = self.content.split("\n")
        has_issues = False

        for i, line in enumerate(lines, start=1):
            # Check for list items with multiple spaces after dash
            if re.match(r"^(\s*)-\s\s+\S", line):
                has_issues = True
                break

        if has_issues:
            self.add_fix(
                "Fix list spacing (single space after dash)",
                lambda: self._fix_list_spacing(),
                Severity.WARNING,
                safe=True,
            )

    def _fix_list_spacing(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Normalize list item spacing.

        Returns:
            Tuple of (new_content, frontmatter)
        """
        # Replace multiple spaces after dash with single space
        new_content = re.sub(r"^(\s*)-\s\s+", r"\1- ", self.content, flags=re.MULTILINE)
        logger.info("list_spacing_fixed", filepath=self.filepath)
        return new_content, self.frontmatter
