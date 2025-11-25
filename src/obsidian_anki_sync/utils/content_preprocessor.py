"""Content preprocessing and validation utilities for Obsidian notes.

Provides enhanced input validation, markdown sanitization, and content quality checks
to prevent common parsing and validation errors.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContentValidationResult:
    """Result of content validation with warnings and suggestions."""

    is_valid: bool
    warnings: List[str]
    suggestions: List[str]
    sanitized_content: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """Configuration for content preprocessing."""

    sanitize_code_fences: bool = True
    add_missing_languages: bool = True
    normalize_whitespace: bool = True
    fix_malformed_frontmatter: bool = True
    max_fence_imbalance: int = 3


class ContentPreprocessor:
    """Enhanced content validation and preprocessing."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def preprocess_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Preprocess content with sanitization and validation.

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (processed_content, warnings)
        """
        warnings = []
        processed_content = content

        # Apply preprocessing steps
        if self.config.normalize_whitespace:
            processed_content, ws_warnings = self._normalize_whitespace(
                processed_content
            )
            warnings.extend(ws_warnings)

        if self.config.sanitize_code_fences:
            processed_content, fence_warnings = self._sanitize_code_fences(
                processed_content
            )
            warnings.extend(fence_warnings)

        if self.config.add_missing_languages:
            processed_content, lang_warnings = self._add_missing_language_hints(
                processed_content
            )
            warnings.extend(lang_warnings)

        if self.config.fix_malformed_frontmatter:
            processed_content, fm_warnings = self._fix_frontmatter(processed_content)
            warnings.extend(fm_warnings)

        return processed_content, warnings

    def validate_content_quality(self, content: str) -> ContentValidationResult:
        """
        Perform comprehensive quality validation on content.

        Args:
            content: Markdown content to validate

        Returns:
            Validation result with warnings and suggestions
        """
        warnings = []
        suggestions = []

        # Check for common issues
        warnings.extend(self._check_code_fence_balance(content))
        warnings.extend(self._check_frontmatter_integrity(content))
        warnings.extend(self._check_content_structure(content))

        # Generate suggestions
        suggestions.extend(self._generate_improvement_suggestions(content, warnings))

        # Determine if content is valid (warnings are acceptable)
        is_valid = len([w for w in warnings if "critical" in w.lower()]) == 0

        return ContentValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            suggestions=suggestions,
            sanitized_content=None,  # Will be set if preprocessing is applied
        )

    def _normalize_whitespace(self, content: str) -> Tuple[str, List[str]]:
        """Normalize whitespace issues."""
        warnings = []
        original_lines = content.splitlines()
        normalized_lines = []

        for i, line in enumerate(original_lines):
            # Check for trailing whitespace
            if line.rstrip() != line:
                warnings.append(f"Line {i+1}: Trailing whitespace detected")

            # Check for mixed tabs and spaces (common issue)
            if "\t" in line and " " in line[:4]:  # Mixed indentation
                warnings.append(f"Line {i+1}: Mixed tabs and spaces detected")

            # Normalize line endings
            normalized_lines.append(line.rstrip())

        # Remove excessive blank lines
        result_lines = []
        blank_count = 0
        for line in normalized_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        return "\n".join(result_lines), warnings

    def _sanitize_code_fences(self, content: str) -> Tuple[str, List[str]]:
        """Sanitize code fences to prevent parsing issues."""
        warnings = []
        lines = content.splitlines()
        sanitized_lines = []
        fence_stack = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("```"):
                if fence_stack and self._is_closing_fence(stripped, fence_stack[-1]):
                    # Valid closing fence
                    fence_stack.pop()
                    sanitized_lines.append(line)
                elif len(fence_stack) < self.config.max_fence_imbalance:
                    # Opening fence
                    fence_stack.append(stripped)
                    sanitized_lines.append(line)
                else:
                    # Too many unbalanced fences, skip this one
                    warnings.append(
                        f"Line {i+1}: Excessive unbalanced code fence ignored"
                    )
            else:
                sanitized_lines.append(line)

            i += 1

        # Close any remaining open fences
        for _ in fence_stack:
            sanitized_lines.append("```")
            warnings.append("Added missing code fence closure")

        return "\n".join(sanitized_lines), warnings

    def _add_missing_language_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add language hints to code blocks that lack them."""
        warnings = []

        def replace_code_block(match):
            fence = match.group(1)
            content = match.group(2)

            # Check if language is already specified
            if fence.strip() and not fence.strip()[3:].strip():
                # Empty language specification, try to detect
                detected_lang = self._detect_code_language(content)
                if detected_lang:
                    new_fence = f"```{detected_lang}"
                    warnings.append(
                        f"Added language hint '{detected_lang}' to code block"
                    )
                    return f"{new_fence}\n{content}\n```"
                else:
                    warnings.append("Could not detect language for code block")

            return match.group(0)  # No change needed

        # Match code blocks: ```language\ncontent\n```
        pattern = r"(```[^\n]*)\n(.*?)\n```"
        modified_content = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)

        return modified_content, warnings

    def _fix_frontmatter(self, content: str) -> Tuple[str, List[str]]:
        """Fix common frontmatter issues."""
        warnings = []

        # Check for frontmatter markers
        if not content.startswith("---"):
            warnings.append("Missing frontmatter opening marker")
            return content, warnings

        lines = content.splitlines()
        if len(lines) < 2:
            return content, warnings

        # Find closing marker
        closing_idx = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                closing_idx = i
                break

        if closing_idx == -1:
            warnings.append("Missing frontmatter closing marker")
            # Try to add one before first heading
            for i, line in enumerate(lines):
                if line.startswith("#") and i > 0:
                    lines.insert(i, "---")
                    warnings.append("Added missing frontmatter closing marker")
                    break

        return "\n".join(lines), warnings

    def _check_code_fence_balance(self, content: str) -> List[str]:
        """Check for code fence balance issues."""
        warnings = []
        lines = content.splitlines()
        fence_count = 0

        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                fence_count += 1

        if fence_count % 2 != 0:
            warnings.append(f"Unbalanced code fences detected ({fence_count} total)")

        return warnings

    def _check_frontmatter_integrity(self, content: str) -> List[str]:
        """Check frontmatter structure."""
        warnings = []

        if not content.startswith("---"):
            warnings.append("Missing frontmatter opening marker")
            return warnings

        lines = content.splitlines()
        if len(lines) < 2:
            return warnings

        # Check for closing marker
        has_closing = any(line.strip() == "---" for line in lines[1:])
        if not has_closing:
            warnings.append("Missing frontmatter closing marker")

        return warnings

    def _check_content_structure(self, content: str) -> List[str]:
        """Check overall content structure."""
        warnings = []

        # Check for required sections based on languages
        lines = content.splitlines()

        # Extract languages from frontmatter
        languages = []
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.startswith("language_tags:"):
                # Extract languages from YAML list
                lang_match = re.search(r"language_tags:\s*\[(.*?)\]", line)
                if lang_match:
                    languages = [
                        lang.strip().strip("\"'")
                        for lang in lang_match.group(1).split(",")
                    ]

        # Check for required sections
        content_str = "\n".join(lines)
        for lang in languages:
            question_marker = f"# Question ({lang.upper()})"
            answer_marker = f"## Answer ({lang.upper()})"

            if question_marker not in content_str:
                warnings.append(f"Missing required section: {question_marker}")
            if answer_marker not in content_str:
                warnings.append(f"Missing required section: {answer_marker}")

        return warnings

    def _generate_improvement_suggestions(
        self, content: str, warnings: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on warnings."""
        suggestions = []

        if any("code fence" in w.lower() for w in warnings):
            suggestions.append(
                "Consider using the content preprocessor to automatically fix code fence issues"
            )

        if any("frontmatter" in w.lower() for w in warnings):
            suggestions.append(
                "Ensure frontmatter has proper opening (---) and closing (---) markers"
            )

        if any("language" in w.lower() for w in warnings):
            suggestions.append(
                "Add language identifiers to code blocks for better syntax highlighting"
            )

        if any("whitespace" in w.lower() for w in warnings):
            suggestions.append("Remove trailing whitespace and normalize indentation")

        return suggestions

    def _is_closing_fence(self, fence: str, opener: str) -> bool:
        """Check if a fence closes an opener."""
        # For now, any fence can close any opener (simplified logic)
        return True

    def _detect_code_language(self, code: str) -> Optional[str]:
        """Attempt to detect programming language from code content."""
        # Simple heuristics for common languages
        code_lower = code.lower().strip()

        # Kotlin indicators
        if any(
            keyword in code_lower
            for keyword in [
                "fun ",
                "val ",
                "var ",
                "class ",
                "interface ",
                "suspend ",
                "coroutine",
            ]
        ):
            return "kotlin"

        # Java indicators
        if any(
            keyword in code_lower
            for keyword in [
                "public class",
                "private class",
                "static void",
                "system.out",
                "import java",
            ]
        ):
            return "java"

        # Python indicators
        if any(
            keyword in code_lower
            for keyword in [
                "def ",
                "import ",
                "from ",
                "class ",
                "print(",
                "if __name__",
            ]
        ):
            return "python"

        # JavaScript indicators
        if any(
            keyword in code_lower
            for keyword in ["function ", "const ", "let ", "var ", "console.log", "=>"]
        ):
            return "javascript"

        # Default to 'text' if nothing detected
        return "text"
