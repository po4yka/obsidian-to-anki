"""Markdown structure validation for APF cards.

Validates Markdown-formatted APF content to ensure proper structure
before conversion to HTML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class MarkdownValidationResult:
    """Result of Markdown validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_markdown(content: str) -> MarkdownValidationResult:
    """
    Validate Markdown structure.

    Args:
        content: Markdown-formatted content

    Returns:
        MarkdownValidationResult with validation status and issues
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not content or not content.strip():
        return MarkdownValidationResult(is_valid=True)

    # Check for unbalanced code fences
    fence_errors = _validate_code_fences(content)
    errors.extend(fence_errors)

    # Check for balanced formatting markers
    format_errors = _validate_formatting_markers(content)
    errors.extend(format_errors)

    # Check for common issues
    issue_warnings = _check_common_issues(content)
    warnings.extend(issue_warnings)

    return MarkdownValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _validate_code_fences(content: str) -> list[str]:
    """Validate that code fences are properly balanced."""
    errors: list[str] = []

    # Find all code fence markers (``` with optional language)
    fence_pattern = r"^```[\w]*\s*$"
    lines = content.split("\n")

    fence_stack: list[int] = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(fence_pattern, stripped) or stripped == "```":
            if fence_stack:
                # Closing fence
                fence_stack.pop()
            else:
                # Opening fence
                fence_stack.append(i)

    if fence_stack:
        errors.append(
            f"Unclosed code fence starting at line {fence_stack[0]}. "
            "Each ``` must have a matching closing ```."
        )

    return errors


def _validate_formatting_markers(content: str) -> list[str]:
    """Validate that formatting markers are balanced."""
    errors: list[str] = []

    # Remove code blocks first to avoid false positives
    content_without_code = _remove_code_blocks(content)

    # Check for unbalanced inline code
    backtick_count = content_without_code.count("`")
    if backtick_count % 2 != 0:
        errors.append(
            "Unbalanced inline code markers (`). "
            "Each opening backtick needs a closing one."
        )

    # Check for unbalanced bold markers
    # Count ** pairs - should be even number
    bold_marker_count = len(re.findall(r"\*\*", content_without_code))
    if bold_marker_count % 2 != 0:
        errors.append(
            "Unbalanced bold markers (**). "
            "Each ** needs a matching closing **."
        )

    return errors


def _remove_code_blocks(content: str) -> str:
    """Remove code blocks from content for validation purposes."""
    # Remove fenced code blocks
    result = re.sub(r"```[\w]*\n.*?```", "", content, flags=re.DOTALL)
    # Remove inline code
    result = re.sub(r"`[^`]+`", "", result)
    return result


def _check_common_issues(content: str) -> list[str]:
    """Check for common Markdown issues that may cause problems."""
    warnings: list[str] = []

    # Check for HTML that might indicate mixed content
    html_tags = ["<pre>", "<code>", "<strong>", "<em>", "<ul>", "<ol>"]
    for tag in html_tags:
        if tag in content.lower():
            warnings.append(
                f"HTML tag {tag} found in Markdown content. "
                "Consider using Markdown syntax instead."
            )
            break  # Only warn once about HTML

    # Check for very long lines in code blocks (may cause display issues)
    lines = content.split("\n")
    in_code_block = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        elif in_code_block and len(line) > 200:
            warnings.append(
                f"Line {i} in code block exceeds 200 characters. "
                "Consider breaking into multiple lines."
            )

    return warnings


def validate_apf_markdown(apf_content: str) -> MarkdownValidationResult:
    """
    Validate APF document with Markdown content.

    Validates both APF structure (sentinels, headers) and Markdown content.

    Args:
        apf_content: Full APF document with Markdown content

    Returns:
        MarkdownValidationResult with validation status and issues
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not apf_content or not apf_content.strip():
        return MarkdownValidationResult(
            is_valid=False,
            errors=["Empty APF content"],
        )

    # Check for APF structure (sentinels)
    if "<!-- BEGIN_CARDS -->" not in apf_content:
        errors.append("Missing BEGIN_CARDS sentinel")

    if "<!-- END_CARDS -->" not in apf_content:
        errors.append("Missing END_CARDS sentinel")

    # Check for card header
    card_header_pattern = r"<!-- Card \d+ \| slug: [a-z0-9-]+ \| CardType: \w+"
    if not re.search(card_header_pattern, apf_content):
        errors.append("Missing or invalid card header")

    # Check for required sections
    if "<!-- Title -->" not in apf_content:
        errors.append("Missing Title section")

    # At least one of Key point or Key point notes should exist
    has_key_point = "<!-- Key point" in apf_content
    has_key_notes = "<!-- Key point notes -->" in apf_content
    if not has_key_point and not has_key_notes:
        warnings.append(
            "Missing Key point sections. "
            "Cards should have Key point or Key point notes."
        )

    # Validate Markdown content within sections
    # Extract content between APF markers
    content_pattern = r"<!--[^>]+-->\s*([^<]+?)(?=<!--|$)"
    for match in re.finditer(content_pattern, apf_content, re.DOTALL):
        section_content = match.group(1).strip()
        if section_content:
            md_result = validate_markdown(section_content)
            errors.extend(md_result.errors)
            warnings.extend(md_result.warnings)

    return MarkdownValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
