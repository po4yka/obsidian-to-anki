"""Static validators for Obsidian note content."""

from __future__ import annotations

from typing import Iterable

from ..models import NoteMetadata

LANGUAGE_SECTIONS = {
    "en": {"question": "# Question (EN)", "answer": "## Answer (EN)"},
    "ru": {"question": "# Вопрос (RU)", "answer": "## Ответ (RU)"},
}


def validate_note_structure(
    metadata: NoteMetadata,
    content: str,
    enforce_languages: bool = True,
    check_code_fences: bool = True,
) -> list[str]:
    """
    Validate that note content satisfies bilingual and formatting requirements.

    Args:
        metadata: Parsed note metadata.
        content: Full note markdown (including frontmatter).
        enforce_languages: Require complete Q/A sections for each language.
        check_code_fences: Ensure code fences are balanced.

    Returns:
        List of human-readable validation errors (empty if valid).
    """

    errors: list[str] = []

    if enforce_languages:
        errors.extend(
            _validate_language_sections(
                languages=metadata.language_tags, content=content, title=metadata.title
            )
        )

    if check_code_fences:
        errors.extend(_validate_code_fences(content, metadata.title))

    return errors


def _validate_language_sections(
    languages: Iterable[str], content: str, title: str | None
) -> list[str]:
    errors: list[str] = []
    normalized_content = content  # keep case sensitivity for Cyrillic

    for lang in languages:
        markers = LANGUAGE_SECTIONS.get(lang.lower())
        if not markers:
            continue

        question_header = markers["question"]
        answer_header = markers["answer"]

        if question_header not in normalized_content:
            errors.append(
                (
                    f"{title or 'Note'}: Missing '{question_header}' section "
                    f"required for language '{lang}'."
                )
            )

        if answer_header not in normalized_content:
            errors.append(
                (
                    f"{title or 'Note'}: Missing '{answer_header}' section "
                    f"required for language '{lang}'."
                )
            )

    return errors


def _validate_code_fences(content: str, title: str | None) -> list[str]:
    """
    Detect unbalanced triple backtick fences with detailed error reporting.

    Tracks fence positions to identify which specific fence is unmatched
    and provides line numbers for better error messages.
    """

    errors: list[str] = []
    lines = content.splitlines()
    fence_lines: list[tuple[int, str, bool]] = []  # (line_num, content, is_opener)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            # Check if this is an opener (has language identifier) or closer
            # Openers typically have: ```language or ```language:something
            # Closers are just: ```
            is_opener = len(stripped) > 3 and not stripped[3:].strip() == ""
            fence_lines.append((idx + 1, stripped, is_opener))

    if not fence_lines:
        return errors

    # Track open/close pairs using a stack
    unmatched_openers: list[tuple[int, str]] = []
    unmatched_closers: list[tuple[int, str]] = []

    for line_num, fence_content, is_opener in fence_lines:
        if is_opener:
            unmatched_openers.append((line_num, fence_content))
        else:
            if unmatched_openers:
                # Matched pair - remove the opener
                unmatched_openers.pop()
            else:
                # Closer without matching opener
                unmatched_closers.append((line_num, fence_content))

    # Report errors
    if unmatched_openers:
        # Find the first unmatched opener for detailed error
        first_unmatched_line, first_unmatched_content = unmatched_openers[0]
        language_hint = ""
        if len(first_unmatched_content) > 3:
            lang = first_unmatched_content[3:].strip().split()[0] if first_unmatched_content[3:].strip() else ""
            if lang:
                language_hint = f" (language: {lang})"

        if len(unmatched_openers) == 1:
            errors.append(
                (
                    f"{title or 'Note'}: Unbalanced code fence detected "
                    f"(line {first_unmatched_line}{language_hint}). "
                    f"Ensure every ``` opener has a matching closer."
                )
            )
        else:
            errors.append(
                (
                    f"{title or 'Note'}: {len(unmatched_openers)} unbalanced code fences detected. "
                    f"First unmatched fence at line {first_unmatched_line}{language_hint}. "
                    f"Ensure every ``` opener has a matching closer."
                )
            )

    if unmatched_closers:
        # This is less common but indicates malformed markdown
        first_unmatched_closer_line = unmatched_closers[0][0]
        errors.append(
            (
                f"{title or 'Note'}: Code fence closer without matching opener "
                f"detected at line {first_unmatched_closer_line}. "
                f"This may indicate malformed markdown."
            )
        )

    return errors

