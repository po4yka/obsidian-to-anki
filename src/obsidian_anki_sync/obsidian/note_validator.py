"""Static validators for Obsidian note content."""

from __future__ import annotations

from collections.abc import Iterable

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
                
                    f"{title or 'Note'}: Missing '{question_header}' section "
                    f"required for language '{lang}'."
                
            )

        if answer_header not in normalized_content:
            errors.append(
                
                    f"{title or 'Note'}: Missing '{answer_header}' section "
                    f"required for language '{lang}'."
                
            )

    return errors


def _validate_code_fences(content: str, title: str | None) -> list[str]:
    """
    Detect unbalanced triple backtick fences with detailed error reporting.

    Tracks fence positions to identify which specific fence is unmatched
    and provides line numbers for better error messages.

    Logic:
    - Use a simple stack-based approach to track opened/closed fences
    - Any ``` that doesn't have a corresponding match is an error
    - This is more lenient and should handle most valid markdown patterns
    """

    errors: list[str] = []
    lines = content.splitlines()

    # Simple stack-based validation
    fence_stack: list[tuple[int, str]] = []  # (line_num, content)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            line_num = idx + 1

            if fence_stack:
                # If we have an opener on stack, this closes it
                opener_line, opener_content = fence_stack.pop()
                # Could log successful matches here if needed
            else:
                # No opener on stack, so this is an opener (or standalone)
                fence_stack.append((line_num, stripped))

    # Any remaining fences on stack are unmatched openers
    if fence_stack:
        first_unmatched_line, first_unmatched_content = fence_stack[0]
        if len(fence_stack) == 1:
            errors.append(
                
                    f"{title or 'Note'}: Unbalanced code fence detected. "
                    f"Fence at line {first_unmatched_line} ({first_unmatched_content!r}) "
                    f"does not have a matching closer. "
                    f"Ensure every ``` opener has a matching closer."
                
            )
        else:
            errors.append(
                
                    f"{title or 'Note'}: {len(fence_stack)} unbalanced code fences detected. "
                    f"First unmatched fence at line {first_unmatched_line} ({first_unmatched_content!r}). "
                    f"Ensure every ``` opener has a matching closer."
                
            )

    return errors
