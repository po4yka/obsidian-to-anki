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
    Detect unbalanced triple backtick fences. Lightweight but effective for truncated code.
    """

    errors: list[str] = []
    fence_lines = [
        (idx + 1, line.strip())
        for idx, line in enumerate(content.splitlines())
        if line.strip().startswith("```")
    ]

    if len(fence_lines) % 2 != 0:
        first_unmatched_line = fence_lines[-1][0] if fence_lines else "unknown"
        errors.append(
            (
                f"{title or 'Note'}: Unbalanced code fences detected "
                f"(line {first_unmatched_line}). Ensure every ``` opener "
                "has a matching closer."
            )
        )

    return errors

