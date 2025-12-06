"""Validation helpers for parser repair."""

from __future__ import annotations

import re
from pathlib import Path


def validate_obsidian_markdown(content: str) -> list[str]:
    """Validate Obsidian markdown conventions."""
    errors = []

    if not content.strip().startswith("---"):
        errors.append("Missing YAML frontmatter start delimiter (---)")

    delimiter_count = content.count("---")
    if delimiter_count < 2:
        errors.append("Missing YAML frontmatter end delimiter (---)")

    lines = content.split("\n")
    has_question_header = False
    has_answer_header = False

    for line in lines:
        if line.strip().startswith("# Question") or line.strip().startswith("# Вопрос"):
            has_question_header = True
        if line.strip().startswith("## Answer") or line.strip().startswith("## Ответ"):
            has_answer_header = True

    if not has_question_header and not has_answer_header and "---" in content:
        errors.append("Missing question/answer headers")

    return errors


def validate_bilingual_consistency(content: str) -> list[str]:
    """Validate bilingual consistency (EN/RU)."""
    errors = []

    has_en_question = "# Question (EN)" in content or "# Question" in content
    has_ru_question = "# Вопрос (RU)" in content or "# Вопрос" in content
    has_en_answer = "## Answer (EN)" in content or "## Answer" in content
    has_ru_answer = "## Ответ (RU)" in content or "## Ответ" in content

    lang_tags_match = re.search(r"language_tags:\s*\[(.*?)\]", content)
    if lang_tags_match:
        lang_tags_str = lang_tags_match.group(1)
        lang_tags = [
            tag.strip().strip('"').strip("'") for tag in lang_tags_str.split(",")
        ]

        if "en" in lang_tags and not (has_en_question and has_en_answer):
            errors.append("Language tag 'en' present but EN content missing")

        if "ru" in lang_tags and not (has_ru_question and has_ru_answer):
            errors.append("Language tag 'ru' present but RU content missing")

    return errors


def validate_frontmatter_structure(content: str) -> list[str]:
    """Validate frontmatter structure."""
    errors = []

    frontmatter_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        errors.append("Invalid or missing YAML frontmatter")
        return errors

    frontmatter = frontmatter_match.group(1)

    required_fields = [
        "id",
        "title",
        "topic",
        "language_tags",
        "created",
        "updated",
    ]
    for field in required_fields:
        if f"{field}:" not in frontmatter:
            errors.append(f"Missing required frontmatter field: {field}")

    lang_tags_match = re.search(r"language_tags:\s*\[(.*?)\]", frontmatter)
    if lang_tags_match:
        lang_tags_str = lang_tags_match.group(1)
        lang_tags = [
            tag.strip().strip('"').strip("'") for tag in lang_tags_str.split(",")
        ]
        valid_langs = {"en", "ru"}
        for tag in lang_tags:
            if tag not in valid_langs:
                errors.append(f"Invalid language tag: {tag} (must be 'en' or 'ru')")

    return errors


def validate_repaired_content(content: str, file_path: Path | None = None) -> list[str]:
    """Validate repaired content against APF and Obsidian requirements."""
    errors = []
    errors.extend(validate_obsidian_markdown(content))
    errors.extend(validate_bilingual_consistency(content))
    errors.extend(validate_frontmatter_structure(content))
    return errors

