"""Tests for note structure validator."""

from datetime import datetime, timezone

import pytest

from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.obsidian.note_validator import validate_note_structure
from obsidian_anki_sync.obsidian.parser import parse_frontmatter


@pytest.fixture
def sample_note_content():
    """Sample note content for validator tests."""
    return """---
id: test-001
title: Test Question
topic: Testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Question (EN)

> What is unit testing?

# Вопрос (RU)

> Что такое юнит-тестирование?

---

## Answer (EN)

Unit testing is testing individual components.

- Tests small units of code
- Runs in isolation
- Fast and reliable

## Ответ (RU)

Юнит-тестирование - это тестирование отдельных компонентов.
"""


@pytest.fixture
def sample_metadata():
    """Sample metadata for validator tests."""
    return NoteMetadata(
        id="test-001",
        title="Test Question",
        topic="Testing",
        language_tags=["en", "ru"],
        created=datetime(2024, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )


class TestNoteValidator:
    """Validate bilingual and formatting checks."""

    def test_validator_passes_for_complete_note(
        self, sample_note_content, sample_metadata
    ) -> None:
        """Validator should return no errors for well-formed bilingual notes."""
        errors = validate_note_structure(sample_metadata, sample_note_content)
        assert errors == []

    def test_missing_english_answer_detected(
        self, sample_note_content, sample_metadata
    ) -> None:
        """Missing English answer should be reported."""
        broken_content = sample_note_content.replace(
            "## Answer (EN)\n\nUnit testing is testing individual components.\n\n- Tests small units of code\n- Runs in isolation\n- Fast and reliable\n\n",
            "",
        )

        errors = validate_note_structure(sample_metadata, broken_content)

        assert any("Missing '## Answer (EN)'" in err for err in errors)

    def test_unbalanced_code_fence_detected(
        self, sample_note_content, sample_metadata
    ) -> None:
        """Unbalanced code blocks should be flagged."""
        content_with_code = (
            sample_note_content + "\n```kotlin\nfun missingClosure() {}\n"
        )

        errors = validate_note_structure(sample_metadata, content_with_code)

        assert any("Unbalanced code fence" in err for err in errors)


class TestLintNoteFrontmatter:
    """Ensure parsing helpers cooperate with validator."""

    def test_parse_frontmatter_integration(self, sample_note_content, tmp_path) -> None:
        """Round-trip parse frontmatter and validate."""
        note_path = tmp_path / "note.md"
        note_path.write_text(sample_note_content)

        metadata = parse_frontmatter(sample_note_content, note_path)
        errors = validate_note_structure(metadata, sample_note_content)

        assert errors == []
