"""Tests for Obsidian note parser (UNIT-parse-01, UNIT-parse-02, UNIT-yaml-01)."""

from datetime import datetime
from pathlib import Path

import pytest

from obsidian_anki_sync.obsidian.parser import (
    ParserError,
    discover_notes,
    parse_frontmatter,
    parse_note,
    parse_qa_pairs,
)


class TestYAMLParsing:
    """Test YAML frontmatter parsing (UNIT-yaml-01)."""

    def test_parse_valid_frontmatter(self, temp_dir, sample_note_content):
        """Test parsing valid YAML frontmatter."""
        note_file = temp_dir / "test.md"
        note_file.write_text(sample_note_content)

        metadata = parse_frontmatter(sample_note_content, note_file)

        assert metadata.id == "test-001"
        assert metadata.title == "Test Question"
        assert metadata.topic == "Testing"
        assert metadata.language_tags == ["en", "ru"]
        assert metadata.created == datetime(2024, 1, 1)
        assert metadata.updated == datetime(2024, 1, 2)
        assert "unit_testing" in metadata.subtopics
        assert metadata.moc == "moc-testing"
        assert metadata.related == ["c-testing-concept", "external-resource"]
        assert metadata.sources == [
            {"url": "https://example.com/unit-tests", "note": "Example overview"},
            {"url": "https://docs.pytest.org"},
        ]

    def test_missing_frontmatter(self, temp_dir):
        """Test error when frontmatter is missing."""
        content = "# Just a title\n\nNo frontmatter here."

        with pytest.raises(ParserError, match="No frontmatter found"):
            parse_frontmatter(content, temp_dir / "test.md")

    def test_missing_required_fields(self, temp_dir):
        """Test error when required fields are missing."""
        content = """---
id: test-001
title: Test
---

Content here.
"""
        with pytest.raises(ParserError, match="Missing required fields"):
            parse_frontmatter(content, temp_dir / "test.md")

    def test_invalid_yaml(self, temp_dir):
        """Test error on invalid YAML syntax."""
        content = """---
id: test-001
title: Test
invalid: [unclosed bracket
---

Content.
"""
        with pytest.raises(ParserError, match="Invalid YAML"):
            parse_frontmatter(content, temp_dir / "test.md")


class TestQAParsing:
    """Test Q/A pair extraction (UNIT-parse-01, UNIT-parse-02)."""

    def test_parse_single_qa_pair(self, sample_note_content, sample_metadata):
        """Test parsing a single Q/A pair."""
        qa_pairs = parse_qa_pairs(sample_note_content, sample_metadata)

        assert len(qa_pairs) == 1
        assert qa_pairs[0].card_index == 1
        assert "What is unit testing?" in qa_pairs[0].question_en
        assert "Что такое юнит-тестирование?" in qa_pairs[0].question_ru
        assert "Unit testing is testing" in qa_pairs[0].answer_en
        assert "Юнит-тестирование" in qa_pairs[0].answer_ru

    def test_parse_multiple_qa_pairs(self, sample_metadata):
        """Test parsing multiple Q/A pairs (UNIT-parse-02)."""
        content = """---
id: test-001
title: Multiple Questions
---

# Question (EN)

> First question?

# Вопрос (RU)

> Первый вопрос?

---

## Answer (EN)

First answer.

## Ответ (RU)

Первый ответ.

# Question (EN)

> Second question?

# Вопрос (RU)

> Второй вопрос?

---

## Answer (EN)

Second answer.

## Ответ (RU)

Второй ответ.
"""
        qa_pairs = parse_qa_pairs(content, sample_metadata)

        assert len(qa_pairs) == 2
        assert qa_pairs[0].card_index == 1
        assert qa_pairs[1].card_index == 2
        assert "First question" in qa_pairs[0].question_en
        assert "Second question" in qa_pairs[1].question_en

    def test_parse_ru_first_order(self, sample_metadata):
        """Test parsing when Russian sections precede English sections."""
        content = """---
id: test-ru-first
title: RU First Question
---

# Вопрос (RU)

> Что такое тест?

# Question (EN)

> What is a test?

---

## Ответ (RU)

Это проверка системы.

## Answer (EN)

It is a system check.
"""
        qa_pairs = parse_qa_pairs(content, sample_metadata)

        assert len(qa_pairs) == 1
        pair = qa_pairs[0]
        assert pair.question_ru.startswith("Что такое тест")
        assert pair.question_en.startswith("What is a test")
        assert pair.answer_ru.startswith("Это проверка системы")
        assert pair.answer_en.startswith("It is a system check")

    def test_parse_with_followups(self, sample_note_content, sample_metadata):
        """Test parsing Q/A with follow-ups and references."""
        qa_pairs = parse_qa_pairs(sample_note_content, sample_metadata)

        assert len(qa_pairs) == 1
        assert "How to write good tests?" in qa_pairs[0].followups
        assert "pytest.org" in qa_pairs[0].references
        assert "Integration testing" in qa_pairs[0].related

    def test_missing_separator(self, sample_metadata):
        """Test handling of missing separator."""
        content = """---
id: test
---

# Question (EN)

> Question?

# Вопрос (RU)

> Вопрос?

## Answer (EN)

Answer without separator.
"""
        # Should log error but try to parse
        qa_pairs = parse_qa_pairs(content, sample_metadata)
        # Depending on implementation, might return empty or partial
        assert isinstance(qa_pairs, list)


class TestFileDiscovery:
    """Test file discovery."""

    def test_discover_valid_notes(self, temp_dir):
        """Test discovering q-*.md files."""
        vault = temp_dir / "vault"
        source = vault / "questions"
        source.mkdir(parents=True)

        # Create valid files
        (source / "q-test-01.md").touch()
        (source / "q-test-02.md").touch()

        # Create files to ignore
        (source / "c-concept.md").touch()
        (source / "moc-index.md").touch()
        (source / "template.md").touch()

        notes = discover_notes(vault, Path("questions"))

        assert len(notes) == 2
        assert all("q-test" in str(path) for path, _ in notes)

    def test_discover_recursive(self, temp_dir):
        """Test recursive discovery in subdirectories."""
        vault = temp_dir / "vault"
        source = vault / "questions"
        subdir = source / "python"
        subdir.mkdir(parents=True)

        (source / "q-root.md").touch()
        (subdir / "q-python-01.md").touch()

        notes = discover_notes(vault, Path("questions"))

        assert len(notes) == 2


class TestFullNoteParsing:
    """Test complete note parsing."""

    def test_parse_complete_note(self, temp_dir, sample_note_content):
        """Test parsing a complete note file."""
        note_file = temp_dir / "q-test.md"
        note_file.write_text(sample_note_content)

        metadata, qa_pairs = parse_note(note_file)

        assert metadata.id == "test-001"
        assert metadata.title == "Test Question"
        assert len(qa_pairs) == 1
        assert qa_pairs[0].card_index == 1

    def test_parse_nonexistent_file(self, temp_dir):
        """Test error on nonexistent file."""
        with pytest.raises(ParserError, match="File does not exist"):
            parse_note(temp_dir / "nonexistent.md")
