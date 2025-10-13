"""Pytest configuration and fixtures."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.models import NoteMetadata, QAPair


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir()
    source_dir = vault_path / "interview_questions" / "InterviewQuestions"
    source_dir.mkdir(parents=True)

    return Config(
        vault_path=vault_path,
        source_dir=Path("interview_questions/InterviewQuestions"),
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Test Deck",
        anki_note_type="APF::Simple",
        openrouter_api_key="test-key",
        openrouter_model="openai/gpt-4",
        llm_temperature=0.2,
        llm_top_p=0.3,
        run_mode="apply",
        delete_mode="delete",
        db_path=temp_dir / "test.db",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_metadata():
    """Create sample note metadata."""
    return NoteMetadata(
        id="test-001",
        title="Test Question",
        topic="Testing",
        language_tags=["en", "ru"],
        created=datetime(2024, 1, 1),
        updated=datetime(2024, 1, 2),
        aliases=["test"],
        subtopics=["unit_testing"],
        question_kind="technical",
        difficulty="medium",
        original_language="en",
        tags=["python", "testing"],
    )


@pytest.fixture
def sample_qa_pair():
    """Create sample Q/A pair."""
    return QAPair(
        card_index=1,
        question_en="What is unit testing?",
        question_ru="Что такое юнит-тестирование?",
        answer_en="Unit testing is testing individual components.",
        answer_ru="Юнит-тестирование - это тестирование отдельных компонентов.",
        followups="How to write good tests?",
        references="pytest.org",
        related="Integration testing",
        context="Testing fundamentals",
    )


@pytest.fixture
def sample_note_content():
    """Create sample Obsidian note content."""
    return """---
id: test-001
title: Test Question
topic: Testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
subtopics: [unit_testing]
difficulty: medium
tags: [python, testing]
---

# Introduction

This is a test note.

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

- Тестирует небольшие блоки кода
- Выполняется изолированно
- Быстро и надежно

## Follow-ups

- How to write good tests?
- What is TDD?

## References

- https://pytest.org

## Related Questions

- Integration testing
- End-to-end testing
"""
