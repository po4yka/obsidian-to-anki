"""Pytest configuration and fixtures for the test suite."""

import pytest
from pathlib import Path

from obsidian_anki_sync.domain.entities.card import Card, CardManifest
from obsidian_anki_sync.domain.entities.note import Note, NoteMetadata, QAPair
from tests.fixtures import (
    MockAnkiClient,
    MockCardGenerator,
    MockLLMProvider,
    MockNoteParser,
    MockStateRepository,
)


@pytest.fixture
def mock_anki_client():
    """Provide a mock Anki client for testing."""
    return MockAnkiClient()


@pytest.fixture
def mock_llm_provider():
    """Provide a mock LLM provider for testing."""
    return MockLLMProvider()


@pytest.fixture
def mock_state_repository():
    """Provide a mock state repository for testing."""
    return MockStateRepository()


@pytest.fixture
def mock_card_generator():
    """Provide a mock card generator for testing."""
    return MockCardGenerator()


@pytest.fixture
def mock_note_parser():
    """Provide a mock note parser for testing."""
    return MockNoteParser()


@pytest.fixture
def sample_note_metadata():
    """Provide sample note metadata for testing."""
    return NoteMetadata(
        topic="Test Topic",
        language_tags=["en", "ru"],
        difficulty="medium",
        question_kind="concept",
        tags=["test", "sample"],
        status="published",
    )


@pytest.fixture
def sample_qa_pair():
    """Provide a sample Q&A pair for testing."""
    return QAPair(
        card_index=1,
        question_en="What is dependency injection?",
        question_ru="Что такое внедрение зависимостей?",
        answer_en="Dependency injection is a design pattern where dependencies are provided to a class rather than created internally.",
        answer_ru="Внедрение зависимостей - это паттерн проектирования, при котором зависимости предоставляются классу извне, а не создаются внутри.",
        followups="How does DI relate to SOLID principles?",
        references="Martin Fowler - Inversion of Control Containers",
    )


@pytest.fixture
def sample_note(sample_note_metadata, tmp_path):
    """Provide a sample note entity for testing."""
    file_path = tmp_path / "sample_note.md"
    file_path.write_text("# Sample Note\n\nThis is a test note.")

    return Note(
        id="test-note-123",
        title="Sample Note",
        content="# Sample Note\n\nThis is a test note.",
        file_path=file_path,
        metadata=sample_note_metadata,
        created_at=sample_note_metadata.created,
        updated_at=sample_note_metadata.updated,
    )


@pytest.fixture
def sample_card(sample_note):
    """Provide a sample card entity for testing."""
    manifest = CardManifest(
        slug="test-note-123-en",
        slug_base="test-note-123",
        lang="en",
        source_path=str(sample_note.file_path),
        source_anchor="p1",
        note_id=sample_note.id,
        note_title=sample_note.title,
        card_index=1,
    )

    return Card(
        slug="test-note-123-en",
        language="en",
        apf_html='<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->\n# Question\nTest?\n# Answer\nAnswer\n<!-- END_CARDS -->',
        manifest=manifest,
        note_type="APF::Simple",
        tags=["test"],
    )


@pytest.fixture
def temp_vault_dir(tmp_path):
    """Provide a temporary directory structure mimicking an Obsidian vault."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    # Create some sample note files
    note1 = vault_dir / "note1.md"
    note1.write_text("""---
id: note1
title: Note 1
topic: Testing
language_tags: [en, ru]
---

# Question (EN)
What is testing?

# Answer (EN)
Testing is verifying code behavior.

# Question (RU)
Что такое тестирование?

# Answer (RU)
Тестирование - это проверка поведения кода.
""")

    note2 = vault_dir / "note2.md"
    note2.write_text("""---
id: note2
title: Note 2
topic: Development
language_tags: [en]
---

# Question (EN)
What is development?

# Answer (EN)
Development is writing code.
""")

    return vault_dir


@pytest.fixture(autouse=True)
def reset_mocks(mock_anki_client, mock_llm_provider, mock_state_repository, mock_card_generator, mock_note_parser):
    """Automatically reset all mocks before each test."""
    mock_anki_client.reset()
    mock_llm_provider.reset()
    mock_state_repository.reset()
    mock_card_generator.reset()
    mock_note_parser.reset()
