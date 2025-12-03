"""Tests for content hash computation."""

from datetime import UTC, datetime, timezone

import pytest

from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.content_hash import compute_content_hash


@pytest.fixture
def sample_qa_pair():
    """Sample QAPair for content hash tests."""
    return QAPair(
        card_index=1,
        question_en="What is dependency injection?",
        question_ru="Chto takoe vnedrenie zavisimostey?",
        answer_en="Dependency injection is a design pattern.",
        answer_ru="Vnedrenie zavisimostey - eto pattern proektirovaniya.",
        followups="How does DI relate to SOLID?",
        references="Martin Fowler - IoC Containers",
        related="",
        context="",
    )


@pytest.fixture
def sample_metadata():
    """Sample NoteMetadata for content hash tests."""
    return NoteMetadata(
        id="test-001",
        title="Test Note",
        topic="testing",
        language_tags=["en", "ru"],
        subtopics=["unit_testing"],
        tags=["test"],
        created=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestContentHash:
    """Validate that content hash captures all relevant sections."""

    def test_hash_changes_with_followups(self, sample_qa_pair, sample_metadata) -> None:
        """Modifying follow-ups should change the hash."""
        hash_original = compute_content_hash(sample_qa_pair, sample_metadata, "en")

        modified_pair = sample_qa_pair.model_copy(
            update={"followups": "New follow-up prompt"}
        )
        hash_modified = compute_content_hash(modified_pair, sample_metadata, "en")

        assert hash_original != hash_modified

    def test_hash_changes_with_references(
        self, sample_qa_pair, sample_metadata
    ) -> None:
        """References contribute to hash."""
        hash_original = compute_content_hash(sample_qa_pair, sample_metadata, "en")

        modified_pair = sample_qa_pair.model_copy(
            update={"references": "https://example.com"}
        )
        hash_modified = compute_content_hash(modified_pair, sample_metadata, "en")

        assert hash_original != hash_modified

    def test_hash_differs_by_language(self, sample_qa_pair, sample_metadata) -> None:
        """Different language surfaces should yield distinct hashes."""
        hash_en = compute_content_hash(sample_qa_pair, sample_metadata, "en")
        hash_ru = compute_content_hash(sample_qa_pair, sample_metadata, "ru")

        assert hash_en != hash_ru
