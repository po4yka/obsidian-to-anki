"""Tests for indexing system."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from obsidian_anki_sync.sync.state_db import StateDB


class TestNoteIndex:
    """Test note index functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_upsert_note_index(self, temp_db):
        """Test inserting and updating note index."""
        # Insert note
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Test Note",
            topic="Testing",
            language_tags=["en", "ru"],
            qa_pair_count=3,
            file_modified_at=datetime.now(),
            metadata_json='{"test": true}',
        )

        # Retrieve note
        note = temp_db.get_note_index("notes/test.md")
        assert note is not None
        assert note["note_id"] == "note1"
        assert note["note_title"] == "Test Note"
        assert note["topic"] == "Testing"
        assert note["language_tags"] == "en,ru"
        assert note["qa_pair_count"] == 3

    def test_upsert_note_index_update(self, temp_db):
        """Test updating existing note index."""
        # Insert initial
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Original Title",
            topic="Topic1",
            language_tags=["en"],
            qa_pair_count=2,
            file_modified_at=datetime.now(),
        )

        # Update
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Updated Title",
            topic="Topic2",
            language_tags=["en", "ru"],
            qa_pair_count=3,
            file_modified_at=datetime.now(),
        )

        # Should have one record with updated values
        note = temp_db.get_note_index("notes/test.md")
        assert note["note_title"] == "Updated Title"
        assert note["topic"] == "Topic2"
        assert note["qa_pair_count"] == 3

    def test_update_note_sync_status(self, temp_db):
        """Test updating note sync status."""
        # Insert note
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Test",
            topic="Testing",
            language_tags=["en"],
            qa_pair_count=1,
            file_modified_at=datetime.now(),
        )

        # Update status
        temp_db.update_note_sync_status(
            "notes/test.md", "completed", error_message=None
        )

        note = temp_db.get_note_index("notes/test.md")
        assert note["sync_status"] == "completed"
        assert note["error_message"] is None
        assert note["last_synced_at"] is not None

    def test_update_note_sync_status_with_error(self, temp_db):
        """Test updating note sync status with error."""
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Test",
            topic="Testing",
            language_tags=["en"],
            qa_pair_count=1,
            file_modified_at=datetime.now(),
        )

        temp_db.update_note_sync_status(
            "notes/test.md", "failed", error_message="Parse error"
        )

        note = temp_db.get_note_index("notes/test.md")
        assert note["sync_status"] == "failed"
        assert note["error_message"] == "Parse error"

    def test_get_all_notes_index(self, temp_db):
        """Test getting all notes from index."""
        # Insert multiple notes
        for i in range(5):
            temp_db.upsert_note_index(
                source_path=f"notes/note{i}.md",
                note_id=f"note{i}",
                note_title=f"Note {i}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )

        notes = temp_db.get_all_notes_index()
        assert len(notes) == 5

    def test_get_notes_by_status(self, temp_db):
        """Test filtering notes by status."""
        # Insert notes with different statuses
        for i in range(3):
            temp_db.upsert_note_index(
                source_path=f"notes/completed{i}.md",
                note_id=f"note{i}",
                note_title=f"Note {i}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )
            temp_db.update_note_sync_status(f"notes/completed{i}.md", "completed")

        for i in range(2):
            temp_db.upsert_note_index(
                source_path=f"notes/failed{i}.md",
                note_id=f"note{i + 3}",
                note_title=f"Note {i + 3}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )
            temp_db.update_note_sync_status(f"notes/failed{i}.md", "failed")

        completed = temp_db.get_notes_by_status("completed")
        failed = temp_db.get_notes_by_status("failed")

        assert len(completed) == 3
        assert len(failed) == 2


class TestCardIndex:
    """Test card index functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_upsert_card_index(self, temp_db):
        """Test inserting and updating card index."""
        # Insert card
        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="test-1-en",
            note_id="note1",
            note_title="Test Note",
            status="expected",
            in_obsidian=True,
            in_anki=False,
            in_database=False,
        )

        # Retrieve card
        card = temp_db.get_card_index_by_slug("test-1-en")
        assert card is not None
        assert card["source_path"] == "notes/test.md"
        assert card["card_index"] == 1
        assert card["lang"] == "en"
        assert card["status"] == "expected"
        assert card["in_obsidian"] == 1
        assert card["in_anki"] == 0

    def test_upsert_card_index_update(self, temp_db):
        """Test updating existing card index."""
        # Insert
        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="test-1-en",
            status="expected",
            in_obsidian=True,
            in_anki=False,
            in_database=False,
        )

        # Update
        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="test-1-en",
            anki_guid=1001,
            status="synced",
            in_obsidian=True,
            in_anki=True,
            in_database=True,
        )

        card = temp_db.get_card_index_by_slug("test-1-en")
        assert card["anki_guid"] == 1001
        assert card["status"] == "synced"
        assert card["in_anki"] == 1
        assert card["in_database"] == 1

    def test_get_card_index_by_source(self, temp_db):
        """Test getting all cards for a note."""
        # Insert multiple cards for one note
        for i in range(3):
            for lang in ["en", "ru"]:
                temp_db.upsert_card_index(
                    source_path="notes/test.md",
                    card_index=i + 1,
                    lang=lang,
                    slug=f"test-{i + 1}-{lang}",
                    in_obsidian=True,
                )

        cards = temp_db.get_card_index_by_source("notes/test.md")
        assert len(cards) == 6  # 3 Q/A pairs × 2 languages

    def test_card_index_unique_constraint(self, temp_db):
        """Test unique constraint on (source_path, card_index, lang)."""
        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="slug1",
        )

        # Upserting with same key but different slug should update
        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="slug2",
        )

        cards = temp_db.get_card_index_by_source("notes/test.md")
        assert len(cards) == 1
        assert cards[0]["slug"] == "slug2"


class TestIndexStatistics:
    """Test index statistics functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_get_index_statistics_empty(self, temp_db):
        """Test statistics on empty index."""
        stats = temp_db.get_index_statistics()

        assert stats["total_notes"] == 0
        assert stats["total_cards"] == 0
        assert stats["cards_in_obsidian"] == 0
        assert stats["cards_in_anki"] == 0
        assert stats["cards_in_database"] == 0

    def test_get_index_statistics_with_data(self, temp_db):
        """Test statistics with data."""
        # Add notes
        for i in range(5):
            temp_db.upsert_note_index(
                source_path=f"notes/note{i}.md",
                note_id=f"note{i}",
                note_title=f"Note {i}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )

        # Add cards with different states
        temp_db.upsert_card_index(
            source_path="notes/note0.md",
            card_index=1,
            lang="en",
            slug="slug1",
            in_obsidian=True,
            in_anki=False,
            in_database=False,
        )

        temp_db.upsert_card_index(
            source_path="notes/note1.md",
            card_index=1,
            lang="en",
            slug="slug2",
            in_obsidian=True,
            in_anki=True,
            in_database=True,
        )

        temp_db.upsert_card_index(
            source_path="notes/note2.md",
            card_index=1,
            lang="en",
            slug="slug3",
            in_obsidian=False,
            in_anki=True,
            in_database=False,
        )

        stats = temp_db.get_index_statistics()

        assert stats["total_notes"] == 5
        assert stats["total_cards"] == 3
        assert stats["cards_in_obsidian"] == 2
        assert stats["cards_in_anki"] == 2
        assert stats["cards_in_database"] == 1

    def test_note_status_breakdown(self, temp_db):
        """Test note status breakdown in statistics."""
        # Add notes with different statuses
        for i in range(3):
            temp_db.upsert_note_index(
                source_path=f"notes/completed{i}.md",
                note_id=f"note{i}",
                note_title=f"Note {i}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )
            temp_db.update_note_sync_status(f"notes/completed{i}.md", "completed")

        for i in range(2):
            temp_db.upsert_note_index(
                source_path=f"notes/pending{i}.md",
                note_id=f"note{i + 3}",
                note_title=f"Note {i + 3}",
                topic="Testing",
                language_tags=["en"],
                qa_pair_count=1,
                file_modified_at=datetime.now(),
            )

        stats = temp_db.get_index_statistics()

        assert stats["note_status"]["completed"] == 3
        assert stats["note_status"]["pending"] == 2

    def test_card_status_breakdown(self, temp_db):
        """Test card status breakdown in statistics."""
        temp_db.upsert_card_index(
            source_path="note1.md",
            card_index=1,
            lang="en",
            slug="slug1",
            status="expected",
        )

        temp_db.upsert_card_index(
            source_path="note2.md",
            card_index=1,
            lang="en",
            slug="slug2",
            status="synced",
        )

        temp_db.upsert_card_index(
            source_path="note3.md",
            card_index=1,
            lang="en",
            slug="slug3",
            status="orphaned",
        )

        stats = temp_db.get_index_statistics()

        assert stats["card_status"]["expected"] == 1
        assert stats["card_status"]["synced"] == 1
        assert stats["card_status"]["orphaned"] == 1

    def test_clear_index(self, temp_db):
        """Test clearing index."""
        # Add data
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Test",
            topic="Testing",
            language_tags=["en"],
            qa_pair_count=1,
            file_modified_at=datetime.now(),
        )

        temp_db.upsert_card_index(
            source_path="notes/test.md",
            card_index=1,
            lang="en",
            slug="slug1",
        )

        # Verify data exists
        stats_before = temp_db.get_index_statistics()
        assert stats_before["total_notes"] > 0
        assert stats_before["total_cards"] > 0

        # Clear index
        temp_db.clear_index()

        # Verify cleared
        stats_after = temp_db.get_index_statistics()
        assert stats_after["total_notes"] == 0
        assert stats_after["total_cards"] == 0


class TestIndexIntegration:
    """Test integration between note and card indexes."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_note_with_multiple_cards(self, temp_db):
        """Test indexing note with multiple cards."""
        # Index note
        temp_db.upsert_note_index(
            source_path="notes/test.md",
            note_id="note1",
            note_title="Test Note",
            topic="Testing",
            language_tags=["en", "ru"],
            qa_pair_count=3,
            file_modified_at=datetime.now(),
        )

        # Index cards (3 Q/A × 2 languages = 6 cards)
        for i in range(3):
            for lang in ["en", "ru"]:
                temp_db.upsert_card_index(
                    source_path="notes/test.md",
                    card_index=i + 1,
                    lang=lang,
                    slug=f"test-{i + 1}-{lang}",
                    note_id="note1",
                    note_title="Test Note",
                )

        # Verify
        note = temp_db.get_note_index("notes/test.md")
        cards = temp_db.get_card_index_by_source("notes/test.md")

        assert note["qa_pair_count"] == 3
        assert len(cards) == 6

    def test_orphaned_card_detection(self, temp_db):
        """Test detecting orphaned cards."""
        # Card exists in Anki but not in vault
        temp_db.upsert_card_index(
            source_path="notes/deleted.md",
            card_index=1,
            lang="en",
            slug="orphaned-1-en",
            anki_guid=1001,
            status="orphaned",
            in_obsidian=False,
            in_anki=True,
            in_database=True,
        )

        stats = temp_db.get_index_statistics()
        assert stats["card_status"]["orphaned"] == 1
        assert stats["cards_in_anki"] == 1
        assert stats["cards_in_obsidian"] == 0
