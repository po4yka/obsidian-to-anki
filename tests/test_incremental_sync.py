"""Tests for incremental sync functionality."""

import tempfile
from pathlib import Path

import pytest

from obsidian_anki_sync.models import Card, Manifest
from obsidian_anki_sync.sync.state_db import StateDB


class TestIncrementalSync:
    """Test incremental sync mode."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_get_processed_note_paths_empty(self, temp_db):
        """Test getting processed paths from empty database."""
        paths = temp_db.get_processed_note_paths()
        assert isinstance(paths, set)
        assert len(paths) == 0

    def test_get_processed_note_paths_with_cards(self, temp_db):
        """Test getting processed paths with cards."""
        # Add test cards
        card1 = self._create_test_card("notes/note1.md", "slug1", 1, "en")
        card2 = self._create_test_card("notes/note1.md", "slug2", 1, "ru")
        card3 = self._create_test_card("notes/note2.md", "slug3", 1, "en")

        temp_db.insert_card(card1, anki_guid=1001)
        temp_db.insert_card(card2, anki_guid=1002)
        temp_db.insert_card(card3, anki_guid=1003)

        # Get processed paths
        paths = temp_db.get_processed_note_paths()

        assert len(paths) == 2
        assert "notes/note1.md" in paths
        assert "notes/note2.md" in paths

    def test_get_processed_note_paths_distinct(self, temp_db):
        """Test that processed paths are distinct."""
        # Add multiple cards from same note
        for i in range(5):
            card = self._create_test_card("notes/same-note.md", f"slug{i}", i + 1, "en")
            temp_db.insert_card(card, anki_guid=1000 + i)

        paths = temp_db.get_processed_note_paths()

        # Should only return one path despite multiple cards
        assert len(paths) == 1
        assert "notes/same-note.md" in paths

    def test_incremental_filtering(self, temp_db):
        """Test filtering notes for incremental sync."""
        # Add some processed notes
        card1 = self._create_test_card("notes/old1.md", "slug1", 1, "en")
        card2 = self._create_test_card("notes/old2.md", "slug2", 1, "en")
        temp_db.insert_card(card1, anki_guid=1001)
        temp_db.insert_card(card2, anki_guid=1002)

        # All notes in vault
        all_notes = [
            ("path1", "notes/old1.md"),  # Already processed
            ("path2", "notes/old2.md"),  # Already processed
            ("path3", "notes/new1.md"),  # New
            ("path4", "notes/new2.md"),  # New
        ]

        # Get processed paths
        processed_paths = temp_db.get_processed_note_paths()

        # Filter for new notes only
        new_notes = [
            (file_path, rel_path)
            for file_path, rel_path in all_notes
            if rel_path not in processed_paths
        ]

        assert len(new_notes) == 2
        assert ("path3", "notes/new1.md") in new_notes
        assert ("path4", "notes/new2.md") in new_notes

    def test_incremental_with_no_new_notes(self, temp_db):
        """Test incremental mode when no new notes exist."""
        # Add all notes to database
        for i in range(3):
            card = self._create_test_card(f"notes/note{i}.md", f"slug{i}", 1, "en")
            temp_db.insert_card(card, anki_guid=1000 + i)

        processed_paths = temp_db.get_processed_note_paths()

        # All notes are already processed
        all_notes = [
            ("path0", "notes/note0.md"),
            ("path1", "notes/note1.md"),
            ("path2", "notes/note2.md"),
        ]

        new_notes = [
            (file_path, rel_path)
            for file_path, rel_path in all_notes
            if rel_path not in processed_paths
        ]

        assert len(new_notes) == 0

    def test_incremental_with_all_new_notes(self, temp_db):
        """Test incremental mode when all notes are new."""
        # Empty database
        processed_paths = temp_db.get_processed_note_paths()
        assert len(processed_paths) == 0

        # All notes are new
        all_notes = [
            ("path1", "notes/new1.md"),
            ("path2", "notes/new2.md"),
            ("path3", "notes/new3.md"),
        ]

        new_notes = [
            (file_path, rel_path)
            for file_path, rel_path in all_notes
            if rel_path not in processed_paths
        ]

        assert len(new_notes) == 3

    def test_incremental_after_card_deletion(self, temp_db):
        """Test incremental mode after card deletion."""
        # Add and then delete a card
        card = self._create_test_card("notes/deleted.md", "slug1", 1, "en")
        temp_db.insert_card(card, anki_guid=1001)

        # Path should be in processed
        paths_before = temp_db.get_processed_note_paths()
        assert "notes/deleted.md" in paths_before

        # Delete card
        temp_db.delete_card("slug1")

        # Path should no longer be in processed
        paths_after = temp_db.get_processed_note_paths()
        assert "notes/deleted.md" not in paths_after

    def test_incremental_with_multilingual_notes(self, temp_db):
        """Test incremental mode with multi-language cards."""
        # Add cards for a note in multiple languages
        card_en = self._create_test_card("notes/multi.md", "slug-en", 1, "en")
        card_ru = self._create_test_card("notes/multi.md", "slug-ru", 1, "ru")
        card_es = self._create_test_card("notes/multi.md", "slug-es", 1, "es")

        temp_db.insert_card(card_en, anki_guid=1001)
        temp_db.insert_card(card_ru, anki_guid=1002)
        temp_db.insert_card(card_es, anki_guid=1003)

        # Should only return one path
        paths = temp_db.get_processed_note_paths()
        assert len(paths) == 1
        assert "notes/multi.md" in paths

    def test_incremental_performance(self, temp_db):
        """Test performance with large number of cards."""
        # Add many cards
        for i in range(100):
            card = self._create_test_card(f"notes/note{i}.md", f"slug{i}", 1, "en")
            temp_db.insert_card(card, anki_guid=1000 + i)

        # Query should be fast (using DISTINCT)
        paths = temp_db.get_processed_note_paths()
        assert len(paths) == 100

    def _create_test_card(
        self, source_path: str, slug: str, card_index: int, lang: str
    ) -> Card:
        """Create a test card."""
        manifest = Manifest(
            slug=slug,
            slug_base=slug.rsplit("-", 1)[0],
            lang=lang,
            source_path=source_path,
            source_anchor=f"#q{card_index}",
            note_id="test-note",
            note_title="Test Note",
            card_index=card_index,
            guid=f"guid-{slug}",
        )

        return Card(
            slug=slug,
            lang=lang,
            apf_html="<div>Test</div>",
            manifest=manifest,
            content_hash="test-hash",
            guid=f"guid-{slug}",
        )


class TestIncrementalSyncStatistics:
    """Test statistics tracking for incremental sync."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_track_processed_count(self, temp_db):
        """Test tracking number of processed notes."""
        # Initially zero
        assert len(temp_db.get_processed_note_paths()) == 0

        # Add cards incrementally
        for i in range(5):
            card = self._create_test_card(f"note{i}.md", f"slug{i}", 1, "en")
            temp_db.insert_card(card, anki_guid=1000 + i)

            paths = temp_db.get_processed_note_paths()
            assert len(paths) == i + 1

    def test_filter_statistics(self, temp_db):
        """Test calculating filter statistics."""
        # Add 3 processed notes
        for i in range(3):
            card = self._create_test_card(f"old{i}.md", f"slug{i}", 1, "en")
            temp_db.insert_card(card, anki_guid=1000 + i)

        # Simulate discovering 5 total notes (3 old + 2 new)
        all_notes = [
            ("p0", "old0.md"),
            ("p1", "old1.md"),
            ("p2", "old2.md"),
            ("p3", "new0.md"),
            ("p4", "new1.md"),
        ]

        processed_paths = temp_db.get_processed_note_paths()
        new_notes = [(fp, rp) for fp, rp in all_notes if rp not in processed_paths]

        total_count = len(all_notes)
        new_count = len(new_notes)
        filtered_count = total_count - new_count

        assert total_count == 5
        assert new_count == 2
        assert filtered_count == 3

    def _create_test_card(
        self, source_path: str, slug: str, card_index: int, lang: str
    ) -> Card:
        """Create a test card."""
        manifest = Manifest(
            slug=slug,
            slug_base=slug.rsplit("-", 1)[0],
            lang=lang,
            source_path=source_path,
            source_anchor=f"#q{card_index}",
            note_id="test-note",
            note_title="Test Note",
            card_index=card_index,
            guid=f"guid-{slug}",
        )

        return Card(
            slug=slug,
            lang=lang,
            apf_html="<div>Test</div>",
            manifest=manifest,
            content_hash="test-hash",
            guid=f"guid-{slug}",
        )
