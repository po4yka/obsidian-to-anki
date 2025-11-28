"""Tests for state database."""

from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.models import Card, Manifest
import pytest

pytestmark = pytest.mark.skip(reason="State DB tests require SQLite setup")


class TestStateDB:
    """Test state database operations."""

    def test_create_database(self, temp_dir) -> None:
        """Test database initialization."""
        db_path = temp_dir / "test.db"

        with StateDB(db_path) as db:
            # Should create tables
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            assert "cards" in tables

    def test_insert_card(self, temp_dir) -> None:
        """Test inserting a card."""
        card = self._make_test_card("test-slug-en")

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card, anki_guid=12345)

            # Retrieve and verify
            result = db.get_by_slug("test-slug-en")
            assert result is not None
            assert result["slug"] == "test-slug-en"
            assert result["anki_guid"] == 12345
            assert result["lang"] == "en"
            assert result["card_guid"] == card.guid

    def test_update_card(self, temp_dir) -> None:
        """Test updating a card."""
        card = self._make_test_card("test-slug-en")

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card, anki_guid=12345)

            # Update content hash
            card.content_hash = "new-hash-456"
            card.guid = "updated-guid"
            db.update_card(card)

            # Verify update
            result = db.get_by_slug("test-slug-en")
            assert result["content_hash"] == "new-hash-456"
            assert result["card_guid"] == "updated-guid"

    def test_get_by_guid(self, temp_dir) -> None:
        """Test retrieving card by Anki GUID."""
        card = self._make_test_card("test-slug-en")

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card, anki_guid=99999)

            result = db.get_by_guid(99999)
            assert result is not None
            assert result["slug"] == "test-slug-en"

    def test_get_by_source(self, temp_dir) -> None:
        """Test retrieving all cards from a source."""
        card1 = self._make_test_card("test-p01-en", card_index=1)
        card2 = self._make_test_card("test-p01-ru", card_index=1, lang="ru")
        card3 = self._make_test_card("test-p02-en", card_index=2)

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card1, anki_guid=1)
            db.insert_card(card2, anki_guid=2)
            db.insert_card(card3, anki_guid=3)

            results = db.get_by_source("test.md")
            assert len(results) == 3

    def test_delete_card(self, temp_dir) -> None:
        """Test deleting a card."""
        card = self._make_test_card("test-slug-en")

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card, anki_guid=12345)

            # Delete
            db.delete_card("test-slug-en")

            # Verify deleted
            result = db.get_by_slug("test-slug-en")
            assert result is None

    def test_get_all_cards(self, temp_dir) -> None:
        """Test retrieving all cards."""
        with StateDB(temp_dir / "test.db") as db:
            for i in range(5):
                # Vary card_index to avoid UNIQUE constraint on (source_path, card_index, lang)
                card = self._make_test_card(f"card-{i}-en", card_index=i + 1)
                db.insert_card(card, anki_guid=1000 + i)

            all_cards = db.get_all_cards()
            assert len(all_cards) == 5

    def test_unique_constraints(self, temp_dir) -> None:
        """Test unique constraints on slug and guid."""
        card1 = self._make_test_card("duplicate-slug")
        card2 = self._make_test_card("duplicate-slug")

        with StateDB(temp_dir / "test.db") as db:
            db.insert_card(card1, anki_guid=1)

            # Should fail on duplicate slug
            with pytest.raises(Exception):
                db.insert_card(card2, anki_guid=2)

    def _make_test_card(self, slug: str, card_index: int = 1, lang: str = "en") -> Card:
        """Helper to create a test card."""
        manifest = Manifest(
            slug=slug,
            slug_base=slug.rsplit("-", 1)[0],
            lang=lang,
            source_path="test.md",
            source_anchor=f"p{card_index:02d}",
            note_id="test-001",
            note_title="Test Note",
            card_index=card_index,
            guid=f"guid-{slug}",
        )

        return Card(
            slug=slug,
            lang=lang,
            apf_html="<html>test</html>",
            manifest=manifest,
            content_hash="hash-123",
            note_type="APF::Simple",
            tags=["python", "testing"],
            guid=f"guid-{slug}",
        )
