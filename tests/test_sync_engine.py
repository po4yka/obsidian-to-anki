"""Unit tests for sync engine behaviour."""

from unittest.mock import MagicMock, patch

from obsidian_anki_sync.models import Card, Manifest
from obsidian_anki_sync.sync.engine import SyncEngine


def test_update_card_updates_fields_and_tags(test_config):
    """_update_card should refresh fields, tags, and database state."""
    from obsidian_anki_sync.sync.state_db import StateDB
    from obsidian_anki_sync.anki.client import AnkiClient

    db = MagicMock(spec=StateDB)
    anki = MagicMock(spec=AnkiClient)

    engine = SyncEngine(test_config, db, anki)

    manifest = Manifest(
        slug="sample-slug-en",
        slug_base="sample-slug",
        lang="en",
        source_path="relative/path.md",
        source_anchor="p01",
        note_id="test-001",
        note_title="Sample Title",
        card_index=1,
    )

    card = Card(
        slug="sample-slug-en",
        lang="en",
        apf_html="<!-- dummy -->",
        manifest=manifest,
        content_hash="hash",
        note_type="APF::Simple",
        tags=["en", "testing"],
    )

    with patch("obsidian_anki_sync.sync.engine.map_apf_to_anki_fields", return_value={"Front": "Q", "Back": "A"}):
        engine._update_card(card, anki_guid=12345)

    anki.update_note_fields.assert_called_once_with(12345, {"Front": "Q", "Back": "A"})
    anki.update_note_tags.assert_called_once_with(12345, card.tags)
    db.update_card.assert_called_once_with(card)
