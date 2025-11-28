"""Unit tests for sync engine behaviour."""

from obsidian_anki_sync.sync.engine import SyncEngine
from obsidian_anki_sync.models import Card, Manifest
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.skip(reason="Sync engine tests require complex setup")


def test_update_card_updates_fields_and_tags(test_config) -> None:
    """_update_card should refresh fields, tags, and database state."""
    from obsidian_anki_sync.anki.client import AnkiClient
    from obsidian_anki_sync.sync.state_db import StateDB

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
        guid="guid-sample",
    )

    card = Card(
        slug="sample-slug-en",
        lang="en",
        apf_html="<!-- dummy -->",
        manifest=manifest,
        content_hash="hash",
        note_type="APF::Simple",
        tags=["en", "testing"],
        guid="guid-sample",
    )

    with patch(
        "obsidian_anki_sync.sync.change_applier.map_apf_to_anki_fields",
        return_value={"Front": "Q", "Back": "A"},
    ):
        engine.change_applier.update_card(card, anki_guid=12345)

    anki.update_note_fields.assert_called_once_with(
        12345, {"Front": "Q", "Back": "A"})
    anki.update_note_tags.assert_called_once_with(12345, card.tags)
    # Updated to match new db.update_card_extended signature
    db.update_card_extended.assert_called_once_with(
        card=card,
        fields={"Front": "Q", "Back": "A"},
        tags=card.tags,
        apf_html=card.apf_html,
    )
