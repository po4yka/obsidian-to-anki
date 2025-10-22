"""Tests for AnkiClient tag update logic."""

from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.anki.client import AnkiClient, AnkiConnectError


def _build_client() -> AnkiClient:
    """Create an AnkiClient instance without triggering HTTP setup."""
    client = object.__new__(AnkiClient)
    client.url = "http://localhost:8765"
    client.session = None  # Avoid real HTTP usage in tests
    return client


def test_update_note_tags_adds_and_removes_diff():
    client = _build_client()
    client.notes_info = MagicMock(return_value=[{"tags": ["existing", "legacy"]}])
    client.add_tags = MagicMock()
    client.remove_tags = MagicMock()

    client.update_note_tags(42, ["existing", "new-tag"])

    client.add_tags.assert_called_once_with([42], "new-tag")
    client.remove_tags.assert_called_once_with([42], "legacy")


def test_update_note_tags_handles_empty_target():
    client = _build_client()
    client.notes_info = MagicMock(return_value=[{"tags": ["old1", "old2"]}])
    client.add_tags = MagicMock()
    client.remove_tags = MagicMock()

    client.update_note_tags(99, [])

    client.add_tags.assert_not_called()
    client.remove_tags.assert_called_once_with([99], "old1 old2")


def test_update_note_tags_raises_when_note_missing():
    client = _build_client()
    client.notes_info = MagicMock(return_value=[])
    client.add_tags = MagicMock()
    client.remove_tags = MagicMock()

    with pytest.raises(AnkiConnectError):
        client.update_note_tags(7, ["tag"])

    client.add_tags.assert_not_called()
    client.remove_tags.assert_not_called()
