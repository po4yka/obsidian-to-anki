"""Tests for AnkiClient tag update logic."""

from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.exceptions import AnkiConnectError


def _build_client() -> AnkiClient:
    """Create an AnkiClient instance with mocked services for testing."""
    # Create mock services
    mock_http_client = MagicMock()
    mock_deck_service = MagicMock()
    mock_model_service = MagicMock()
    mock_note_service = MagicMock()
    mock_card_service = MagicMock()
    mock_tag_service = MagicMock()
    mock_media_service = MagicMock()
    mock_cache = MagicMock()

    # Configure mocks to not raise exceptions
    mock_tag_service.update_note_tags.return_value = None

    # Create client with mocked services
    client = object.__new__(AnkiClient)
    client.url = "http://localhost:8765"
    client.enable_health_checks = False
    client._http_client = mock_http_client
    client._deck_service = mock_deck_service
    client._model_service = mock_model_service
    client._note_service = mock_note_service
    client._card_service = mock_card_service
    client._tag_service = mock_tag_service
    client._media_service = mock_media_service
    client._cache = mock_cache
    client._async_runner = MagicMock()

    return client


def test_update_note_tags_delegates_to_service() -> None:
    """Test that update_note_tags delegates to the tag service."""
    client = _build_client()

    client.update_note_tags(42, ["existing", "new-tag"])

    client._tag_service.update_note_tags.assert_called_once_with(42, ["existing", "new-tag"])


def test_update_note_tags_handles_empty_target() -> None:
    """Test that update_note_tags handles empty tag list."""
    client = _build_client()

    client.update_note_tags(99, [])

    client._tag_service.update_note_tags.assert_called_once_with(99, [])


def test_update_note_tags_propagates_exceptions() -> None:
    """Test that update_note_tags propagates exceptions from the service."""
    client = _build_client()
    client._tag_service.update_note_tags.side_effect = AnkiConnectError("Note not found")

    with pytest.raises(AnkiConnectError):
        client.update_note_tags(7, ["tag"])

    client._tag_service.update_note_tags.assert_called_once_with(7, ["tag"])
