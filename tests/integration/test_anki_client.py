"""Integration tests for AnkiConnect client (INT-01, INT-crud-01)."""

import httpx
import pytest
import respx

from obsidian_anki_sync.anki.client import AnkiClient, AnkiConnectError


@pytest.fixture()
def mock_anki_url():
    """Mock AnkiConnect URL."""
    return "http://localhost:8765"


class TestAnkiClient:
    """Test AnkiConnect client (INT-01)."""

    @respx.mock
    def test_successful_invoke(self, mock_anki_url) -> None:
        """Test successful API call."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(200, json={"result": "success", "error": None})
        )

        client = AnkiClient(mock_anki_url)
        result = client.invoke("testAction", {"param": "value"})

        assert result == "success"

    @respx.mock
    def test_error_response(self, mock_anki_url) -> None:
        """Test error handling."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(
                200, json={"result": None, "error": "Test error"}
            )
        )

        client = AnkiClient(mock_anki_url, enable_health_checks=False)

        with pytest.raises(AnkiConnectError, match="Test error"):
            client.invoke("testAction")

    @respx.mock
    def test_find_notes(self, mock_anki_url) -> None:
        """Test finding notes."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(200, json={"result": [1, 2, 3], "error": None})
        )

        client = AnkiClient(mock_anki_url)
        notes = client.find_notes("deck:Test")

        assert notes == [1, 2, 3]

    @respx.mock
    def test_notes_info(self, mock_anki_url) -> None:
        """Test getting note info."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "noteId": 1,
                            "fields": {"Front": {"value": "Question"}},
                            "tags": ["test"],
                        }
                    ],
                    "error": None,
                },
            )
        )

        client = AnkiClient(mock_anki_url)
        info = client.notes_info([1])

        assert len(info) == 1
        assert info[0]["noteId"] == 1

    @respx.mock
    def test_add_note(self, mock_anki_url) -> None:
        """Test adding a note (INT-crud-01)."""
        route = respx.post(mock_anki_url)
        route.mock(
            return_value=httpx.Response(200, json={"result": 12345, "error": None})
        )

        client = AnkiClient(mock_anki_url)
        note_id = client.add_note(
            deck_name="Test Deck",
            model_name="Basic",
            fields={"Front": "Q", "Back": "A"},
            tags=["test"],
        )

        assert note_id == 12345
        assert route.called

    @respx.mock
    def test_can_add_notes(self, mock_anki_url) -> None:
        """Test checking if notes can be added."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(
                200, json={"result": [True, False], "error": None}
            )
        )

        client = AnkiClient(mock_anki_url)
        can_add = client.can_add_notes(
            [
                {
                    "deckName": "Test Deck",
                    "modelName": "Basic",
                    "fields": {"Front": "Q1", "Back": "A1"},
                },
                {
                    "deckName": "Test Deck",
                    "modelName": "Basic",
                    "fields": {"Front": "Q2", "Back": "A2"},
                },
            ]
        )

        assert can_add == [True, False]

    @respx.mock
    def test_store_media_file(self, mock_anki_url) -> None:
        """Test storing a media file."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(
                200, json={"result": "stored_image.png", "error": None}
            )
        )

        client = AnkiClient(mock_anki_url)
        result = client.store_media_file("image.png", "base64data")

        assert result == "stored_image.png"

    @respx.mock
    def test_suspend_cards(self, mock_anki_url) -> None:
        """Test suspending cards."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(200, json={"result": None, "error": None})
        )

        client = AnkiClient(mock_anki_url)
        client.suspend_cards([1, 2, 3])

        # Verify the request was made
        assert mock_anki_url.rstrip("/") in str(respx.calls.last.request.url)

    @respx.mock
    def test_update_note_fields(self, mock_anki_url) -> None:
        """Test updating note fields (INT-crud-01)."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(200, json={"result": None, "error": None})
        )

        client = AnkiClient(mock_anki_url)
        client.update_note_fields(note_id=12345, fields={"Front": "Updated Q"})

        # Should not raise

    @respx.mock
    def test_delete_notes(self, mock_anki_url) -> None:
        """Test deleting notes (INT-crud-01)."""
        respx.post(mock_anki_url).mock(
            return_value=httpx.Response(200, json={"result": None, "error": None})
        )

        client = AnkiClient(mock_anki_url)
        client.delete_notes([1, 2, 3])

        # Should not raise

    @respx.mock
    def test_http_error(self, mock_anki_url) -> None:
        """Test HTTP error handling."""
        respx.post(mock_anki_url).mock(return_value=httpx.Response(500))

        client = AnkiClient(mock_anki_url, enable_health_checks=False)

        with pytest.raises(AnkiConnectError, match="HTTP 500 from AnkiConnect"):
            client.invoke("testAction")
