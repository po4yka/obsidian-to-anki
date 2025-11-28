"""Mock implementation of IAnkiClient for testing."""

from typing import Any
from unittest.mock import MagicMock

from obsidian_anki_sync.domain.interfaces.anki_client import IAnkiClient


class MockAnkiClient(IAnkiClient):
    """Mock implementation of Anki client for testing.

    Provides controllable responses for testing sync operations
    without requiring a real AnkiConnect instance.
    """

    def __init__(self):
        """Initialize mock client."""
        self.mock = MagicMock()
        self._decks = ["Test Deck", "Interview Questions"]
        self._models = ["Basic", "Cloze", "APF::Simple"]
        self._notes = {}  # note_id -> note_data
        self._cards = {}  # card_id -> card_data
        self._note_counter = 1000
        self._card_counter = 2000

    def check_connection(self) -> bool:
        """Check connection (always succeeds for mock)."""
        return True

    def get_deck_names(self) -> list[str]:
        """Get available deck names."""
        return self._decks.copy()

    def get_model_names(self) -> list[str]:
        """Get available model names."""
        return self._models.copy()

    def get_model_field_names(self, model_name: str) -> list[str]:
        """Get field names for a model."""
        if model_name == "APF::Simple":
            return ["Question", "Answer", "Manifest"]
        elif model_name == "Basic":
            return ["Front", "Back"]
        return ["Field1", "Field2"]

    def find_notes(self, query: str) -> list[int]:
        """Find notes matching query."""
        # Simple mock implementation
        if "deck:" in query:
            # Return some mock note IDs
            return [1001, 1002, 1003]
        return []

    def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get note information."""
        result = []
        for note_id in note_ids:
            if note_id in self._notes:
                result.append(self._notes[note_id])
            else:
                # Create mock note data
                result.append(
                    {
                        "noteId": note_id,
                        "modelName": "APF::Simple",
                        "deckName": "Test Deck",
                        "fields": {
                            "Question": {"value": f"Question {note_id}"},
                            "Answer": {"value": f"Answer {note_id}"},
                            "Manifest": {"value": f'{{"slug": "test-{note_id}"}}'},
                        },
                        "tags": ["test"],
                    }
                )
        return result

    def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        """Get card information."""
        result = []
        for card_id in card_ids:
            if card_id in self._cards:
                result.append(self._cards[card_id])
            else:
                # Create mock card data
                result.append(
                    {
                        "cardId": card_id,
                        "noteId": card_id - 1000,
                        "deckName": "Test Deck",
                        "modelName": "APF::Simple",
                        "fields": {
                            "Question": {"value": f"Question {card_id}"},
                            "Answer": {"value": f"Answer {card_id}"},
                        },
                        "interval": 1,
                        "due": 1234567890,
                        "reps": 0,
                        "lapses": 0,
                        "queue": 0,
                        "mod": 1234567890,
                    }
                )
        return result

    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> int:
        """Add a new note."""
        note_id = self._note_counter
        self._note_counter += 1

        note_data = {
            "noteId": note_id,
            "modelName": model_name,
            "deckName": deck_name,
            "fields": {k: {"value": v} for k, v in fields.items()},
            "tags": tags or [],
        }
        self._notes[note_id] = note_data

        return note_id

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Update note fields."""
        if note_id in self._notes:
            for field_name, field_value in fields.items():
                if "fields" in self._notes[note_id]:
                    self._notes[note_id]["fields"][field_name] = {"value": field_value}

    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes."""
        for note_id in note_ids:
            self._notes.pop(note_id, None)

    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID."""
        # Mock implementation - assume card_id = note_id + 1000
        return card_id - 1000

    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID."""
        # Mock implementation - assume 1 card per note
        return [note_id + 1000]

    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards."""
        # Mock implementation - just mark as suspended
        for card_id in card_ids:
            if card_id in self._cards:
                self._cards[card_id]["suspended"] = True

    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards."""
        # Mock implementation - mark as not suspended
        for card_id in card_ids:
            if card_id in self._cards:
                self._cards[card_id]["suspended"] = False

    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get deck statistics."""
        return {
            "deck_id": 1,
            "name": deck_name,
            "total_cards": len(self._cards),
            "new_cards": 10,
            "review_cards": 20,
            "learned_cards": 5,
        }

    # Helper methods for testing

    def add_mock_note(self, note_id: int, data: dict[str, Any]) -> None:
        """Add a mock note for testing."""
        self._notes[note_id] = data

    def add_mock_card(self, card_id: int, data: dict[str, Any]) -> None:
        """Add a mock card for testing."""
        self._cards[card_id] = data

    def get_notes(self) -> dict[int, dict[str, Any]]:
        """Get all mock notes (for testing)."""
        return self._notes.copy()

    def get_cards(self) -> dict[int, dict[str, Any]]:
        """Get all mock cards (for testing)."""
        return self._cards.copy()

    def reset(self) -> None:
        """Reset mock state."""
        self._notes.clear()
        self._cards.clear()
        self._note_counter = 1000
        self._card_counter = 2000
