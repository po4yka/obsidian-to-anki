"""Interface for Anki client operations."""

from abc import ABC, abstractmethod
from typing import Any



class IAnkiClient(ABC):
    """Interface for Anki connectivity and card operations.

    This interface defines the contract for communicating with Anki
    through the AnkiConnect API.
    """

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if AnkiConnect is available and responsive.

        Returns:
            True if connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def get_deck_names(self) -> list[str]:
        """Get list of available deck names.

        Returns:
            List of deck names
        """
        pass

    @abstractmethod
    def get_model_names(self) -> list[str]:
        """Get list of available note model names.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def get_model_field_names(self, model_name: str) -> list[str]:
        """Get field names for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of field names
        """
        pass

    @abstractmethod
    def find_notes(self, query: str) -> list[int]:
        """Find notes matching a query.

        Args:
            query: Anki query string

        Returns:
            List of note IDs
        """
        pass

    @abstractmethod
    def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about notes.

        Args:
            note_ids: List of note IDs

        Returns:
            List of note information dictionaries
        """
        pass

    @abstractmethod
    def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about cards.

        Args:
            card_ids: List of card IDs

        Returns:
            List of card information dictionaries
        """
        pass

    @abstractmethod
    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] = None,
        options: dict[str, Any | None] = None
    ) -> int:
        """Add a new note to Anki.

        Args:
            deck_name: Name of the deck
            model_name: Name of the note model
            fields: Field name -> value mapping
            tags: Optional list of tags
            options: Optional additional options

        Returns:
            Note ID of the created note
        """
        pass

    @abstractmethod
    def update_note_fields(
        self,
        note_id: int,
        fields: dict[str, str]
    ) -> None:
        """Update fields of an existing note.

        Args:
            note_id: ID of the note to update
            fields: Field name -> new value mapping
        """
        pass

    @abstractmethod
    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes from Anki.

        Args:
            note_ids: List of note IDs to delete
        """
        pass

    @abstractmethod
    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID.

        Args:
            card_id: Card ID

        Returns:
            Note ID
        """
        pass

    @abstractmethod
    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID.

        Args:
            note_id: Note ID

        Returns:
            List of card IDs
        """
        pass

    @abstractmethod
    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards.

        Args:
            card_ids: List of card IDs to suspend
        """
        pass

    @abstractmethod
    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards.

        Args:
            card_ids: List of card IDs to unsuspend
        """
        pass

    @abstractmethod
    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck.

        Args:
            deck_name: Name of the deck

        Returns:
            Deck statistics
        """
        pass
