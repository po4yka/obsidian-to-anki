"""Interface for Anki note operations."""

from abc import ABC, abstractmethod
from typing import Any


class IAnkiNoteService(ABC):
    """Interface for Anki note operations.

    Defines operations for working with Anki notes, including
    creating, updating, deleting, and querying notes.
    """

    @abstractmethod
    def find_notes(self, query: str) -> list[int]:
        """Find notes matching a query.

        Args:
            query: Anki search query

        Returns:
            List of note IDs
        """

    @abstractmethod
    async def find_notes_async(self, query: str) -> list[int]:
        """Find notes matching a query (async).

        Args:
            query: Anki search query

        Returns:
            List of note IDs
        """

    @abstractmethod
    def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about notes.

        Args:
            note_ids: List of note IDs

        Returns:
            List of note information dictionaries
        """

    @abstractmethod
    async def notes_info_async(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about notes (async).

        Args:
            note_ids: List of note IDs

        Returns:
            List of note information dictionaries
        """

    @abstractmethod
    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any] | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note to Anki.

        Args:
            deck_name: Name of the deck
            model_name: Name of the note model
            fields: Field name -> value mapping
            tags: Optional list of tags
            options: Optional additional options
            guid: Optional GUID for the note

        Returns:
            Note ID of the created note
        """

    @abstractmethod
    async def add_note_async(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any] | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note to Anki (async).

        Args:
            deck_name: Name of the deck
            model_name: Name of the note model
            fields: Field name -> value mapping
            tags: Optional list of tags
            options: Optional additional options
            guid: Optional GUID for the note

        Returns:
            Note ID of the created note
        """

    @abstractmethod
    def add_notes(self, notes: list[dict[str, Any]]) -> list[int | None]:
        """Add multiple notes in a single batch operation.

        Args:
            notes: List of note payloads, each containing:
                - deckName: str
                - modelName: str
                - fields: dict[str, str]
                - tags: list[str]
                - options: dict (optional)
                - guid: str (optional)

        Returns:
            List of note IDs (or None for failed notes)
        """

    @abstractmethod
    async def add_notes_async(self, notes: list[dict[str, Any]]) -> list[int | None]:
        """Add multiple notes in a single batch operation (async).

        Args:
            notes: List of note payloads, each containing:
                - deckName: str
                - modelName: str
                - fields: dict[str, str]
                - tags: list[str]
                - options: dict (optional)
                - guid: str (optional)

        Returns:
            List of note IDs (or None for failed notes)
        """

    @abstractmethod
    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Update fields of an existing note.

        Args:
            note_id: ID of the note to update
            fields: Field name -> new value mapping
        """

    @abstractmethod
    async def update_note_fields_async(self, note_id: int, fields: dict[str, str]) -> None:
        """Update fields of an existing note (async).

        Args:
            note_id: ID of the note to update
            fields: Field name -> new value mapping
        """

    @abstractmethod
    def update_notes_fields(self, updates: list[dict[str, Any]]) -> list[bool]:
        """Update multiple notes' fields in a single batch operation.

        Args:
            updates: List of update dicts, each containing:
                - id: int (note ID)
                - fields: dict[str, str]

        Returns:
            List of booleans indicating success for each update
        """

    @abstractmethod
    async def update_notes_fields_async(self, updates: list[dict[str, Any]]) -> list[bool]:
        """Update multiple notes' fields in a single batch operation (async).

        Args:
            updates: List of update dicts, each containing:
                - id: int (note ID)
                - fields: dict[str, str]

        Returns:
            List of booleans indicating success for each update
        """

    @abstractmethod
    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes from Anki.

        Args:
            note_ids: List of note IDs to delete
        """

    @abstractmethod
    async def delete_notes_async(self, note_ids: list[int]) -> None:
        """Delete notes from Anki (async).

        Args:
            note_ids: List of note IDs to delete
        """

    @abstractmethod
    def can_add_notes(self, notes: list[dict[str, Any]]) -> list[bool]:
        """Check if notes can be added (duplicate prevention).

        Args:
            notes: List of note payloads to check, each containing:
                - deckName: str
                - modelName: str
                - fields: dict[str, str]

        Returns:
            List of booleans indicating whether each note can be added
        """
