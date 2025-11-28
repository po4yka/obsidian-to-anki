"""Interface for state persistence operations."""

from abc import ABC, abstractmethod
from typing import Any

from ..entities.card import Card
from ..entities.note import Note


class IStateRepository(ABC):
    """Interface for state persistence and retrieval.

    This interface defines the contract for storing and retrieving
    synchronization state, notes, cards, and metadata.
    """

    @abstractmethod
    def get_note_by_id(self, note_id: str) -> Note | None:
        """Retrieve a note by its ID.

        Args:
            note_id: Unique note identifier

        Returns:
            Note instance if found, None otherwise
        """
        pass

    @abstractmethod
    def get_notes_by_path(self, file_path: str) -> list[Note]:
        """Retrieve notes by file path.

        Args:
            file_path: Path to the note file

        Returns:
            List of notes from that file
        """
        pass

    @abstractmethod
    def save_note(self, note: Note) -> None:
        """Save a note to the repository.

        Args:
            note: Note instance to save
        """
        pass

    @abstractmethod
    def delete_note(self, note_id: str) -> None:
        """Delete a note from the repository.

        Args:
            note_id: ID of note to delete
        """
        pass

    @abstractmethod
    def get_card_by_slug(self, slug: str) -> Card | None:
        """Retrieve a card by its slug.

        Args:
            slug: Card slug

        Returns:
            Card instance if found, None otherwise
        """
        pass

    @abstractmethod
    def get_cards_by_note_id(self, note_id: str) -> list[Card]:
        """Retrieve all cards for a note.

        Args:
            note_id: Note ID

        Returns:
            List of cards for the note
        """
        pass

    @abstractmethod
    def save_card(self, card: Card) -> None:
        """Save a card to the repository.

        Args:
            card: Card instance to save
        """
        pass

    @abstractmethod
    def delete_card(self, slug: str) -> None:
        """Delete a card from the repository.

        Args:
            slug: Slug of card to delete
        """
        pass

    @abstractmethod
    def get_all_notes(self) -> list[Note]:
        """Retrieve all notes.

        Returns:
            List of all notes
        """
        pass

    @abstractmethod
    def get_all_cards(self) -> list[Card]:
        """Retrieve all cards.

        Returns:
            List of all cards
        """
        pass

    @abstractmethod
    def get_sync_stats(self) -> dict[str, Any]:
        """Get synchronization statistics.

        Returns:
            Dictionary with sync statistics
        """
        pass

    @abstractmethod
    def save_sync_session(self, session_data: dict[str, Any]) -> str:
        """Save sync session data.

        Args:
            session_data: Session information

        Returns:
            Session ID
        """
        pass

    @abstractmethod
    def get_sync_session(self, session_id: str) -> dict[str, Any | None]:
        """Retrieve sync session data.

        Args:
            session_id: Session identifier

        Returns:
            Session data if found, None otherwise
        """
        pass

    @abstractmethod
    def update_sync_progress(
        self, session_id: str, progress_data: dict[str, Any]
    ) -> None:
        """Update sync progress for a session.

        Args:
            session_id: Session identifier
            progress_data: Progress information
        """
        pass

    @abstractmethod
    def get_content_hash(self, resource_id: str) -> str | None:
        """Get stored content hash for a resource.

        Args:
            resource_id: Unique resource identifier

        Returns:
            Content hash if found, None otherwise
        """
        pass

    @abstractmethod
    def save_content_hash(self, resource_id: str, hash_value: str) -> None:
        """Save content hash for a resource.

        Args:
            resource_id: Unique resource identifier
            hash_value: Content hash
        """
        pass

    @abstractmethod
    def clear_expired_data(self, max_age_days: int) -> int:
        """Clear expired data from repository.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of records cleared
        """
        pass
