"""Interface for Anki tag operations."""

from abc import ABC, abstractmethod


class IAnkiTagService(ABC):
    """Interface for Anki tag operations.

    Defines operations for managing tags on Anki notes,
    including adding, removing, and updating tags.
    """

    @abstractmethod
    def update_note_tags(self, note_id: int, tags: list[str]) -> None:
        """Synchronize tags for a single note by applying minimal add/remove operations.

        Args:
            note_id: Note ID
            tags: Desired set of tags
        """

    @abstractmethod
    async def update_note_tags_async(self, note_id: int, tags: list[str]) -> None:
        """Synchronize tags for a single note by applying minimal add/remove operations (async).

        Args:
            note_id: Note ID
            tags: Desired set of tags
        """

    @abstractmethod
    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Add tags to notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to add
        """

    @abstractmethod
    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remove tags from notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to remove
        """

    @abstractmethod
    def update_notes_tags(self, note_tag_pairs: list[tuple[int, list[str]]]) -> list[bool]:
        """Update tags for multiple notes in a batch operation.

        Optimizes by grouping notes with identical tag sets into single
        replaceTags calls, reducing the number of API actions.

        Args:
            note_tag_pairs: List of (note_id, tags) tuples

        Returns:
            List of booleans indicating success for each update
        """

    @abstractmethod
    async def update_notes_tags_async(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """Update tags for multiple notes in a batch operation (async).

        Args:
            note_tag_pairs: List of (note_id, tags) tuples

        Returns:
            List of booleans indicating success for each update
        """
