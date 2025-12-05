"""Interface for Anki media operations."""

from abc import ABC, abstractmethod


class IAnkiMediaService(ABC):
    """Interface for Anki media operations.

    Defines operations for managing media files in Anki's
    media collection, including storing and retrieving media.
    """

    @abstractmethod
    def store_media_file(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection.

        Args:
            filename: Name of the file to store
            data: Base64-encoded file data

        Returns:
            The filename as stored in Anki (may be modified)
        """

    @abstractmethod
    async def store_media_file_async(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection (async).

        Args:
            filename: Name of the file to store
            data: Base64-encoded file data

        Returns:
            The filename as stored in Anki (may be modified)
        """
