"""Service for Anki media operations."""

from typing import cast

from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.domain.interfaces.anki_media_service import IAnkiMediaService
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiMediaService(IAnkiMediaService):
    """Service for Anki media operations.

    Handles media file operations including storing files
    in Anki's media collection.
    """

    def __init__(self, http_client: AnkiHttpClient):
        """
        Initialize media service.

        Args:
            http_client: HTTP client for AnkiConnect communication
        """
        self._http_client = http_client
        logger.debug("anki_media_service_initialized")

    def store_media_file(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection."""
        return cast(
            "str", self._http_client.invoke("storeMediaFile", {"filename": filename, "data": data})
        )

    async def store_media_file_async(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection (async)."""
        return cast(
            "str",
            await self._http_client.invoke_async(
                "storeMediaFile", {"filename": filename, "data": data}
            ),
        )
