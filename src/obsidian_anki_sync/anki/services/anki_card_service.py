"""Service for Anki card operations."""

from typing import Any, cast

from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.domain.interfaces.anki_card_service import IAnkiCardService
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiCardService(IAnkiCardService):
    """Service for Anki card operations.

    Handles card-related operations including suspending/unsuspending cards,
    retrieving card information, and card-note relationships.
    """

    def __init__(self, http_client: AnkiHttpClient):
        """
        Initialize card service.

        Args:
            http_client: HTTP client for AnkiConnect communication
        """
        self._http_client = http_client
        logger.debug("anki_card_service_initialized")

    def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about cards."""
        if not card_ids:
            return []
        return cast(
            "list[dict[Any, Any]]", self._http_client.invoke("cardsInfo", {"cards": card_ids})
        )

    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards by ID."""
        if not card_ids:
            return

        self._http_client.invoke("suspend", {"cards": card_ids})
        logger.info("cards_suspended", count=len(card_ids))

    async def suspend_cards_async(self, card_ids: list[int]) -> None:
        """Suspend cards by ID (async)."""
        if not card_ids:
            return
        await self._http_client.invoke_async("suspend", {"cards": card_ids})
        logger.info("cards_suspended", count=len(card_ids))

    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID."""
        if not card_ids:
            return

        self._http_client.invoke("unsuspend", {"cards": card_ids})
        logger.info("cards_unsuspended", count=len(card_ids))

    async def unsuspend_cards_async(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID (async)."""
        if not card_ids:
            return
        await self._http_client.invoke_async("unsuspend", {"cards": card_ids})
        logger.info("cards_unsuspended", count=len(card_ids))

    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID."""
        return cast("int", self._http_client.invoke("cardsToNotes", {"cards": [card_id]})[0])

    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID."""
        return cast("list[int]", self._http_client.invoke("notesToCards", {"notes": [note_id]})[0])
