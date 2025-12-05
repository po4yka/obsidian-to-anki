"""Service for Anki deck operations."""

from typing import Any, cast

from obsidian_anki_sync.anki.services.anki_cache import AnkiCache
from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.domain.interfaces.anki_deck_service import IAnkiDeckService
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiDeckService(IAnkiDeckService):
    """Service for Anki deck operations.

    Handles deck-related operations including retrieving deck names
    and statistics, with caching support.
    """

    def __init__(self, http_client: AnkiHttpClient, cache: AnkiCache):
        """
        Initialize deck service.

        Args:
            http_client: HTTP client for AnkiConnect communication
            cache: Cache for metadata
        """
        self._http_client = http_client
        self._cache = cache
        logger.debug("anki_deck_service_initialized")

    def get_deck_names(self, use_cache: bool = True) -> list[str]:
        """Get list of available deck names."""
        cache_key = "deck_names"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast("list[str]", self._http_client.invoke("deckNames"))
        self._cache.set(cache_key, result)
        return result

    async def get_deck_names_async(self, use_cache: bool = True) -> list[str]:
        """Get list of available deck names (async)."""
        cache_key = "deck_names"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cast("list[str]", cached)

        result = cast("list[str]", await self._http_client.invoke_async("deckNames"))
        self._cache.set(cache_key, result)
        return result

    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck."""
        try:
            # Try to use getDeckStats if available (e.g. if using the fork)
            return cast(
                "dict[str, Any]", self._http_client.invoke("getDeckStats", {"decks": [deck_name]})
            )
        except AnkiConnectError:
            # Fallback to manual calculation
            logger.debug("getDeckStats_not_supported_falling_back", deck=deck_name)

            # Using findCards which is faster than findNotes for stats
            total_cards = len(
                self._http_client.invoke("findCards", {"query": f'deck:"{deck_name}"'})
            )
            new_cards = len(
                self._http_client.invoke("findCards", {"query": f'deck:"{deck_name}" is:new'})
            )
            learn_cards = len(
                self._http_client.invoke("findCards", {"query": f'deck:"{deck_name}" is:learn'})
            )
            due_cards = len(
                self._http_client.invoke("findCards", {"query": f'deck:"{deck_name}" is:due'})
            )

            # Note: is:due includes learn cards that are due, but we separate them roughly here
            # Ideally we'd use getDeckStats if possible.

            # Construct a response similar to what's expected
            return {
                "deck_id": 0,  # We don't have the deck ID easily available
                "name": deck_name,
                "total_in_deck": total_cards,
                "new_count": new_cards,
                "learn_count": learn_cards,
                "review_count": due_cards,
            }
