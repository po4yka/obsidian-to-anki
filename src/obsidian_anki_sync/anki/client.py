"""AnkiConnect HTTP API client."""

import contextlib
from types import TracebackType
from typing import Any, Literal, TypedDict, cast

from obsidian_anki_sync.anki.services.anki_cache import AnkiCache
from obsidian_anki_sync.anki.services.anki_card_service import AnkiCardService
from obsidian_anki_sync.anki.services.anki_deck_service import AnkiDeckService
from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.anki.services.anki_media_service import AnkiMediaService
from obsidian_anki_sync.anki.services.anki_model_service import AnkiModelService
from obsidian_anki_sync.anki.services.anki_note_service import AnkiNoteService
from obsidian_anki_sync.anki.services.anki_tag_service import AnkiTagService
from obsidian_anki_sync.domain.interfaces.anki_client import IAnkiClient
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class DuplicateScopeOptions(TypedDict, total=False):
    """Options for duplicate checking scope."""

    deckName: str
    checkFlds: list[str]
    checkChildren: bool


class NoteOptions(TypedDict, total=False):
    """Options for note creation."""

    allowDuplicate: bool
    duplicateScope: Literal["deck", "collection"]
    duplicateScopeOptions: DuplicateScopeOptions


class AnkiClient(IAnkiClient):
    """Client for AnkiConnect HTTP API.

    This client uses composition with smaller, focused services to follow
    SOLID principles and Clean Architecture. Each service handles a specific
    domain of operations (notes, cards, decks, etc.) while maintaining the
    same external interface for backward compatibility.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 180.0,
        enable_health_checks: bool = True,
        async_runner: AsyncioRunner | None = None,
        max_keepalive_connections: int = 10,
        max_connections: int = 20,
        keepalive_expiry: float = 60.0,
        verify_connectivity: bool = True,
    ):
        """
        Initialize client with composed services.

        Args:
            url: AnkiConnect URL
            timeout: Request timeout in seconds
            enable_health_checks: Whether to perform periodic health checks
            async_runner: Optional async runner for sync/async bridging
            max_keepalive_connections: Max idle connections to keep alive
            max_connections: Max total connections in pool
            keepalive_expiry: Seconds before idle connections expire
            verify_connectivity: If True, verify AnkiConnect is reachable at startup

        Raises:
            AnkiConnectError: If verify_connectivity=True and Anki is unreachable
        """
        # Initialize core infrastructure services
        self._cache = AnkiCache()
        self._http_client = AnkiHttpClient(
            url=url,
            timeout=timeout,
            enable_health_checks=enable_health_checks,
            async_runner=async_runner,
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            keepalive_expiry=keepalive_expiry,
            verify_connectivity=verify_connectivity,
        )

        # Initialize domain services
        self._deck_service = AnkiDeckService(self._http_client, self._cache)
        self._model_service = AnkiModelService(self._http_client, self._cache)
        self._note_service = AnkiNoteService(
            self._http_client, self._deck_service, self._model_service, self._cache
        )
        self._card_service = AnkiCardService(self._http_client)
        self._tag_service = AnkiTagService(self._http_client, self._note_service)
        self._media_service = AnkiMediaService(self._http_client)

        # Expose HTTP client properties for backward compatibility
        self.url = url
        self.enable_health_checks = enable_health_checks
        self._async_runner = async_runner or AsyncioRunner.get_global()

        logger.info("anki_client_initialized_with_services", url=url)

    @property
    def session(self):
        """Get the HTTP session for backward compatibility."""
        return self._http_client.session

    @property
    def _async_client(self):
        """Get the async HTTP client for backward compatibility."""
        return self._http_client.async_client

    def invalidate_metadata_cache(self) -> None:
        """Invalidate all cached metadata (deck/model names)."""
        self._cache.invalidate_all()

    def invoke(self, action: str, params: dict | None = None) -> Any:
        """Invoke AnkiConnect action via HTTP client."""
        return self._http_client.invoke(action, params)

    async def invoke_async(self, action: str, params: dict | None = None) -> Any:
        """Invoke AnkiConnect action asynchronously via HTTP client."""
        return await self._http_client.invoke_async(action, params)

    def find_notes(self, query: str) -> list[int]:
        """Find notes matching query."""
        return self._note_service.find_notes(query)

    async def find_notes_async(self, query: str) -> list[int]:
        """Find notes matching query (async)."""
        return await self._note_service.find_notes_async(query)

    def notes_info(self, note_ids: list[int]) -> list[dict]:
        """Get information about notes."""
        return self._note_service.notes_info(note_ids)

    async def notes_info_async(self, note_ids: list[int]) -> list[dict]:
        """Get information about notes (async)."""
        return await self._note_service.notes_info_async(note_ids)

    def cards_info(self, card_ids: list[int]) -> list[dict]:
        """Get information about cards."""
        return self._card_service.cards_info(card_ids)

    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: NoteOptions | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note."""
        return self._note_service.add_note(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

    async def add_note_async(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: NoteOptions | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note (async)."""
        return await self._note_service.add_note_async(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

    def add_notes(self, notes: list[dict[str, Any]]) -> list[int | None]:
        """Add multiple notes in a single batch operation."""
        return self._note_service.add_notes(notes)

    async def add_notes_async(self, notes: list[dict[str, Any]]) -> list[int | None]:
        """Add multiple notes in a single batch operation (async)."""
        return await self._note_service.add_notes_async(notes)

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Update note fields."""
        self._note_service.update_note_fields(note_id, fields)

    async def update_note_fields_async(
        self, note_id: int, fields: dict[str, str]
    ) -> None:
        """Update note fields (async)."""
        await self._note_service.update_note_fields_async(note_id, fields)

    def update_notes_fields(self, updates: list[dict[str, Any]]) -> list[bool]:
        """Update multiple notes' fields in a single batch operation."""
        return self._note_service.update_notes_fields(updates)

    async def update_notes_fields_async(
        self, updates: list[dict[str, Any]]
    ) -> list[bool]:
        """Update multiple notes' fields in a single batch operation (async)."""
        return await self._note_service.update_notes_fields_async(updates)

    def update_note_tags(self, note_id: int, tags: list[str]) -> None:
        """Synchronize tags for a single note."""
        self._tag_service.update_note_tags(note_id, tags)

    async def update_note_tags_async(self, note_id: int, tags: list[str]) -> None:
        """Synchronize tags for a single note (async)."""
        await self._tag_service.update_note_tags_async(note_id, tags)

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Add tags to notes."""
        self._tag_service.add_tags(note_ids, tags)

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remove tags from notes."""
        self._tag_service.remove_tags(note_ids, tags)

    def update_notes_tags(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """Update tags for multiple notes in a batch operation."""
        return self._tag_service.update_notes_tags(note_tag_pairs)

    async def update_notes_tags_async(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """Update tags for multiple notes in a batch operation (async)."""
        return await self._tag_service.update_notes_tags_async(note_tag_pairs)

    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes."""
        self._note_service.delete_notes(note_ids)

    async def delete_notes_async(self, note_ids: list[int]) -> None:
        """Delete notes (async)."""
        await self._note_service.delete_notes_async(note_ids)

    def get_deck_names(self, use_cache: bool = True) -> list[str]:
        """Get all deck names."""
        return self._deck_service.get_deck_names(use_cache)

    def get_model_names(self, use_cache: bool = True) -> list[str]:
        """Get all note type (model) names."""
        return self._model_service.get_model_names(use_cache)

    def get_model_field_names(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a note type."""
        return self._model_service.get_model_field_names(model_name, use_cache)

    def can_add_notes(self, notes: list[dict[str, Any]]) -> list[bool]:
        """Check if notes can be added (duplicate prevention)."""
        return self._note_service.can_add_notes(notes)

    def store_media_file(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection."""
        return self._media_service.store_media_file(filename, data)

    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards by ID."""
        self._card_service.suspend_cards(card_ids)

    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID."""
        self._card_service.unsuspend_cards(card_ids)

    def gui_browse(self, query: str) -> list[int]:
        """Open Anki browser with a search query."""
        return cast(
            "list[int]", self._http_client.invoke("guiBrowse", {"query": query})
        )

    def get_collection_stats(self) -> str:
        """Get collection statistics as HTML."""
        return cast("str", self._http_client.invoke("getCollectionStatsHtml"))

    def get_model_names_and_ids(self) -> dict[str, int]:
        """Get note type names and their IDs."""
        return self._model_service.get_model_names_and_ids()

    def get_num_cards_reviewed_today(self) -> int:
        """Get the number of cards reviewed today."""
        return cast("int", self._http_client.invoke("getNumCardsReviewedToday"))

    def sync(self) -> None:
        """Trigger Anki sync."""
        self._http_client.invoke("sync")
        logger.info("anki_sync_triggered")

    # Async counterparts for methods without them

    async def get_deck_names_async(self, use_cache: bool = True) -> list[str]:
        """Get all deck names (async)."""
        return await self._deck_service.get_deck_names_async(use_cache)

    async def get_model_names_async(self, use_cache: bool = True) -> list[str]:
        """Get all note type (model) names (async)."""
        return await self._model_service.get_model_names_async(use_cache)

    async def get_model_field_names_async(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a note type (async)."""
        return await self._model_service.get_model_field_names_async(
            model_name, use_cache
        )

    async def can_add_notes_async(self, notes: list[dict[str, Any]]) -> list[bool]:
        """Check if notes can be added (async)."""
        return await self._note_service.can_add_notes_async(notes)

    async def store_media_file_async(self, filename: str, data: str) -> str:
        """Store a media file in Anki's media collection (async)."""
        return await self._media_service.store_media_file_async(filename, data)

    async def suspend_cards_async(self, card_ids: list[int]) -> None:
        """Suspend cards by ID (async)."""
        await self._card_service.suspend_cards_async(card_ids)

    async def unsuspend_cards_async(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID (async)."""
        await self._card_service.unsuspend_cards_async(card_ids)

    async def sync_async(self) -> None:
        """Trigger Anki sync (async)."""
        await self._http_client.invoke_async("sync")
        logger.info("anki_sync_triggered")

    async def check_connection_async(self) -> bool:
        """Check if AnkiConnect is accessible (async)."""
        return await self._http_client.invoke_async("version") is not None

    # IAnkiClient interface implementation

    def check_connection(self) -> bool:
        """Check if AnkiConnect is accessible and healthy."""
        return self._http_client.check_connection()

    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID."""
        return self._card_service.get_note_id_from_card_id(card_id)

    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID."""
        return self._card_service.get_card_ids_from_note_id(note_id)

    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck."""
        return self._deck_service.get_deck_stats(deck_name)

    def close(self) -> None:
        """Close HTTP sessions and cleanup resources."""
        self._http_client.close()

    async def aclose(self) -> None:
        """Async cleanup for async contexts."""
        await self._http_client.aclose()

    def __enter__(self) -> "AnkiClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit with cleanup."""
        self.close()
        return False

    async def __aenter__(self) -> "AnkiClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Async context manager exit with cleanup."""
        await self.aclose()
        return False

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            # Only close sync client in __del__ to avoid issues with event loop
            if hasattr(self, "_http_client") and self._http_client:
                self._http_client.close()


__all__ = ["AnkiClient", "AnkiConnectError"]
