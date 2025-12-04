"""AnkiConnect HTTP API client."""

import contextlib
import time
from types import TracebackType
from typing import Any, Literal, cast

import httpx

from obsidian_anki_sync.domain.interfaces.anki_client import IAnkiClient
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.retry import retry

logger = get_logger(__name__)


class AnkiClient(IAnkiClient):
    """Client for AnkiConnect HTTP API.

    Note: This client uses synchronous httpx.Client for compatibility with
    existing sync code paths. It should only be used in synchronous contexts.
    For async contexts, consider using asyncio.run() wrapper or migrating to
    async HTTP client in the future.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
        enable_health_checks: bool = True,
        async_runner: AsyncioRunner | None = None,
    ):
        """
        Initialize client.

        Args:
            url: AnkiConnect URL
            timeout: Request timeout in seconds
            enable_health_checks: Whether to perform periodic health checks

        Note:
            Uses synchronous httpx.Client. This is intentional for compatibility
            with existing sync code. Do not use in async contexts without proper
            synchronization (e.g., asyncio.run() wrapper).
        """
        self.url = url
        self.enable_health_checks = enable_health_checks
        self._last_health_check = 0.0
        self._health_check_interval = 60.0  # Check health every 60 seconds
        self._is_healthy = True
        self._async_runner = async_runner or AsyncioRunner.get_global()

        # Configure connection pooling for better performance
        # Using sync client for compatibility with existing sync code paths
        self.session = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            ),
        )
        self._async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            ),
        )
        logger.info(
            "anki_client_initialized", url=url, health_checks=enable_health_checks
        )

    def _check_health(self) -> bool:
        """
        Check if AnkiConnect is healthy and responding.

        Returns:
            True if healthy, False otherwise
        """
        if not self.enable_health_checks:
            return True

        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_healthy

        try:
            # Simple health check using version action (direct call to avoid recursion)
            payload = {"action": "version", "version": 6, "params": {}}
            response = self.session.post(self.url, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                msg = f"AnkiConnect error: {result['error']}"
                raise AnkiConnectError(msg)
            self._is_healthy = True
        except Exception:
            self._is_healthy = False
            logger.warning("anki_health_check_failed", url=self.url)

        self._last_health_check = current_time
        return self._is_healthy

    @retry(
        max_attempts=3,
        initial_delay=1.0,
        exceptions=(httpx.HTTPError, httpx.TimeoutException, AnkiConnectError),
    )
    def invoke(self, action: str, params: dict | None = None) -> Any:
        """
        Invoke AnkiConnect action.

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Action result

        Raises:
            AnkiConnectError: If the action fails
        """
        # Perform health check before making requests
        if not self._check_health():
            logger.warning(
                "anki_unhealthy_skipping_request", action=action, url=self.url
            )
            msg = "AnkiConnect is not responding - check if Anki is running"
            raise AnkiConnectError(msg)

        return self._async_runner.run(self.invoke_async(action, params))

    async def invoke_async(self, action: str, params: dict | None = None) -> Any:
        payload = {"action": action, "version": 6, "params": params or {}}

        logger.debug("anki_invoke", action=action)

        try:
            response = await self._async_client.post(self.url, json=payload)
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            self._is_healthy = False  # Mark as unhealthy on connection errors
            msg = f"Connection error to AnkiConnect: {e}"
            raise AnkiConnectError(msg)
        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code} from AnkiConnect: {e}"
            raise AnkiConnectError(msg)
        except httpx.HTTPError as e:
            msg = f"HTTP error calling AnkiConnect: {e}"
            raise AnkiConnectError(msg)

        try:
            result = response.json()
        except (ValueError, TypeError) as e:
            msg = f"Invalid JSON response: {e}"
            raise AnkiConnectError(msg)

        if result.get("error"):
            msg = f"AnkiConnect error: {result['error']}"
            raise AnkiConnectError(msg)

        return result.get("result")

    def find_notes(self, query: str) -> list[int]:
        """
        Find notes matching query.

        Args:
            query: Anki search query

        Returns:
            List of note IDs
        """
        return cast("list[int]", self.invoke("findNotes", {"query": query}))

    def _build_note_payload(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any | None] | None = None,
        guid: str | None = None,
    ) -> dict[str, Any]:
        note_payload: dict[str, Any] = {
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "options": options or {"allowDuplicate": False},
        }
        if tags:
            clean_tags = [t for t in tags if t is not None]
            if clean_tags:
                note_payload["tags"] = clean_tags
        if guid:
            note_payload["guid"] = guid
        return note_payload

    def notes_info(self, note_ids: list[int]) -> list[dict]:
        """
        Get information about notes.

        Args:
            note_ids: List of note IDs

        Returns:
            List of note info dicts
        """
        if not note_ids:
            return []
        return cast(
            "list[dict[Any, Any]]", self.invoke("notesInfo", {"notes": note_ids})
        )

    async def notes_info_async(self, note_ids: list[int]) -> list[dict]:
        if not note_ids:
            return []
        return cast(
            "list[dict[Any, Any]]",
            await self.invoke_async("notesInfo", {"notes": note_ids}),
        )

    def cards_info(self, card_ids: list[int]) -> list[dict]:
        """
        Get information about cards.

        Args:
            card_ids: List of card IDs

        Returns:
            List of card info dicts
        """
        if not card_ids:
            return []
        return cast(
            "list[dict[Any, Any]]", self.invoke("cardsInfo", {"cards": card_ids})
        )

    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any | None] | None = None,
        guid: str | None = None,
    ) -> int:
        """
        Add a new note.

        Args:
            deck_name: Deck name
            model_name: Note type name
            fields: Field values
            tags: Optional tags
            options: Optional note options

        Returns:
            Note ID
        """
        note_payload = self._build_note_payload(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

        result = cast("int", self.invoke("addNote", {"note": note_payload}))

        logger.info("note_added", note_id=result, deck=deck_name, note_type=model_name)
        return result

    async def add_note_async(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any | None] | None = None,
        guid: str | None = None,
    ) -> int:
        note_payload = self._build_note_payload(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

        result = cast("int", await self.invoke_async("addNote", {"note": note_payload}))
        logger.info("note_added", note_id=result, deck=deck_name, note_type=model_name)
        return result

    def add_notes(
        self,
        notes: list[dict[str, Any]],
    ) -> list[int | None]:
        """
        Add multiple notes in a single batch operation.

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
        if not notes:
            return []

        result = cast("list[int | None]", self.invoke("addNotes", {"notes": notes}))

        successful = sum(1 for note_id in result if note_id is not None)
        failed = len(result) - successful

        logger.info(
            "notes_added_batch",
            total=len(notes),
            successful=successful,
            failed=failed,
        )

        return result

    async def add_notes_async(self, notes: list[dict[str, Any]]) -> list[int | None]:
        if not notes:
            return []

        result = cast(
            "list[int | None]", await self.invoke_async("addNotes", {"notes": notes})
        )

        return result

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """
        Update note fields.

        Args:
            note_id: Note ID
            fields: New field values
        """
        self.invoke("updateNoteFields", {"note": {"id": note_id, "fields": fields}})

        logger.info("note_updated", note_id=note_id)

    async def update_note_fields_async(
        self, note_id: int, fields: dict[str, str]
    ) -> None:
        await self.invoke_async(
            "updateNoteFields", {"note": {"id": note_id, "fields": fields}}
        )

    def update_notes_fields(self, updates: list[dict[str, Any]]) -> list[bool]:
        """
        Update multiple notes' fields in a single batch operation using multi action.

        Args:
            updates: List of update dicts, each containing:
                - id: int (note ID)
                - fields: dict[str, str]

        Returns:
            List of booleans indicating success for each update
        """
        if not updates:
            return []

        # Use AnkiConnect's multi action to batch update operations
        actions = [
            {"action": "updateNoteFields", "params": {"note": update}}
            for update in updates
        ]

        try:
            # Execute all updates in a single multi request
            results = cast(
                "list[dict[str, Any]]",
                self.invoke("multi", {"actions": actions}),
            )

            # Extract success status from results
            success_list = []
            for i, result in enumerate(results):
                if result.get("error") is None:
                    success_list.append(True)
                else:
                    logger.warning(
                        "batch_update_failed",
                        note_id=updates[i].get("id"),
                        error=result.get("error"),
                    )
                    success_list.append(False)

            successful = sum(1 for r in success_list if r)
            logger.info(
                "notes_updated_batch",
                total=len(updates),
                successful=successful,
                failed=len(updates) - successful,
            )

            return success_list

        except Exception as e:
            # If multi fails, fall back to individual updates
            logger.warning(
                "batch_update_multi_failed",
                error=str(e),
                falling_back_to_individual=True,
            )
            fallback_results: list[bool] = []
            for update in updates:
                try:
                    self.invoke("updateNoteFields", {"note": update})
                    fallback_results.append(True)
                except Exception as update_error:
                    logger.warning(
                        "batch_update_failed",
                        note_id=update.get("id"),
                        error=str(update_error),
                    )
                    fallback_results.append(False)

            return fallback_results

    async def update_notes_fields_async(
        self, updates: list[dict[str, Any]]
    ) -> list[bool]:
        if not updates:
            return []

        actions = [
            {"action": "updateNoteFields", "params": {"note": update}}
            for update in updates
        ]

        try:
            results = cast(
                "list[dict[str, Any]]",
                await self.invoke_async("multi", {"actions": actions}),
            )
            return [result.get("error") is None for result in results]
        except Exception:
            return [False] * len(updates)

    def update_note_tags(self, note_id: int, tags: list[str]) -> None:
        """
        Synchronize tags for a single note by applying minimal add/remove operations.

        Args:
            note_id: Note ID
            tags: Desired set of tags
        """
        desired_tags = sorted({tag for tag in tags if tag})

        note_info = self.notes_info([note_id])
        if not note_info:
            msg = f"Note not found for tag update: {note_id}"
            raise AnkiConnectError(msg)

        current_tags = set(note_info[0].get("tags", []))
        desired_set = set(desired_tags)

        to_add = sorted(desired_set - current_tags)
        to_remove = sorted(current_tags - desired_set)

        if to_add:
            self.add_tags([note_id], " ".join(to_add))
        if to_remove:
            self.remove_tags([note_id], " ".join(to_remove))

        logger.info(
            "note_tags_updated",
            note_id=note_id,
            added=to_add,
            removed=to_remove,
        )

    async def update_note_tags_async(self, note_id: int, tags: list[str]) -> None:
        desired_tags = sorted({tag for tag in tags if tag})
        await self.invoke_async(
            "replaceTags", {"notes": [note_id], "tags": " ".join(desired_tags)}
        )

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """
        Add tags to notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to add
        """
        self.invoke("addTags", {"notes": note_ids, "tags": tags})

        logger.info("tags_added", note_ids=note_ids, tags=tags)

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """
        Remove tags from notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to remove
        """
        self.invoke("removeTags", {"notes": note_ids, "tags": tags})

        logger.info("tags_removed", note_ids=note_ids, tags=tags)

    def update_notes_tags(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """
        Update tags for multiple notes in a batch operation.

        Args:
            note_tag_pairs: List of (note_id, tags) tuples

        Returns:
            List of booleans indicating success for each update
        """
        if not note_tag_pairs:
            return []

        # Group by operation type (add/remove) for efficiency
        # For now, use multi action to batch tag updates
        actions = []
        for note_id, tags in note_tag_pairs:
            # Get current tags first (we'll need to fetch them)
            # For simplicity, use replaceTags action if available
            # Otherwise, use addTags/removeTags combination
            desired_tags = sorted({tag for tag in tags if tag})
            if desired_tags:
                actions.append(
                    {
                        "action": "replaceTags",
                        "params": {
                            "notes": [note_id],
                            "tags": " ".join(desired_tags),
                        },
                    }
                )

        if not actions:
            return [True] * len(note_tag_pairs)

        try:
            results = cast(
                "list[dict[str, Any]]",
                self.invoke("multi", {"actions": actions}),
            )

            success_list = []
            for result in results:
                success_list.append(result.get("error") is None)

            successful = sum(1 for r in success_list if r)
            logger.info(
                "notes_tags_updated_batch",
                total=len(note_tag_pairs),
                successful=successful,
                failed=len(note_tag_pairs) - successful,
            )

            return success_list

        except Exception as e:
            # Fall back to individual updates
            logger.warning(
                "batch_tags_update_failed",
                error=str(e),
                falling_back_to_individual=True,
            )
            fallback_results: list[bool] = []
            for note_id, tags in note_tag_pairs:
                try:
                    self.update_note_tags(note_id, tags)
                    fallback_results.append(True)
                except Exception:
                    fallback_results.append(False)

            return fallback_results

    async def update_notes_tags_async(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        if not note_tag_pairs:
            return []

        actions = []
        for note_id, tags in note_tag_pairs:
            desired_tags = sorted({tag for tag in tags if tag})
            if desired_tags:
                actions.append(
                    {
                        "action": "replaceTags",
                        "params": {"notes": [note_id], "tags": " ".join(desired_tags)},
                    }
                )

        if not actions:
            return [True] * len(note_tag_pairs)

        try:
            results = cast(
                "list[dict[str, Any]]",
                await self.invoke_async("multi", {"actions": actions}),
            )
            return [result.get("error") is None for result in results]
        except Exception:
            return [False] * len(note_tag_pairs)

    def delete_notes(self, note_ids: list[int]) -> None:
        """
        Delete notes.

        Args:
            note_ids: List of note IDs to delete
        """
        if not note_ids:
            return

        self.invoke("deleteNotes", {"notes": note_ids})
        logger.info("notes_deleted", count=len(note_ids))

    async def delete_notes_async(self, note_ids: list[int]) -> None:
        if not note_ids:
            return
        await self.invoke_async("deleteNotes", {"notes": note_ids})

    def get_deck_names(self) -> list[str]:
        """Get all deck names."""
        return cast("list[str]", self.invoke("deckNames"))

    def get_model_names(self) -> list[str]:
        """Get all note type (model) names."""
        return cast("list[str]", self.invoke("modelNames"))

    def get_model_field_names(self, model_name: str) -> list[str]:
        """
        Get field names for a note type.

        Args:
            model_name: Note type name

        Returns:
            List of field names
        """
        return cast(
            "list[str]", self.invoke("modelFieldNames", {"modelName": model_name})
        )

    def can_add_notes(self, notes: list[dict[str, Any]]) -> list[bool]:
        """
        Check if notes can be added (duplicate prevention).

        Args:
            notes: List of note payloads to check, each containing:
                - deckName: str
                - modelName: str
                - fields: dict[str, str]

        Returns:
            List of booleans indicating whether each note can be added
        """
        if not notes:
            return []

        return cast("list[bool]", self.invoke("canAddNotes", {"notes": notes}))

    def store_media_file(self, filename: str, data: str) -> str:
        """
        Store a media file in Anki's media collection.

        Args:
            filename: Name of the file to store
            data: Base64-encoded file data

        Returns:
            The filename as stored in Anki (may be modified)
        """
        return cast(
            "str", self.invoke("storeMediaFile", {"filename": filename, "data": data})
        )

    def suspend_cards(self, card_ids: list[int]) -> None:
        """
        Suspend cards by ID.

        Args:
            card_ids: List of card IDs to suspend
        """
        if not card_ids:
            return

        self.invoke("suspend", {"cards": card_ids})
        logger.info("cards_suspended", count=len(card_ids))

    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """
        Unsuspend cards by ID.

        Args:
            card_ids: List of card IDs to unsuspend
        """
        if not card_ids:
            return

        self.invoke("unsuspend", {"cards": card_ids})
        logger.info("cards_unsuspended", count=len(card_ids))

    def gui_browse(self, query: str) -> list[int]:
        """
        Open Anki browser with a search query.

        Args:
            query: Anki search query

        Returns:
            List of note IDs matching the query
        """
        return cast("list[int]", self.invoke("guiBrowse", {"query": query}))

    def get_collection_stats(self) -> str:
        """
        Get collection statistics as HTML.

        Returns:
            HTML string containing collection statistics
        """
        return cast("str", self.invoke("getCollectionStatsHtml"))

    def get_model_names_and_ids(self) -> dict[str, int]:
        """
        Get note type names and their IDs.

        Returns:
            Dictionary mapping model names to model IDs
        """
        return cast("dict[str, int]", self.invoke("modelNamesAndIds"))

    def get_num_cards_reviewed_today(self) -> int:
        """
        Get the number of cards reviewed today.

        Returns:
            Number of cards reviewed in the current day
        """
        return cast("int", self.invoke("getNumCardsReviewedToday"))

    def sync(self) -> None:
        """Trigger Anki sync."""
        self.invoke("sync")
        logger.info("anki_sync_triggered")

    # IAnkiClient interface implementation

    def check_connection(self) -> bool:
        """Check if AnkiConnect is accessible and healthy."""
        return self._check_health()

    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID."""
        return cast("int", self.invoke("cardsToNotes", {"cards": [card_id]})[0])

    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID."""
        return cast("list[int]", self.invoke("notesToCards", {"notes": [note_id]})[0])

    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck.

        Note:
            We implement this manually using findCards queries because getDeckStats
            is not available in the standard AnkiConnect API (it's from a fork).
            The returned dictionary mimics the structure of getDeckStats but might
            be slightly different.

        Args:
            deck_name: Deck name

        Returns:
            Dictionary with statistics:
            {
                "deck_id": 0,  # Placeholder
                "name": deck_name,
                "total_cards": int,
                "new_cards": int,
                "review_cards": int,  # Due for review
                "learned_cards": int, # In learning
            }
        """
        try:
            # Try to use getDeckStats if available (e.g. if using the fork)
            return cast(
                "dict[str, Any]", self.invoke("getDeckStats", {"decks": [deck_name]})
            )
        except AnkiConnectError:
            # Fallback to manual calculation
            logger.debug("getDeckStats_not_supported_falling_back", deck=deck_name)

            # Using findCards which is faster than findNotes for stats
            total_cards = len(self.invoke("findCards", {"query": f'deck:"{deck_name}"'}))
            new_cards = len(self.invoke("findCards", {"query": f'deck:"{deck_name}" is:new'}))
            learn_cards = len(self.invoke("findCards", {"query": f'deck:"{deck_name}" is:learn'}))
            due_cards = len(self.invoke("findCards", {"query": f'deck:"{deck_name}" is:due'}))

            # Note: is:due includes learn cards that are due, but we separate them roughly here
            # Ideally we'd use getDeckStats if possible.

            # Construct a response similar to what's expected
            return {
                "deck_id": 0,  # We don't have the deck ID easily available
                "name": deck_name,
                "total_in_deck": total_cards,
                "new_count": new_cards,
                "learn_count": learn_cards,
                "review_count": due_cards
            }

    def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if hasattr(self, "session") and self.session:
            try:
                self.session.close()
                logger.debug("anki_client_closed", url=self.url)
            except Exception as e:
                logger.warning("anki_client_cleanup_failed", url=self.url, error=str(e))

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

    def __del__(self) -> None:
        """Cleanup on deletion."""
        with contextlib.suppress(Exception):
            self.close()
