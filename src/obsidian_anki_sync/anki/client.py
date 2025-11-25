"""AnkiConnect HTTP API client."""

from types import TracebackType
from typing import Any, Literal, cast

import httpx

from ..exceptions import AnkiConnectError
from ..utils.logging import get_logger
from ..utils.retry import retry

logger = get_logger(__name__)


class AnkiClient:
    """Client for AnkiConnect HTTP API.

    Note: This client uses synchronous httpx.Client for compatibility with
    existing sync code paths. It should only be used in synchronous contexts.
    For async contexts, consider using asyncio.run() wrapper or migrating to
    async HTTP client in the future.
    """

    def __init__(self, url: str, timeout: float = 30.0):
        """
        Initialize client.

        Args:
            url: AnkiConnect URL
            timeout: Request timeout in seconds

        Note:
            Uses synchronous httpx.Client. This is intentional for compatibility
            with existing sync code. Do not use in async contexts without proper
            synchronization (e.g., asyncio.run() wrapper).
        """
        self.url = url
        # Configure connection pooling for better performance
        # Using sync client for compatibility with existing sync code paths
        self.session = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
            ),
        )
        logger.info("anki_client_initialized", url=url)

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
        payload = {"action": action, "version": 6, "params": params or {}}

        logger.debug("anki_invoke", action=action)

        try:
            response = self.session.post(self.url, json=payload)
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            raise AnkiConnectError(f"Connection error to AnkiConnect: {e}")
        except httpx.HTTPStatusError as e:
            raise AnkiConnectError(
                f"HTTP {e.response.status_code} from AnkiConnect: {e}"
            )
        except httpx.HTTPError as e:
            raise AnkiConnectError(f"HTTP error calling AnkiConnect: {e}")

        try:
            result = response.json()
        except (ValueError, TypeError) as e:
            raise AnkiConnectError(f"Invalid JSON response: {e}")

        if result.get("error"):
            raise AnkiConnectError(f"AnkiConnect error: {result['error']}")

        return result.get("result")

    def find_notes(self, query: str) -> list[int]:
        """
        Find notes matching query.

        Args:
            query: Anki search query

        Returns:
            List of note IDs
        """
        return cast(list[int], self.invoke("findNotes", {"query": query}))

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
        return cast(list[dict[Any, Any]], self.invoke("notesInfo", {"notes": note_ids}))

    def add_note(
        self,
        deck: str,
        note_type: str,
        fields: dict[str, str],
        tags: list[str],
        guid: str | None = None,
    ) -> int:
        """
        Add a new note.

        Args:
            deck: Deck name
            note_type: Note type name
            fields: Field values
            tags: Tags

        Returns:
            Note ID
        """
        note_payload = {
            "deckName": deck,
            "modelName": note_type,
            "fields": fields,
            "tags": tags,
            "options": {"allowDuplicate": False},
        }
        if guid:
            note_payload["guid"] = guid

        result = cast(int, self.invoke("addNote", {"note": note_payload}))

        logger.info("note_added", note_id=result, deck=deck, note_type=note_type)
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

        result = cast(list[int | None], self.invoke("addNotes", {"notes": notes}))

        successful = sum(1 for note_id in result if note_id is not None)
        failed = len(result) - successful

        logger.info(
            "notes_added_batch",
            total=len(notes),
            successful=successful,
            failed=failed,
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
                list[dict[str, Any]],
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
            results = []
            for update in updates:
                try:
                    self.invoke("updateNoteFields", {"note": update})
                    results.append(True)
                except Exception as update_error:
                    logger.warning(
                        "batch_update_failed",
                        note_id=update.get("id"),
                        error=str(update_error),
                    )
                    results.append(False)

            return results

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
            raise AnkiConnectError(f"Note not found for tag update: {note_id}")

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
                list[dict[str, Any]],
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
            results = []
            for note_id, tags in note_tag_pairs:
                try:
                    self.update_note_tags(note_id, tags)
                    results.append(True)
                except Exception:
                    results.append(False)

            return results

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

    def get_deck_names(self) -> list[str]:
        """Get all deck names."""
        return cast(list[str], self.invoke("deckNames"))

    def get_model_names(self) -> list[str]:
        """Get all note type (model) names."""
        return cast(list[str], self.invoke("modelNames"))

    def get_model_field_names(self, model_name: str) -> list[str]:
        """
        Get field names for a note type.

        Args:
            model_name: Note type name

        Returns:
            List of field names
        """
        return cast(
            list[str], self.invoke("modelFieldNames", {"modelName": model_name})
        )

    def sync(self) -> None:
        """Trigger Anki sync."""
        self.invoke("sync")
        logger.info("anki_sync_triggered")

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
        try:
            self.close()
        except Exception:
            pass
