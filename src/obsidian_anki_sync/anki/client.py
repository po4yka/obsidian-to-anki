"""AnkiConnect HTTP API client."""

from typing import Any, Optional

import httpx  # type: ignore

from ..utils.logging import get_logger
from ..utils.retry import retry

logger = get_logger(__name__)


class AnkiConnectError(Exception):
    """Error from AnkiConnect API."""
    pass


class AnkiClient:
    """Client for AnkiConnect HTTP API."""

    def __init__(self, url: str, timeout: float = 30.0):
        """
        Initialize client.

        Args:
            url: AnkiConnect URL
            timeout: Request timeout in seconds
        """
        self.url = url
        # Configure connection pooling for better performance
        self.session = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
        )
        logger.info("anki_client_initialized", url=url)

    @retry(
        max_attempts=3,
        initial_delay=1.0,
        exceptions=(httpx.HTTPError, httpx.TimeoutException, AnkiConnectError)
    )
    def invoke(self, action: str, params: Optional[dict] = None) -> Any:
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
        payload = {
            "action": action,
            "version": 6,
            "params": params or {}
        }

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
            raise AnkiConnectError(
                f"HTTP error calling AnkiConnect: {e}"
            )

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
        return self.invoke("findNotes", {"query": query})

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
        return self.invoke("notesInfo", {"notes": note_ids})

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
            "options": {
                "allowDuplicate": False
            }
        }
        if guid:
            note_payload["guid"] = guid

        result = self.invoke("addNote", {
            "note": note_payload
        })

        logger.info("note_added", note_id=result, deck=deck, note_type=note_type)
        return result

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """
        Update note fields.

        Args:
            note_id: Note ID
            fields: New field values
        """
        self.invoke("updateNoteFields", {
            "note": {
                "id": note_id,
                "fields": fields
            }
        })

        logger.info("note_updated", note_id=note_id)

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """
        Add tags to notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to add
        """
        self.invoke("addTags", {
            "notes": note_ids,
            "tags": tags
        })

        logger.info("tags_added", note_ids=note_ids, tags=tags)

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """
        Remove tags from notes.

        Args:
            note_ids: List of note IDs
            tags: Space-separated tags to remove
        """
        self.invoke("removeTags", {
            "notes": note_ids,
            "tags": tags
        })

        logger.info("tags_removed", note_ids=note_ids, tags=tags)

    def replace_tags(self, note_ids: list[int], tag_to_replace: str, replace_with: str) -> None:
        """
        Replace tags in notes.

        Args:
            note_ids: List of note IDs
            tag_to_replace: Tag to replace
            replace_with: New tag
        """
        self.invoke("replaceTags", {
            "notes": note_ids,
            "tag_to_replace": tag_to_replace,
            "replace_with": replace_with
        })

        logger.info("tags_replaced", note_ids=note_ids, old=tag_to_replace, new=replace_with)

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
        return self.invoke("deckNames")

    def get_model_names(self) -> list[str]:
        """Get all note type (model) names."""
        return self.invoke("modelNames")

    def get_model_field_names(self, model_name: str) -> list[str]:
        """
        Get field names for a note type.

        Args:
            model_name: Note type name

        Returns:
            List of field names
        """
        return self.invoke("modelFieldNames", {"modelName": model_name})

    def can_add_notes(self, notes: list[dict]) -> list[bool]:
        """
        Check if notes can be added (no duplicates).

        Args:
            notes: List of note objects to check

        Returns:
            List of booleans indicating if each note can be added
        """
        return self.invoke("canAddNotes", {"notes": notes})

    def store_media_file(self, filename: str, data: str) -> str:
        """
        Store a media file in Anki's media folder.

        Args:
            filename: Filename to store as
            data: Base64-encoded file data

        Returns:
            Stored filename
        """
        return self.invoke("storeMediaFile", {
            "filename": filename,
            "data": data
        })

    def gui_browse(self, query: str) -> list[int]:
        """
        Open browser with search query.

        Args:
            query: Search query

        Returns:
            List of note IDs shown in browser
        """
        return self.invoke("guiBrowse", {"query": query})

    def sync(self) -> None:
        """Trigger Anki sync."""
        self.invoke("sync")
        logger.info("anki_sync_triggered")

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self) -> 'AnkiClient':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
