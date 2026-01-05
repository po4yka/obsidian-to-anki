"""Service for Anki tag operations."""

from typing import Any, cast

from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.anki.services.anki_note_service import AnkiNoteService
from obsidian_anki_sync.domain.interfaces.anki_tag_service import IAnkiTagService
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiTagService(IAnkiTagService):
    """Service for Anki tag operations.

    Handles all tag-related operations including adding, removing,
    and synchronizing tags on notes, with support for both individual
    and batch operations.
    """

    def __init__(self, http_client: AnkiHttpClient, note_service: AnkiNoteService):
        """
        Initialize tag service.

        Args:
            http_client: HTTP client for AnkiConnect communication
            note_service: Service for note operations
        """
        self._http_client = http_client
        self._note_service = note_service
        logger.debug("anki_tag_service_initialized")

    def update_note_tags(self, note_id: int, tags: list[str]) -> None:
        """Synchronize tags for a single note by applying minimal add/remove operations."""
        desired_tags = sorted({tag for tag in tags if tag})

        note_info = self._note_service.notes_info([note_id])
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
        """Synchronize tags for a single note by applying minimal add/remove operations (async)."""
        desired_tags = sorted({tag for tag in tags if tag})
        await self._http_client.invoke_async(
            "replaceTags", {"notes": [note_id], "tags": " ".join(desired_tags)}
        )

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Add tags to notes."""
        self._http_client.invoke("addTags", {"notes": note_ids, "tags": tags})
        logger.info("tags_added", note_ids=note_ids, tags=tags)

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remove tags from notes."""
        self._http_client.invoke("removeTags", {"notes": note_ids, "tags": tags})
        logger.info("tags_removed", note_ids=note_ids, tags=tags)

    def update_notes_tags(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """Update tags for multiple notes in a batch operation."""
        if not note_tag_pairs:
            return []

        # Group notes by their desired tag set to reduce API calls
        # Key: frozenset of tags, Value: list of note_ids
        tag_groups: dict[frozenset[str], list[int]] = {}
        note_to_tagset: dict[int, frozenset[str]] = {}
        empty_tag_notes: list[int] = []

        for note_id, tags in note_tag_pairs:
            desired_tags = frozenset(tag for tag in tags if tag)
            note_to_tagset[note_id] = desired_tags
            if desired_tags:
                if desired_tags not in tag_groups:
                    tag_groups[desired_tags] = []
                tag_groups[desired_tags].append(note_id)
            else:
                empty_tag_notes.append(note_id)

        # Build actions - one per unique tag set (much more efficient)
        actions: list[dict[str, Any]] = []
        tagset_to_action_idx: dict[frozenset[str], int] = {}

        for tagset, note_ids in tag_groups.items():
            tagset_to_action_idx[tagset] = len(actions)
            actions.append(
                {
                    "action": "replaceTags",
                    "params": {
                        "notes": note_ids,
                        "tags": " ".join(sorted(tagset)),
                    },
                }
            )

        if not actions:
            # All notes had empty tags
            return [True] * len(note_tag_pairs)

        logger.debug(
            "batch_tags_optimized",
            original_count=len(note_tag_pairs),
            grouped_count=len(actions),
            reduction_pct=round((1 - len(actions) / len(note_tag_pairs)) * 100, 1),
        )

        try:
            results = cast(
                "list[dict[str, Any]]",
                self._http_client.invoke("multi", {"actions": actions}),
            )

            # Map results back to original note order
            success_list: list[bool] = []
            for note_id, tags in note_tag_pairs:
                tagset = note_to_tagset[note_id]
                if not tagset:
                    # Empty tags always succeed (no action taken)
                    success_list.append(True)
                else:
                    action_idx = tagset_to_action_idx[tagset]
                    success_list.append(results[action_idx].get("error") is None)

            successful = sum(1 for r in success_list if r)
            logger.info(
                "notes_tags_updated_batch",
                total=len(note_tag_pairs),
                successful=successful,
                failed=len(note_tag_pairs) - successful,
                actions_executed=len(actions),
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
                except AnkiConnectError:
                    fallback_results.append(False)

            return fallback_results

    async def update_notes_tags_async(
        self, note_tag_pairs: list[tuple[int, list[str]]]
    ) -> list[bool]:
        """Update tags for multiple notes in a batch operation (async)."""
        if not note_tag_pairs:
            return []

        # Group notes by their desired tag set to reduce API calls
        tag_groups: dict[frozenset[str], list[int]] = {}
        note_to_tagset: dict[int, frozenset[str]] = {}

        for note_id, tags in note_tag_pairs:
            desired_tags = frozenset(tag for tag in tags if tag)
            note_to_tagset[note_id] = desired_tags
            if desired_tags:
                if desired_tags not in tag_groups:
                    tag_groups[desired_tags] = []
                tag_groups[desired_tags].append(note_id)

        # Build actions - one per unique tag set
        actions: list[dict[str, Any]] = []
        tagset_to_action_idx: dict[frozenset[str], int] = {}

        for tagset, note_ids in tag_groups.items():
            tagset_to_action_idx[tagset] = len(actions)
            actions.append(
                {
                    "action": "replaceTags",
                    "params": {
                        "notes": note_ids,
                        "tags": " ".join(sorted(tagset)),
                    },
                }
            )

        if not actions:
            return [True] * len(note_tag_pairs)

        try:
            results = cast(
                "list[dict[str, Any]]",
                await self._http_client.invoke_async("multi", {"actions": actions}),
            )

            # Map results back to original note order
            success_list: list[bool] = []
            for note_id, tags in note_tag_pairs:
                tagset = note_to_tagset[note_id]
                if not tagset:
                    success_list.append(True)
                else:
                    action_idx = tagset_to_action_idx[tagset]
                    if results[action_idx].get("error") is not None:
                        logger.warning(
                            "batch_tags_async_failed",
                            note_id=note_id,
                            error=results[action_idx].get("error"),
                        )
                        success_list.append(False)
                    else:
                        success_list.append(True)

            successful = sum(1 for r in success_list if r)
            logger.info(
                "notes_tags_updated_batch_async",
                total=len(note_tag_pairs),
                successful=successful,
                failed=len(note_tag_pairs) - successful,
                actions_executed=len(actions),
            )
            return success_list

        except Exception as e:
            logger.error(
                "batch_tags_async_multi_failed",
                error=str(e),
                error_type=type(e).__name__,
                total=len(note_tag_pairs),
            )
            return [False] * len(note_tag_pairs)
