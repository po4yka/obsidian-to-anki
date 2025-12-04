"""Change application component for SyncEngine.

Handles applying sync actions to Anki (create, update, delete operations).
"""

import asyncio
from typing import TYPE_CHECKING, Any, cast

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.anki.field_mapper import map_apf_to_anki_fields
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.models import Card, SyncAction
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.sync.transactions import CardOperationError, CardTransaction
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class ChangeApplier:
    """Handles applying sync changes to Anki."""

    def __init__(
        self,
        config: Config,
        state_db: StateDB,
        anki_client: AnkiClient,
        progress_tracker: "ProgressTracker | None" = None,
        stats: dict[str, Any] | None = None,
        async_runner: AsyncioRunner | None = None,
    ):
        """Initialize change applier.

        Args:
            config: Service configuration
            state_db: State database
            anki_client: AnkiConnect client
            progress_tracker: Optional progress tracker
            stats: Statistics dictionary to update
        """
        self.config = config
        self.db = state_db
        self.anki = anki_client
        self.progress = progress_tracker
        self.stats = stats or {}
        self._async_runner = async_runner or AsyncioRunner.get_global()

    def apply_changes(self, changes: list[SyncAction]) -> None:
        """Apply sync actions to Anki.

        Args:
            changes: List of sync actions to apply
        """
        logger.info("applying_changes", count=len(changes))

        # Use batch operations if enabled
        if self.config.enable_batch_operations:
            self.apply_batched(changes)
        else:
            self.apply_sequential(changes)

    def apply_sequential(self, changes: list[SyncAction]) -> None:
        """Apply changes sequentially (original implementation).

        Args:
            changes: List of sync actions to apply
        """
        for action in changes:
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            try:
                if action.type == "create":
                    self.create_card(action.card)
                    self.stats["created"] = self.stats.get("created", 0) + 1
                    if self.progress:
                        self.progress.increment_stat("created")

                elif action.type == "update":
                    if action.anki_guid:
                        self.update_card(action.card, action.anki_guid)
                        self.stats["updated"] = self.stats.get("updated", 0) + 1
                        if self.progress:
                            self.progress.increment_stat("updated")

                elif action.type == "delete":
                    if action.anki_guid:
                        self.delete_card(action.card, action.anki_guid)
                        self.stats["deleted"] = self.stats.get("deleted", 0) + 1
                        if self.progress:
                            self.progress.increment_stat("deleted")

                elif action.type == "restore":
                    self.create_card(action.card)
                    self.stats["restored"] = self.stats.get("restored", 0) + 1
                    if self.progress:
                        self.progress.increment_stat("restored")

                elif action.type == "skip":
                    self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                    if self.progress:
                        self.progress.increment_stat("skipped")

            except CardOperationError as e:
                logger.error(
                    "action_failed_card_operation",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")
            except AnkiConnectError as e:
                logger.error(
                    "action_failed_anki_connect",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "action_failed_unexpected",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                    exc_info=True,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")

    def apply_batched(self, changes: list[SyncAction]) -> None:
        """Apply changes using batch operations for better performance.

        Args:
            changes: List of sync actions to apply
        """
        # Group actions by type
        creates: list[SyncAction] = []
        updates: list[SyncAction] = []
        deletes: list[SyncAction] = []
        restores: list[SyncAction] = []

        for action in changes:
            if action.type == "create":
                creates.append(action)
            elif action.type == "update" and action.anki_guid:
                updates.append(action)
            elif action.type == "delete" and action.anki_guid:
                deletes.append(action)
            elif action.type == "restore":
                restores.append(action)
            elif action.type == "skip":
                self.stats["skipped"] = self.stats.get("skipped", 0) + 1
                if self.progress:
                    self.progress.increment_stat("skipped")

        # Process creates in batches
        if creates or restores:
            all_creates = creates + restores
            self.create_batch(all_creates)

        # Process updates in batches
        if updates:
            self.update_batch(updates)

        # Process deletes
        if deletes:
            self.delete_batch(deletes)

    def create_card(self, card: Card) -> None:
        """Create card in Anki with atomic transaction.

        The transaction ensures atomicity: if verification fails after adding
        to Anki, the note is rolled back (deleted from Anki).

        Args:
            card: Card to create
        """
        logger.info("creating_card", slug=card.slug)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        try:
            with CardTransaction(self.anki, self.db) as txn:
                # Step 1: Add to Anki
                note_id = self.anki.add_note(
                    deck_name=self.config.anki_deck_name,
                    model_name=self._get_anki_model_name(card.note_type),
                    fields=fields,
                    tags=card.tags,
                    guid=card.guid,
                )

                # Register rollback action
                txn.rollback_actions.append(("delete_anki_note", note_id))

                # Step 2: Verify card exists in Anki BEFORE committing
                # This ensures atomicity - if verification fails, we roll back
                if getattr(self.config, "verify_card_creation", True):
                    self._verify_card_exists(card, note_id, fields, card.tags)

                # Step 3: Save to database (only after verification succeeds)
                # Use upsert to handle cases where card was already inserted as "pending"
                self.db.upsert_card_extended(
                    card=card,
                    anki_guid=note_id,
                    fields=fields,
                    tags=card.tags,
                    deck_name=self.config.anki_deck_name,
                    apf_html=card.apf_html,
                    creation_status="success",
                )

                txn.commit()

                logger.info(
                    "card_created_successfully", slug=card.slug, anki_guid=note_id
                )

        except AnkiConnectError as e:
            logger.error("anki_create_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

        except Exception as e:
            logger.error("card_create_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            msg = f"Failed to create card {card.slug}: {e}"
            raise CardOperationError(msg)

    def update_card(self, card: Card, anki_guid: int) -> None:
        """Update card in Anki with atomic transaction.

        The transaction ensures atomicity: if verification fails after updating
        in Anki, the note fields/tags are rolled back to their previous state.

        Args:
            card: Card to update
            anki_guid: Anki note ID
        """
        logger.info("updating_card", slug=card.slug, anki_guid=anki_guid)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        try:
            with CardTransaction(self.anki, self.db) as txn:
                # Get current state for potential rollback
                current_info = self.anki.notes_info([anki_guid])
                old_fields = {}
                old_tags = []
                if current_info and current_info[0]:
                    old_fields = current_info[0].get("fields", {})
                    old_tags = current_info[0].get("tags", [])

                # Step 1: Update in Anki
                self.anki.update_note_fields(anki_guid, fields)
                self.anki.update_note_tags(anki_guid, card.tags)

                # Register rollback
                if current_info and current_info[0]:
                    txn.rollback_actions.append(
                        ("restore_anki_note", anki_guid, old_fields, old_tags)
                    )

                # Step 2: Verify update was applied BEFORE committing
                if getattr(self.config, "verify_card_creation", True):
                    self._verify_update_applied(card, anki_guid)

                # Step 3: Update database (only after verification succeeds)
                self.db.update_card_extended(
                    card=card, fields=fields, tags=card.tags, apf_html=card.apf_html
                )

                txn.commit()
                logger.info("card_updated_successfully", slug=card.slug)

        except AnkiConnectError as e:
            logger.error("anki_update_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

        except Exception as e:
            logger.error("card_update_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            msg = f"Failed to update card {card.slug}: {e}"
            raise CardOperationError(msg)

    def delete_card(self, card: Card, anki_guid: int) -> None:
        """Delete card from Anki.

        Args:
            card: Card to delete
            anki_guid: Anki note ID
        """
        logger.info("deleting_card", slug=card.slug, anki_guid=anki_guid)

        if self.config.delete_mode == "delete":
            # Actually delete from Anki
            self.anki.delete_notes([anki_guid])
        # else: archive mode - just remove from database

        # Remove from database
        self.db.delete_card(card.slug)

    def create_batch(self, actions: list[SyncAction]) -> None:
        """Create multiple cards in batch.

        Args:
            actions: List of create/restore actions
        """
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("creating_cards_batch", total=total, batch_size=batch_size)

        async def _create_notes_async(
            payloads: list[dict[str, Any]], limit: int
        ) -> list[int | None]:
            semaphore = asyncio.Semaphore(limit)

            async def _create_one(payload: dict[str, Any]) -> int | None:
                async with semaphore:
                    return cast(
                        "int | None",
                        await self.anki.invoke_async("addNote", {"note": payload}),
                    )

            return await asyncio.gather(*(_create_one(payload) for payload in payloads))

        async def _update_cards_async(
            updates: list[dict[str, Any]],
            tag_payloads: list[tuple[int, list[str]]],
            limit: int,
        ) -> list[tuple[bool, bool]]:
            semaphore = asyncio.Semaphore(limit)

            async def _update_one(
                update_payload: dict[str, Any], tag_payload: tuple[int, list[str]]
            ) -> tuple[bool, bool]:
                async with semaphore:
                    field_success = True
                    tag_success = True
                    try:
                        await self.anki.update_note_fields_async(
                            update_payload["id"], update_payload["fields"]
                        )
                    except Exception as exc:  # pragma: no cover - network
                        field_success = False
                        logger.error(
                            "batch_field_update_failed_async",
                            note_id=update_payload["id"],
                            error=str(exc),
                        )
                    try:
                        await self.anki.update_note_tags_async(
                            tag_payload[0], tag_payload[1]
                        )
                    except Exception as exc:  # pragma: no cover - network
                        tag_success = False
                        logger.error(
                            "batch_tag_update_failed_async",
                            note_id=tag_payload[0],
                            error=str(exc),
                        )
                    return field_success, tag_success

            return await asyncio.gather(
                *(
                    _update_one(update_payload, tag_payload)
                    for update_payload, tag_payload in zip(updates, tag_payloads)
                )
            )

        for batch_start in range(0, total, batch_size):
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            # Prepare batch payloads
            note_payloads = []
            card_data = []

            for action in batch_actions:
                card = action.card
                logger.debug(
                    "preparing_card_for_anki",
                    slug=card.slug,
                    note_type=card.note_type,
                    apf_html_length=len(card.apf_html) if card.apf_html else 0,
                    apf_html_preview=(card.apf_html[:200] if card.apf_html else "empty"),
                )
                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

                note_payload = {
                    "deckName": self.config.anki_deck_name,
                    "modelName": self._get_anki_model_name(card.note_type),
                    "fields": fields,
                    "tags": card.tags,
                    "options": {"allowDuplicate": False},
                }
                if card.guid:
                    note_payload["guid"] = card.guid

                # Debug: Log payload for empty note check
                if not note_payload["fields"].get("Primary Title") or not note_payload["fields"].get("Primary Key point (code block)"):
                     logger.warning(
                         "sending_potentially_empty_card_to_anki",
                         slug=card.slug,
                         fields_preview={k: v[:50] + "..." if v else "EMPTY" for k,v in note_payload["fields"].items()}
                     )

                note_payloads.append(note_payload)
                card_data.append((card, fields, card.tags, card.apf_html))

            # Batch create in Anki
            try:
                with CardTransaction(self.anki, self.db) as txn:
                    concurrency = max(
                        1,
                        min(
                            self.config.max_concurrent_generations,
                            len(note_payloads),
                        ),
                    )

                    note_ids = self._async_runner.run(
                        _create_notes_async(note_payloads, concurrency)
                    )

                    # Process results
                    successful_cards = []
                    for i, (note_id, (card, fields, tags, apf_html)) in enumerate(
                        zip(note_ids, card_data)
                    ):
                        if note_id is not None:
                            txn.rollback_actions.append(("delete_anki_note", note_id))
                            successful_cards.append(
                                (card, note_id, fields, tags, apf_html)
                            )
                        else:
                            logger.error(
                                "batch_create_failed",
                                slug=card.slug,
                                index=i,
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                            self.db.update_card_status(
                                card.slug, "failed", "Anki batch create returned None"
                            )

                    # Verify cards exist in Anki BEFORE committing
                    # This ensures atomicity - if any verification fails, we roll back all
                    verified_cards = []
                    if getattr(self.config, "verify_card_creation", True):
                        for card, note_id, fields, tags, apf_html in successful_cards:
                            try:
                                self._verify_card_exists(card, note_id, fields, tags)
                                verified_cards.append(
                                    (card, note_id, fields, tags, apf_html)
                                )
                            except CardOperationError as verify_error:
                                # Verification failed - remove from rollback and mark as failed
                                logger.error(
                                    "batch_verify_failed",
                                    slug=card.slug,
                                    note_id=note_id,
                                    error=str(verify_error),
                                )
                                self.stats["errors"] = self.stats.get("errors", 0) + 1
                                if self.progress:
                                    self.progress.increment_stat("errors")
                                self.db.update_card_status(
                                    card.slug, "failed", str(verify_error)
                                )
                                # Remove from rollback actions since card doesn't exist
                                txn.rollback_actions = [
                                    action
                                    for action in txn.rollback_actions
                                    if action[1] != note_id
                                ]
                    else:
                        verified_cards = successful_cards

                    # Batch insert into database (only verified cards)
                    if verified_cards:
                        # insert_cards_batch now handles upserts internally
                        self.db.insert_cards_batch(
                            verified_cards, self.config.anki_deck_name
                        )

                    txn.commit()

                    # Update stats
                    created_count = len(verified_cards)
                    self.stats["created"] = self.stats.get("created", 0) + created_count
                    if self.progress:
                        for _ in range(created_count):
                            self.progress.increment_stat("created")

                    logger.info(
                        "batch_create_completed",
                        batch_start=batch_start,
                        batch_end=batch_end,
                        successful=created_count,
                        failed=len(batch_actions) - created_count,
                    )

            except AnkiConnectError as e:
                logger.error(
                    "batch_create_failed_anki_connect",
                    error=str(e),
                    batch_start=batch_start,
                )
                # Fall back to individual creates
                for action in batch_actions:
                    try:
                        self.create_card(action.card)
                        if action.type == "restore":
                            self.stats["restored"] = self.stats.get("restored", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("restored")
                        else:
                            self.stats["created"] = self.stats.get("created", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("created")
                    except Exception as card_error:
                        logger.error(
                            "individual_create_failed_after_batch",
                            slug=action.card.slug,
                            error=str(card_error),
                        )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        if self.progress:
                            self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "batch_create_failed_unexpected",
                    error=str(e),
                    batch_start=batch_start,
                    exc_info=True,
                )
                # Fall back to individual creates
                for action in batch_actions:
                    try:
                        self.create_card(action.card)
                        if action.type == "restore":
                            self.stats["restored"] = self.stats.get("restored", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("restored")
                        else:
                            self.stats["created"] = self.stats.get("created", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("created")
                    except Exception as card_error:
                        logger.error(
                            "individual_create_failed_after_batch_fallback",
                            slug=action.card.slug,
                            error=str(card_error),
                        )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        if self.progress:
                            self.progress.increment_stat("errors")

    def update_batch(self, actions: list[SyncAction]) -> None:
        """Update multiple cards in batch.

        Args:
            actions: List of update actions
        """
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("updating_cards_batch", total=total, batch_size=batch_size)

        async def _update_cards_async(
            updates: list[dict[str, Any]],
            tag_payloads: list[tuple[int, list[str]]],
            limit: int,
        ) -> list[tuple[bool, bool]]:
            semaphore = asyncio.Semaphore(limit)

            async def _update_one(
                update_payload: dict[str, Any], tag_payload: tuple[int, list[str]]
            ) -> tuple[bool, bool]:
                async with semaphore:
                    field_success = True
                    tag_success = True
                    try:
                        await self.anki.update_note_fields_async(
                            update_payload["id"], update_payload["fields"]
                        )
                    except Exception as exc:  # pragma: no cover - network
                        field_success = False
                        logger.error(
                            "batch_field_update_failed_async",
                            note_id=update_payload["id"],
                            error=str(exc),
                        )
                    try:
                        await self.anki.update_note_tags_async(
                            tag_payload[0], tag_payload[1]
                        )
                    except Exception as exc:  # pragma: no cover - network
                        tag_success = False
                        logger.error(
                            "batch_tag_update_failed_async",
                            note_id=tag_payload[0],
                            error=str(exc),
                        )
                    return field_success, tag_success

            return await asyncio.gather(
                *(
                    _update_one(update_payload, tag_payload)
                    for update_payload, tag_payload in zip(updates, tag_payloads)
                )
            )

        for batch_start in range(0, total, batch_size):
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            # Prepare batch updates
            field_updates = []
            tag_updates = []
            card_data = []

            for action in batch_actions:
                if not action.anki_guid:
                    continue

                card = action.card
                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

                field_updates.append({"id": action.anki_guid, "fields": fields})
                tag_updates.append((action.anki_guid, card.tags))
                card_data.append((card, fields, card.tags))

            if not field_updates:
                continue

            # Batch update in Anki
            try:
                with CardTransaction(self.anki, self.db) as txn:
                    # Get current state for rollback
                    anki_guids = [update["id"] for update in field_updates]
                    current_info = self.anki.notes_info(anki_guids)

                    # Register rollback actions
                    for info in current_info:
                        note_id = info["noteId"]
                        old_fields = info.get("fields", {})
                        old_tags = info.get("tags", [])
                        txn.rollback_actions.append(
                            ("restore_anki_note", note_id, old_fields, old_tags)
                        )

                    concurrency = max(
                        1,
                        min(self.config.max_concurrent_generations, len(field_updates)),
                    )

                    update_results = self._async_runner.run(
                        _update_cards_async(field_updates, tag_updates, concurrency)
                    )
                    field_results = [result[0] for result in update_results]
                    tag_results = [result[1] for result in update_results]

                    # Process results
                    successful_cards = []
                    for i, (
                        field_success,
                        tag_success,
                        (card, fields, tags),
                    ) in enumerate(zip(field_results, tag_results, card_data)):
                        if field_success and tag_success:
                            successful_cards.append((card, fields, tags))
                        else:
                            logger.error(
                                "batch_update_failed",
                                slug=card.slug,
                                field_success=field_success,
                                tag_success=tag_success,
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                            self.db.update_card_status(
                                card.slug,
                                "failed",
                                f"Batch update failed: field={field_success}, tag={tag_success}",
                            )

                    # Batch update database
                    if successful_cards:
                        self.db.update_cards_batch(successful_cards)

                    txn.commit()

                    # Update stats
                    updated_count = len(successful_cards)
                    self.stats["updated"] = self.stats.get("updated", 0) + updated_count
                    if self.progress:
                        for _ in range(updated_count):
                            self.progress.increment_stat("updated")

                    logger.info(
                        "batch_update_completed",
                        batch_start=batch_start,
                        batch_end=batch_end,
                        successful=updated_count,
                        failed=len(batch_actions) - updated_count,
                    )

            except AnkiConnectError as e:
                logger.error(
                    "batch_update_failed_anki_connect",
                    error=str(e),
                    batch_start=batch_start,
                )
                # Fall back to individual updates
                for action in batch_actions:
                    if action.anki_guid:
                        try:
                            self.update_card(action.card, action.anki_guid)
                            self.stats["updated"] = self.stats.get("updated", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("updated")
                        except Exception as card_error:
                            logger.error(
                                "individual_update_failed_after_batch",
                                slug=action.card.slug,
                                error=str(card_error),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "batch_update_failed_unexpected",
                    error=str(e),
                    batch_start=batch_start,
                    exc_info=True,
                )
                # Fall back to individual updates
                for action in batch_actions:
                    if action.anki_guid:
                        try:
                            self.update_card(action.card, action.anki_guid)
                            self.stats["updated"] = self.stats.get("updated", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("updated")
                        except Exception as card_error:
                            logger.error(
                                "individual_update_failed_after_batch_fallback",
                                slug=action.card.slug,
                                error=str(card_error),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")

    def delete_batch(self, actions: list[SyncAction]) -> None:
        """Delete multiple cards in batch.

        Args:
            actions: List of delete actions
        """
        if not actions:
            return

        # Group by delete mode
        anki_guids_to_delete = []
        slugs_to_delete = []

        for action in actions:
            if action.anki_guid:
                slugs_to_delete.append(action.card.slug)
                if self.config.delete_mode == "delete":
                    anki_guids_to_delete.append(action.anki_guid)

        # Batch delete from Anki
        if anki_guids_to_delete:
            try:
                self.anki.delete_notes(anki_guids_to_delete)
                logger.info("batch_delete_anki", count=len(anki_guids_to_delete))
            except AnkiConnectError as e:
                logger.error("batch_delete_anki_failed_connect_error", error=str(e))
                # Fall back to individual deletes
                for action in actions:
                    if action.anki_guid:
                        try:
                            self.delete_card(action.card, action.anki_guid)
                            self.stats["deleted"] = self.stats.get("deleted", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("deleted")
                        except Exception as del_error:
                            logger.error(
                                "individual_delete_failed_after_batch",
                                slug=action.card.slug,
                                error=str(del_error),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                return
            except Exception as e:
                logger.error(
                    "batch_delete_anki_failed_unexpected", error=str(e), exc_info=True
                )
                # Fall back to individual deletes
                for action in actions:
                    if action.anki_guid:
                        try:
                            self.delete_card(action.card, action.anki_guid)
                            self.stats["deleted"] = self.stats.get("deleted", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("deleted")
                        except Exception as del_error:
                            logger.error(
                                "individual_delete_failed_after_batch_fallback",
                                slug=action.card.slug,
                                error=str(del_error),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                return

        # Batch delete from database
        if slugs_to_delete:
            try:
                self.db.delete_cards_batch(slugs_to_delete)
                deleted_count = len(slugs_to_delete)
                self.stats["deleted"] = self.stats.get("deleted", 0) + deleted_count
                if self.progress:
                    for _ in range(deleted_count):
                        self.progress.increment_stat("deleted")
                logger.info("batch_delete_db", count=deleted_count)
            except Exception as e:
                logger.error("batch_delete_db_failed", error=str(e), exc_info=True)
                # Fall back to individual deletes
                for slug in slugs_to_delete:
                    try:
                        self.db.delete_card(slug)
                    except Exception as db_error:
                        logger.warning(
                            "individual_db_delete_failed_after_batch",
                            slug=slug,
                            error=str(db_error),
                        )

    def _verify_card_exists(
        self,
        card: Card,
        note_id: int,
        expected_fields: dict[str, str],
        expected_tags: list[str],
    ) -> None:
        """Verify card exists in Anki (for use inside transactions).

        This method is used inside CardTransaction to verify the card
        was created successfully BEFORE committing. Unlike verify_creation(),
        this method always raises an exception on failure to trigger rollback.

        Args:
            card: Card object that was created
            note_id: Anki note ID returned from creation
            expected_fields: Expected field values
            expected_tags: Expected tags

        Raises:
            CardOperationError: If card not found or verification fails
        """
        try:
            notes_info = self.anki.notes_info([note_id])
            if not notes_info or notes_info[0] is None:
                msg = (
                    f"Card {card.slug} (note_id={note_id}) not found in Anki "
                    f"immediately after creation"
                )
                logger.error(
                    "atomic_verification_failed_not_found",
                    slug=card.slug,
                    note_id=note_id,
                )
                raise CardOperationError(msg)

            note_info = notes_info[0]

            # Verify note ID matches
            if note_info.get("noteId") != note_id:
                msg = (
                    f"Card {card.slug} verification failed: note ID mismatch "
                    f"(expected {note_id}, got {note_info.get('noteId')})"
                )
                logger.error(
                    "atomic_verification_failed_id_mismatch",
                    slug=card.slug,
                    expected_note_id=note_id,
                    actual_note_id=note_info.get("noteId"),
                )
                raise CardOperationError(msg)

            # Log successful verification
            logger.debug(
                "atomic_verification_succeeded",
                slug=card.slug,
                note_id=note_id,
            )

        except CardOperationError:
            # Re-raise to trigger transaction rollback
            raise
        except AnkiConnectError as e:
            # Network/API error - fail safe by raising
            msg = f"Card {card.slug} verification failed due to Anki error: {e}"
            logger.error(
                "atomic_verification_failed_anki_error",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            raise CardOperationError(msg) from e
        except Exception as e:
            # Unexpected error - fail safe by raising
            msg = f"Card {card.slug} verification failed unexpectedly: {e}"
            logger.error(
                "atomic_verification_failed_unexpected",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            raise CardOperationError(msg) from e

    def _verify_update_applied(self, card: Card, note_id: int) -> None:
        """Verify update was applied in Anki (for use inside transactions).

        This method is used inside CardTransaction to verify the update
        was applied successfully BEFORE committing. Raises an exception
        on failure to trigger rollback.

        Args:
            card: Card object that was updated
            note_id: Anki note ID

        Raises:
            CardOperationError: If note not found or verification fails
        """
        try:
            notes_info = self.anki.notes_info([note_id])
            if not notes_info or notes_info[0] is None:
                msg = (
                    f"Card {card.slug} (note_id={note_id}) not found in Anki "
                    f"after update"
                )
                logger.error(
                    "atomic_update_verification_failed_not_found",
                    slug=card.slug,
                    note_id=note_id,
                )
                raise CardOperationError(msg)

            # Log successful verification
            logger.debug(
                "atomic_update_verification_succeeded",
                slug=card.slug,
                note_id=note_id,
            )

        except CardOperationError:
            # Re-raise to trigger transaction rollback
            raise
        except AnkiConnectError as e:
            # Network/API error - fail safe by raising
            msg = f"Card {card.slug} update verification failed due to Anki error: {e}"
            logger.error(
                "atomic_update_verification_failed_anki_error",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            raise CardOperationError(msg) from e
        except Exception as e:
            # Unexpected error - fail safe by raising
            msg = f"Card {card.slug} update verification failed unexpectedly: {e}"
            logger.error(
                "atomic_update_verification_failed_unexpected",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            raise CardOperationError(msg) from e

    def verify_creation(
        self,
        card: Card,
        note_id: int,
        expected_fields: dict[str, str],
        expected_tags: list[str],
    ) -> None:
        """Verify that a card was successfully created in Anki.

        Args:
            card: Card object that was created
            note_id: Anki note ID returned from creation
            expected_fields: Expected field values
            expected_tags: Expected tags

        Raises:
            CardOperationError: If verification fails critically
        """
        try:
            # Get note info from Anki
            notes_info = self.anki.notes_info([note_id])
            if not notes_info:
                logger.error(
                    "card_verification_failed_not_found",
                    slug=card.slug,
                    note_id=note_id,
                )
                msg = f"Card {card.slug} (note_id={note_id}) not found in Anki after creation"
                raise CardOperationError(msg)

            note_info = notes_info[0]

            # Verify note exists
            if note_info.get("noteId") != note_id:
                logger.error(
                    "card_verification_failed_id_mismatch",
                    slug=card.slug,
                    expected_note_id=note_id,
                    actual_note_id=note_info.get("noteId"),
                )
                msg = f"Card {card.slug} verification failed: note ID mismatch"
                raise CardOperationError(msg)

            # Verify deck
            actual_deck = note_info.get("deckName", "")
            expected_deck = self.config.anki_deck_name
            if actual_deck != expected_deck:
                logger.warning(
                    "card_verification_deck_mismatch",
                    slug=card.slug,
                    expected_deck=expected_deck,
                    actual_deck=actual_deck,
                )

            # Verify note type
            actual_note_type = note_info.get("modelName", "")
            expected_note_type = self._get_anki_model_name(card.note_type)
            if actual_note_type != expected_note_type:
                logger.warning(
                    "card_verification_note_type_mismatch",
                    slug=card.slug,
                    expected_note_type=expected_note_type,
                    actual_note_type=actual_note_type,
                )

            # Verify fields
            actual_fields = note_info.get("fields", {})
            field_mismatches = []
            for field_name, expected_value in expected_fields.items():
                actual_value = actual_fields.get(field_name, {}).get("value", "")
                # Normalize whitespace for comparison
                expected_normalized = " ".join(expected_value.split())
                actual_normalized = " ".join(actual_value.split())
                if expected_normalized != actual_normalized:
                    field_mismatches.append(field_name)
                    logger.debug(
                        "card_verification_field_mismatch",
                        slug=card.slug,
                        field=field_name,
                        expected_length=len(expected_value),
                        actual_length=len(actual_value),
                    )

            if field_mismatches:
                logger.warning(
                    "card_verification_field_mismatches",
                    slug=card.slug,
                    mismatched_fields=field_mismatches,
                )

            # Verify tags
            actual_tags = set(note_info.get("tags", []))
            expected_tags_set = set(expected_tags)
            if actual_tags != expected_tags_set:
                missing_tags = expected_tags_set - actual_tags
                extra_tags = actual_tags - expected_tags_set
                if missing_tags or extra_tags:
                    logger.warning(
                        "card_verification_tag_mismatch",
                        slug=card.slug,
                        missing_tags=list(missing_tags),
                        extra_tags=list(extra_tags),
                    )

            logger.debug(
                "card_verification_succeeded",
                slug=card.slug,
                note_id=note_id,
            )

        except AnkiConnectError as e:
            logger.error(
                "card_verification_failed_anki_error",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            # Don't raise - verification failure shouldn't break the sync

        except CardOperationError:
            # Re-raise critical verification failures
            raise

        except Exception as e:
            logger.error(
                "card_verification_failed_unexpected",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            # Don't raise - verification failure shouldn't break the sync

    def _get_anki_model_name(self, internal_note_type: str) -> str:
        """Get the actual Anki model name from internal note type.

        Args:
            internal_note_type: Internal note type identifier

        Returns:
            Actual Anki model name from config.model_names mapping
        """
        return self.config.model_names.get(internal_note_type, internal_note_type)
