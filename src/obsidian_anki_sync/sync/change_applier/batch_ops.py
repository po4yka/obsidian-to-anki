"""Batch operations for applying sync changes to Anki."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from obsidian_anki_sync.anki.field_mapper import (
    EmptyNoteError,
    map_apf_to_anki_fields,
    validate_anki_note_fields,
    validate_field_names_match_anki,
)
from obsidian_anki_sync.exceptions import FieldMappingError
from obsidian_anki_sync.sync.transactions import CardOperationError, CardTransaction
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.config import Config
    from obsidian_anki_sync.models import SyncAction

logger = get_logger(__name__)


class BatchChangeApplier:
    """Handles batch create/update/delete operations."""

    def __init__(
        self,
        config: Config,
        anki_client,
        state_db,
        progress_tracker=None,
        stats: dict[str, Any] | None = None,
        async_runner: AsyncioRunner | None = None,
    ):
        self.config = config
        self.anki = anki_client
        self.db = state_db
        self.progress = progress_tracker
        self.stats = stats or {}
        self._async_runner = async_runner or AsyncioRunner.get_global()

    # --------------------------------------------------------------------- #
    # Batch create
    # --------------------------------------------------------------------- #
    def create_batch(self, actions: list[SyncAction]) -> None:
        """Create multiple cards in batch."""
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
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            note_payloads: list[dict[str, Any]] = []
            card_data: list[tuple[SyncAction, dict[str, str], list[str]]] = []

            for action in batch_actions:
                card = action.card
                logger.debug(
                    "preparing_card_for_anki",
                    slug=card.slug,
                    note_type=card.note_type,
                    apf_html_length=len(card.apf_html) if card.apf_html else 0,
                    apf_html_preview=card.apf_html[:200] if card.apf_html else "empty",
                )

                try:
                    fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
                except FieldMappingError as e:
                    logger.error(
                        "batch_card_mapping_failed",
                        slug=card.slug,
                        error=str(e),
                        apf_html_length=len(card.apf_html) if card.apf_html else 0,
                    )
                    self._mark_error(card.slug, str(e))
                    continue

                try:
                    validate_anki_note_fields(fields, card.note_type, card.slug)
                except EmptyNoteError as e:
                    logger.error(
                        "batch_card_validation_failed_empty_fields",
                        slug=card.slug,
                        error=str(e),
                    )
                    self._mark_error(card.slug, str(e))
                    continue

                model_name = self._get_anki_model_name(card.note_type)
                try:
                    anki_field_names = self.anki.get_model_field_names(model_name)
                    validate_field_names_match_anki(
                        fields, model_name, anki_field_names, card.slug
                    )
                except FieldMappingError as e:
                    logger.error(
                        "batch_card_field_name_mismatch",
                        slug=card.slug,
                        model_name=model_name,
                        error=str(e),
                    )
                    self._mark_error(card.slug, str(e))
                    continue

                note_payload = {
                    "deckName": self.config.anki_deck_name,
                    "modelName": model_name,
                    "fields": fields,
                    "options": {"allowDuplicate": False, "duplicateScope": "deck"},
                    "tags": card.tags,
                }

                note_payloads.append(note_payload)
                card_data.append((action, fields, card.tags))

            if not note_payloads:
                continue

            note_ids = self._async_runner.run(
                _create_notes_async(note_payloads, self.config.max_concurrent_generations)
            )

            for action, fields, tags in card_data:
                card = action.card
                note_id = note_ids.pop(0) if note_ids else None
                if note_id is None:
                    logger.error("batch_note_create_failed", slug=card.slug)
                    self._mark_error(card.slug, "Failed to create note via AnkiConnect")
                    continue

                self.db.create_anki_mapping(card.slug, note_id)
                self.db.update_card_status(card.slug, "created", note_id)

                update_payload = {"id": note_id, "fields": fields}
                tag_payload = (note_id, tags)

                self._async_runner.run(
                    _update_cards_async(
                        [update_payload],
                        [tag_payload],
                        self.config.max_concurrent_generations,
                    )
                )

                if self.progress:
                    self.progress.increment_stat("created")
                self.stats["created"] = self.stats.get("created", 0) + 1

    # --------------------------------------------------------------------- #
    # Batch update
    # --------------------------------------------------------------------- #
    def update_batch(self, actions: list[SyncAction]) -> None:
        """Update multiple cards in batch."""
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("updating_cards_batch", total=total, batch_size=batch_size)

        async def _update_notes_async(
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
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            update_payloads: list[dict[str, Any]] = []
            tag_payloads: list[tuple[int, list[str]]] = []
            cards_for_validation: list[tuple[SyncAction, dict[str, str], list[str]]] = []

            for action in batch_actions:
                card = action.card
                note_id = action.anki_guid
                if note_id is None:
                    logger.warning(
                        "batch_update_missing_guid",
                        slug=card.slug,
                        action=action.type,
                    )
                    continue

                try:
                    fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
                    validate_anki_note_fields(fields, card.note_type, card.slug)
                except (FieldMappingError, EmptyNoteError) as e:
                    logger.error(
                        "batch_update_field_validation_failed",
                        slug=card.slug,
                        error=str(e),
                    )
                    self._mark_error(card.slug, str(e))
                    continue

                model_name = self._get_anki_model_name(card.note_type)
                try:
                    anki_field_names = self.anki.get_model_field_names(model_name)
                    validate_field_names_match_anki(
                        fields, model_name, anki_field_names, card.slug
                    )
                except FieldMappingError as e:
                    logger.error(
                        "batch_update_field_name_mismatch",
                        slug=card.slug,
                        model_name=model_name,
                        error=str(e),
                    )
                    self._mark_error(card.slug, str(e))
                    continue

                update_payloads.append({"id": note_id, "fields": fields})
                tag_payloads.append((note_id, card.tags))
                cards_for_validation.append((action, fields, card.tags))

            if not update_payloads:
                continue

            results = self._async_runner.run(
                _update_notes_async(
                    update_payloads,
                    tag_payloads,
                    self.config.max_concurrent_generations,
                )
            )

            for (action, fields, tags), (field_success, tag_success) in zip(
                cards_for_validation, results
            ):
                card = action.card
                note_id = action.anki_guid
                if note_id is None:
                    continue

                success = field_success and tag_success
                if not success:
                    self._mark_error(
                        card.slug, "Failed to update fields or tags for note"
                    )
                    continue

                try:
                    self._verify_update_applied(card, note_id)
                    self.db.update_card_status(card.slug, "updated", note_id)
                    self.db.record_last_reviewed(card.slug)
                    if self.progress:
                        self.progress.increment_stat("updated")
                    self.stats["updated"] = self.stats.get("updated", 0) + 1
                except Exception as e:  # pragma: no cover - verification lenient
                    logger.warning(
                        "batch_update_verification_failed",
                        slug=card.slug,
                        error=str(e),
                    )

    # --------------------------------------------------------------------- #
    # Batch delete
    # --------------------------------------------------------------------- #
    def delete_batch(self, actions: list[SyncAction]) -> None:
        """Delete multiple cards in batch."""
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("deleting_cards_batch", total=total, batch_size=batch_size)

        for batch_start in range(0, total, batch_size):
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            note_ids: list[int] = []
            slugs: list[str] = []

            for action in batch_actions:
                card = action.card
                note_id = action.anki_guid
                if note_id is None:
                    logger.warning(
                        "batch_delete_missing_guid",
                        slug=card.slug,
                        action=action.type,
                    )
                    continue

                note_ids.append(note_id)
                slugs.append(card.slug)

            if not note_ids:
                continue

            transaction = CardTransaction(self.anki)
            transaction.add_operation(
                lambda ids=note_ids: self.anki.delete_notes(ids),
                lambda restore_slugs=slugs: self.db.restore_cards(restore_slugs),
            )

            success = transaction.commit()

            if success:
                logger.info(
                    "batch_delete_success",
                    count=len(note_ids),
                    slugs=slugs,
                )
                for slug in slugs:
                    self.db.delete_card(slug)
                if self.progress:
                    self.progress.increment_stat_by("deleted", len(slugs))
                self.stats["deleted"] = self.stats.get("deleted", 0) + len(slugs)
            else:  # pragma: no cover - transactional rollback
                logger.error(
                    "batch_delete_failed",
                    note_ids=note_ids,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + len(slugs)
                if self.progress:
                    self.progress.increment_stat_by("errors", len(slugs))

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _mark_error(self, slug: str, error: str) -> None:
        self.stats["errors"] = self.stats.get("errors", 0) + 1
        if self.progress:
            self.progress.increment_stat("errors")
        self.db.update_card_status(slug, "failed", error)

    def _verify_update_applied(self, card, note_id: int) -> None:
        """Verify that updates were applied correctly."""
        try:
            note_info = self.anki.get_notes_info([note_id])[0]
            if not note_info:
                msg = f"Note not found after update: note_id={note_id}"
                raise CardOperationError(msg)

            model_name = note_info.get("modelName")
            expected_model_name = self._get_anki_model_name(card.note_type)
            if model_name != expected_model_name:
                logger.warning(
                    "update_verification_note_type_mismatch",
                    slug=card.slug,
                    expected_model_name=expected_model_name,
                    actual_model_name=model_name,
                )

            field_mismatches = []
            expected_fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
            for field_name, expected_value in expected_fields.items():
                actual_value = note_info.get("fields", {}).get(field_name, {}).get(
                    "value", ""
                )
                if " ".join(actual_value.split()) != " ".join(expected_value.split()):
                    field_mismatches.append(field_name)

            if field_mismatches:
                logger.warning(
                    "update_verification_field_mismatches",
                    slug=card.slug,
                    mismatched_fields=field_mismatches,
                )

            actual_tags = set(note_info.get("tags", []))
            expected_tags = set(card.tags)
            if actual_tags != expected_tags:
                logger.warning(
                    "update_verification_tag_mismatch",
                    slug=card.slug,
                    missing_tags=list(expected_tags - actual_tags),
                    extra_tags=list(actual_tags - expected_tags),
                )

        except Exception as e:  # pragma: no cover - lenient verification
            logger.error(
                "update_verification_failed",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )

    def _get_anki_model_name(self, internal_note_type: str) -> str:
        return self.config.model_names.get(internal_note_type, internal_note_type)


__all__ = ["BatchChangeApplier"]

