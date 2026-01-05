"""Service for Anki note operations."""

import re
from typing import Any, cast

from obsidian_anki_sync.anki.services.anki_cache import AnkiCache
from obsidian_anki_sync.anki.services.anki_deck_service import AnkiDeckService
from obsidian_anki_sync.anki.services.anki_http_client import AnkiHttpClient
from obsidian_anki_sync.anki.services.anki_model_service import AnkiModelService
from obsidian_anki_sync.domain.interfaces.anki_note_service import IAnkiNoteService
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiNoteService(IAnkiNoteService):
    """Service for Anki note operations.

    Handles all note-related operations including creation, updates,
    deletion, and validation, with support for both individual and
    batch operations.
    """

    def __init__(
        self,
        http_client: AnkiHttpClient,
        deck_service: AnkiDeckService,
        model_service: AnkiModelService,
        cache: AnkiCache,
    ):
        """
        Initialize note service.

        Args:
            http_client: HTTP client for AnkiConnect communication
            deck_service: Service for deck operations
            model_service: Service for model operations
            cache: Cache for metadata
        """
        self._http_client = http_client
        self._deck_service = deck_service
        self._model_service = model_service
        self._cache = cache
        logger.debug("anki_note_service_initialized")

    def find_notes(self, query: str) -> list[int]:
        """Find notes matching a query."""
        return cast(
            "list[int]", self._http_client.invoke("findNotes", {"query": query})
        )

    async def find_notes_async(self, query: str) -> list[int]:
        """Find notes matching a query (async)."""
        return cast(
            "list[int]",
            await self._http_client.invoke_async("findNotes", {"query": query}),
        )

    def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about notes."""
        if not note_ids:
            return []
        return cast(
            "list[dict[Any, Any]]",
            self._http_client.invoke("notesInfo", {"notes": note_ids}),
        )

    async def notes_info_async(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about notes (async)."""
        if not note_ids:
            return []
        return cast(
            "list[dict[Any, Any]]",
            await self._http_client.invoke_async("notesInfo", {"notes": note_ids}),
        )

    def _build_note_payload(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any] | None = None,
        guid: str | None = None,
    ) -> dict[str, Any]:
        """Build note payload for API calls."""
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

    def _validate_note_fields(
        self,
        model_name: str,
        fields: dict[str, str],
        context: str | None = None,
    ) -> None:
        """Validate note fields before sending to Anki."""
        # Get expected field names from Anki
        expected_fields = self._model_service.get_model_field_names(model_name)
        if not expected_fields or not isinstance(expected_fields, (list, tuple)):
            # Skip strict validation when model metadata is unavailable
            logger.warning("note_field_validation_skipped", model=model_name)
            return

        sent_fields = set(fields.keys())
        expected_set = set(expected_fields)

        # Check for field name mismatch
        missing = expected_set - sent_fields
        extra = sent_fields - expected_set

        if missing or extra:
            logger.error(
                "note_field_validation_failed",
                context=context,
                model_name=model_name,
                missing_fields=sorted(missing) if missing else None,
                extra_fields=sorted(extra) if extra else None,
                sent_fields=sorted(sent_fields),
                expected_fields=expected_fields,
            )
            msg = (
                f"Field name mismatch for model '{model_name}'. "
                f"Missing: {sorted(missing) if missing else 'none'}. "
                f"Extra: {sorted(extra) if extra else 'none'}. "
                f"Anki expects: {expected_fields}"
            )
            if context:
                msg = f"[{context}] {msg}"
            raise AnkiConnectError(msg)

        # Check that the first field (sort field) is not empty
        # Anki considers a note "empty" if the first field has no content
        first_field = expected_fields[0]
        first_value = fields.get(first_field, "")

        # Strip HTML tags and whitespace to check for actual content
        stripped = re.sub(r"<[^>]+>", "", first_value).strip()
        if not stripped or stripped in ("&nbsp;", " "):
            logger.error(
                "note_first_field_empty",
                context=context,
                model_name=model_name,
                first_field=first_field,
                first_value_preview=first_value[:100] if first_value else "EMPTY",
            )
            msg = (
                f"First field '{first_field}' is empty or contains only whitespace/HTML. "
                f"Anki will reject this note as empty."
            )
            if context:
                msg = f"[{context}] {msg}"
            raise AnkiConnectError(msg)

        logger.debug(
            "note_field_validation_passed",
            context=context,
            model_name=model_name,
            field_count=len(fields),
            first_field=first_field,
        )

    async def _validate_note_fields_async(
        self,
        model_name: str,
        fields: dict[str, str],
        context: str | None = None,
    ) -> None:
        """Async version of _validate_note_fields."""
        expected_fields = await self._model_service.get_model_field_names_async(
            model_name
        )
        if not expected_fields:
            msg = f"No fields found for model '{model_name}'"
            if context:
                msg = f"[{context}] {msg}"
            raise AnkiConnectError(msg)

        sent_fields = set(fields.keys())
        expected_set = set(expected_fields)

        missing = expected_set - sent_fields
        extra = sent_fields - expected_set

        if missing or extra:
            logger.error(
                "note_field_validation_failed",
                context=context,
                model_name=model_name,
                missing_fields=sorted(missing) if missing else None,
                extra_fields=sorted(extra) if extra else None,
                sent_fields=sorted(sent_fields),
                expected_fields=expected_fields,
            )
            msg = (
                f"Field name mismatch for model '{model_name}'. "
                f"Missing: {sorted(missing) if missing else 'none'}. "
                f"Extra: {sorted(extra) if extra else 'none'}. "
                f"Anki expects: {expected_fields}"
            )
            if context:
                msg = f"[{context}] {msg}"
            raise AnkiConnectError(msg)

        first_field = expected_fields[0]
        first_value = fields.get(first_field, "")
        stripped = re.sub(r"<[^>]+>", "", first_value).strip()

        if not stripped or stripped in ("&nbsp;", " "):
            logger.error(
                "note_first_field_empty",
                context=context,
                model_name=model_name,
                first_field=first_field,
                first_value_preview=first_value[:100] if first_value else "EMPTY",
            )
            msg = (
                f"First field '{first_field}' is empty or contains only whitespace/HTML. "
                f"Anki will reject this note as empty."
            )
            if context:
                msg = f"[{context}] {msg}"
            raise AnkiConnectError(msg)

        logger.debug(
            "note_field_validation_passed",
            context=context,
            model_name=model_name,
            field_count=len(fields),
            first_field=first_field,
        )

    def add_note(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any] | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note to Anki."""
        # Validate deck existence
        known_decks = self._deck_service.get_deck_names()
        if isinstance(known_decks, (list, set, tuple)) and known_decks:
            if deck_name not in known_decks:
                known_decks = self._deck_service.get_deck_names(use_cache=False)
                if (
                    isinstance(known_decks, (list, set, tuple))
                    and deck_name not in known_decks
                ):
                    msg = f"Deck '{deck_name}' not found in Anki."
                    raise AnkiConnectError(msg)

        # Validate model existence
        known_models = self._model_service.get_model_names()
        if isinstance(known_models, (list, set, tuple)) and known_models:
            if model_name not in known_models:
                known_models = self._model_service.get_model_names(use_cache=False)
                if (
                    isinstance(known_models, (list, set, tuple))
                    and model_name not in known_models
                ):
                    msg = f"Note type '{model_name}' not found in Anki."
                    raise AnkiConnectError(msg)

        # Validate field names and content before sending to Anki
        self._validate_note_fields(model_name, fields, context=guid)

        note_payload = self._build_note_payload(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

        result = cast(
            "int", self._http_client.invoke("addNote", {"note": note_payload})
        )

        logger.info("note_added", note_id=result, deck=deck_name, note_type=model_name)
        return result

    async def add_note_async(
        self,
        deck_name: str,
        model_name: str,
        fields: dict[str, str],
        tags: list[str | None] | None = None,
        options: dict[str, Any] | None = None,
        guid: str | None = None,
    ) -> int:
        """Add a new note to Anki (async)."""
        # Validate deck existence
        known_decks = await self._deck_service.get_deck_names_async()
        if deck_name not in known_decks:
            known_decks = await self._deck_service.get_deck_names_async(use_cache=False)
            if deck_name not in known_decks:
                msg = f"Deck '{deck_name}' not found in Anki."
                raise AnkiConnectError(msg)

        # Validate model existence
        known_models = await self._model_service.get_model_names_async()
        if model_name not in known_models:
            known_models = await self._model_service.get_model_names_async(
                use_cache=False
            )
            if model_name not in known_models:
                msg = f"Note type '{model_name}' not found in Anki."
                raise AnkiConnectError(msg)

        # Validate field names and content before sending to Anki
        await self._validate_note_fields_async(model_name, fields, context=guid)

        note_payload = self._build_note_payload(
            deck_name=deck_name,
            model_name=model_name,
            fields=fields,
            tags=tags,
            options=options,
            guid=guid,
        )

        result = cast(
            "int",
            await self._http_client.invoke_async("addNote", {"note": note_payload}),
        )
        logger.info("note_added", note_id=result, deck=deck_name, note_type=model_name)
        return result

    def add_notes(self, notes: list[dict[str, Any]]) -> list[int | None]:
        """Add multiple notes in a single batch operation."""
        if not notes:
            return []

        # Validate existence of all decks and models in the batch
        required_decks = {n["deckName"] for n in notes if n.get("deckName")}
        required_models = {n["modelName"] for n in notes if n.get("modelName")}

        known_decks = set(self._deck_service.get_deck_names())
        missing_decks = required_decks - known_decks
        if missing_decks:
            known_decks = set(self._deck_service.get_deck_names(use_cache=False))
            missing_decks = required_decks - known_decks
            if missing_decks:
                msg = f"Decks not found: {', '.join(sorted(missing_decks))}"
                raise AnkiConnectError(msg)

        known_models = set(self._model_service.get_model_names())
        missing_models = required_models - known_models
        if missing_models:
            known_models = set(self._model_service.get_model_names(use_cache=False))
            missing_models = required_models - known_models
            if missing_models:
                msg = f"Note types not found: {', '.join(sorted(missing_models))}"
                raise AnkiConnectError(msg)

        # Validate field names and content for each note before sending to Anki
        for i, note in enumerate(notes):
            model_name = note.get("modelName", "")
            fields = note.get("fields", {})
            context = note.get("guid") or f"batch_note_{i}"
            self._validate_note_fields(model_name, fields, context=context)

        result = cast(
            "list[int | None]", self._http_client.invoke("addNotes", {"notes": notes})
        )

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
        """Add multiple notes in a single batch operation (async)."""
        if not notes:
            return []

        # Validate existence of all decks and models in the batch
        required_decks = {n["deckName"] for n in notes if n.get("deckName")}
        required_models = {n["modelName"] for n in notes if n.get("modelName")}

        known_decks = set(await self._deck_service.get_deck_names_async())
        missing_decks = required_decks - known_decks
        if missing_decks:
            known_decks = set(
                await self._deck_service.get_deck_names_async(use_cache=False)
            )
            missing_decks = required_decks - known_decks
            if missing_decks:
                msg = f"Decks not found: {', '.join(sorted(missing_decks))}"
                raise AnkiConnectError(msg)

        known_models = set(await self._model_service.get_model_names_async())
        missing_models = required_models - known_models
        if missing_models:
            known_models = set(
                await self._model_service.get_model_names_async(use_cache=False)
            )
            missing_models = required_models - known_models
            if missing_models:
                msg = f"Note types not found: {', '.join(sorted(missing_models))}"
                raise AnkiConnectError(msg)

        # Validate field names and content for each note before sending to Anki
        for i, note in enumerate(notes):
            model_name = note.get("modelName", "")
            fields = note.get("fields", {})
            context = note.get("guid") or f"batch_note_{i}"
            await self._validate_note_fields_async(model_name, fields, context=context)

        result = cast(
            "list[int | None]",
            await self._http_client.invoke_async("addNotes", {"notes": notes}),
        )

        return result

    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Update fields of an existing note."""
        self._http_client.invoke(
            "updateNoteFields", {"note": {"id": note_id, "fields": fields}}
        )
        logger.info("note_updated", note_id=note_id)

    async def update_note_fields_async(
        self, note_id: int, fields: dict[str, str]
    ) -> None:
        """Update fields of an existing note (async)."""
        await self._http_client.invoke_async(
            "updateNoteFields", {"note": {"id": note_id, "fields": fields}}
        )

    def update_notes_fields(self, updates: list[dict[str, Any]]) -> list[bool]:
        """Update multiple notes' fields in a single batch operation using multi action."""
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
                self._http_client.invoke("multi", {"actions": actions}),
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
                    self._http_client.invoke("updateNoteFields", {"note": update})
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
        """Update multiple notes' fields in a single batch operation (async)."""
        if not updates:
            return []

        actions = [
            {"action": "updateNoteFields", "params": {"note": update}}
            for update in updates
        ]

        try:
            results = cast(
                "list[dict[str, Any]]",
                await self._http_client.invoke_async("multi", {"actions": actions}),
            )

            # Extract success status and log failures
            success_list = []
            for i, result in enumerate(results):
                if result.get("error") is None:
                    success_list.append(True)
                else:
                    logger.warning(
                        "batch_update_async_failed",
                        note_id=updates[i].get("id"),
                        error=result.get("error"),
                    )
                    success_list.append(False)

            successful = sum(1 for r in success_list if r)
            logger.info(
                "notes_updated_batch_async",
                total=len(updates),
                successful=successful,
                failed=len(updates) - successful,
            )
            return success_list

        except Exception as e:
            logger.error(
                "batch_update_async_multi_failed",
                error=str(e),
                error_type=type(e).__name__,
                total=len(updates),
            )
            return [False] * len(updates)

    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes from Anki."""
        if not note_ids:
            return

        self._http_client.invoke("deleteNotes", {"notes": note_ids})
        logger.info("notes_deleted", count=len(note_ids))

    async def delete_notes_async(self, note_ids: list[int]) -> None:
        """Delete notes from Anki (async)."""
        if not note_ids:
            return
        await self._http_client.invoke_async("deleteNotes", {"notes": note_ids})

    def can_add_notes(self, notes: list[dict[str, Any]]) -> list[bool]:
        """Check if notes can be added (duplicate prevention)."""
        if not notes:
            return []

        return cast(
            "list[bool]", self._http_client.invoke("canAddNotes", {"notes": notes})
        )
