"""Safe and robust card importer with validation, rollback, and progress tracking.

This module provides a safer alternative to the basic importer with:
- Comprehensive input validation
- Transaction-like operations with rollback
- Batch processing for large imports
- Progress tracking and detailed error reporting
- Resource cleanup and memory management
"""

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.anki.validation import (
    ContentValidator,
    DataValidator,
    PathValidator,
    SafeFileOperations,
)
from obsidian_anki_sync.exceptions import AnkiError, DeckImportError
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ImportProgress:
    """Progress tracking for import operations."""

    def __init__(self, total_cards: int, progress_callback: Callable[[dict[str, Any]], None] | None = None):
        """Initialize progress tracker.

        Args:
            total_cards: Total number of cards to import
            progress_callback: Optional callback for progress updates
        """
        self.total_cards = total_cards
        self.processed = 0
        self.created = 0
        self.updated = 0
        self.errors = 0
        self.progress_callback = progress_callback

    def update(self, created: int = 0, updated: int = 0, errors: int = 0) -> None:
        """Update progress counters.

        Args:
            created: Number of cards created
            updated: Number of cards updated
            errors: Number of errors
        """
        self.created += created
        self.updated += updated
        self.errors += errors
        self.processed = self.created + self.updated + self.errors

        if self.progress_callback:
            progress_data = {
                'total': self.total_cards,
                'processed': self.processed,
                'created': self.created,
                'updated': self.updated,
                'errors': self.errors,
                'percentage': (self.processed / self.total_cards * 100) if self.total_cards > 0 else 0,
            }
            self.progress_callback(progress_data)

    def get_summary(self) -> dict[str, Any]:
        """Get progress summary.

        Returns:
            Dictionary with final counts
        """
        return {
            'total': self.total_cards,
            'processed': self.processed,
            'created': self.created,
            'updated': self.updated,
            'errors': self.errors,
        }


class ImportTransaction:
    """Transaction-like import operation with rollback capability."""

    def __init__(self, client: AnkiClient, deck_name: str):
        """Initialize import transaction.

        Args:
            client: AnkiClient instance
            deck_name: Target deck name
        """
        self.client = client
        self.deck_name = deck_name
        self.operations: list[dict[str, Any]] = []
        self.created_note_ids: list[int] = []
        self.updated_note_ids: list[tuple[int, dict[str, Any]]] = []  # (note_id, original_fields)

    def add_create_operation(self, note_id: int, fields: dict[str, Any], tags: list[str]) -> None:
        """Record a note creation operation.

        Args:
            note_id: ID of created note
            fields: Note fields
            tags: Note tags
        """
        self.operations.append({
            'type': 'create',
            'note_id': note_id,
            'fields': fields.copy(),
            'tags': tags.copy(),
        })
        self.created_note_ids.append(note_id)

    def add_update_operation(self, note_id: int, original_fields: dict[str, Any]) -> None:
        """Record a note update operation.

        Args:
            note_id: ID of updated note
            original_fields: Original field values for rollback
        """
        self.operations.append({
            'type': 'update',
            'note_id': note_id,
            'original_fields': original_fields.copy(),
        })
        self.updated_note_ids.append((note_id, original_fields))

    def rollback(self) -> dict[str, int]:
        """Rollback all operations.

        Returns:
            Dictionary with rollback counts: {'deleted': int, 'restored': int, 'errors': int}
        """
        logger.warning("rolling_back_import_operations", operation_count=len(self.operations))

        deleted = 0
        restored = 0
        errors = 0

        # Rollback in reverse order
        for operation in reversed(self.operations):
            try:
                if operation['type'] == 'create':
                    # Delete created notes
                    self.client.delete_notes([operation['note_id']])
                    deleted += 1
                    logger.debug("rolled_back_note_creation", note_id=operation['note_id'])

                elif operation['type'] == 'update':
                    # Restore original field values
                    self.client.update_note_fields(operation['note_id'], operation['original_fields'])
                    restored += 1
                    logger.debug("rolled_back_note_update", note_id=operation['note_id'])

            except AnkiError as e:
                errors += 1
                logger.error(
                    "rollback_operation_failed",
                    operation=operation,
                    error=str(e),
                )

        result = {'deleted': deleted, 'restored': restored, 'errors': errors}
        logger.info("rollback_complete", **result)
        return result


class SafeCardImporter:
    """Safe card importer with validation, rollback, and progress tracking."""

    def __init__(self, client: AnkiClient):
        """Initialize safe importer.

        Args:
            client: AnkiClient instance
        """
        self.client = client

    def import_from_yaml(
        self,
        input_path: str | Path,
        deck_name: str,
        note_type: str | None = None,
        key_field: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """Safely import cards from YAML file.

        Args:
            input_path: Path to YAML file
            deck_name: Target deck name
            note_type: Note type to use (auto-detected if not provided)
            key_field: Field to use for identifying existing notes
            progress_callback: Optional callback for progress updates
            batch_size: Number of cards to process in each batch

        Returns:
            Dictionary with operation counts: {'created': int, 'updated': int, 'errors': int}

        Raises:
            DeckImportError: If import fails
        """
        # Validate inputs
        input_path = PathValidator.validate_file_path(input_path)
        deck_name = ContentValidator.validate_deck_name(deck_name)
        if note_type:
            note_type = ContentValidator.validate_note_type(note_type)

        # Load and validate data
        yaml_data = SafeFileOperations.safe_load_yaml(input_path)
        cards_data = DataValidator.validate_yaml_data(yaml_data)

        # Import with transaction support
        return self._import_cards(
            cards_data=cards_data,
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )

    def import_from_csv(
        self,
        input_path: str | Path,
        deck_name: str,
        note_type: str | None = None,
        key_field: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """Safely import cards from CSV file.

        Args:
            input_path: Path to CSV file
            deck_name: Target deck name
            note_type: Note type to use (auto-detected if not provided)
            key_field: Field to use for identifying existing notes
            progress_callback: Optional callback for progress updates
            batch_size: Number of cards to process in each batch

        Returns:
            Dictionary with operation counts: {'created': int, 'updated': int, 'errors': int}

        Raises:
            DeckImportError: If import fails
        """
        # Validate inputs
        input_path = PathValidator.validate_file_path(input_path)
        deck_name = ContentValidator.validate_deck_name(deck_name)
        if note_type:
            note_type = ContentValidator.validate_note_type(note_type)

        # Load and validate data
        csv_rows = SafeFileOperations.safe_load_csv(input_path)
        cards_data = DataValidator.validate_csv_data(csv_rows)

        # Import with transaction support
        return self._import_cards(
            cards_data=cards_data,
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )

    def _import_cards(
        self,
        cards_data: list[dict[str, Any]],
        deck_name: str,
        note_type: str | None = None,
        key_field: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """Import cards with transaction support and batching.

        Args:
            cards_data: List of card dictionaries
            deck_name: Target deck name
            note_type: Note type to use
            key_field: Field for identifying existing notes
            progress_callback: Optional progress callback
            batch_size: Batch size for processing

        Returns:
            Dictionary with operation counts
        """
        progress = ImportProgress(len(cards_data), progress_callback)
        transaction = ImportTransaction(self.client, deck_name)

        # Auto-detect note type if not provided
        if not note_type:
            note_type = self._auto_detect_note_type(deck_name)

        # Auto-detect key field if not provided
        if not key_field:
            key_field = self._auto_detect_key_field(note_type)

        logger.info(
            "starting_safe_import",
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
            card_count=len(cards_data),
            batch_size=batch_size,
        )

        try:
            # Process cards in batches
            for i in range(0, len(cards_data), batch_size):
                batch = cards_data[i:i + batch_size]
                self._process_batch(
                    batch=batch,
                    deck_name=deck_name,
                    note_type=note_type,
                    key_field=key_field,
                    transaction=transaction,
                    progress=progress,
                )

            result = progress.get_summary()
            logger.info("safe_import_complete", **result)
            return result

        except Exception as e:
            # Rollback on failure
            logger.error("import_failed_rolling_back", error=str(e))
            rollback_result = transaction.rollback()

            # Re-raise with additional context
            msg = f"Import failed and was rolled back: {e}"
            context = {
                'deck_name': deck_name,
                'processed_cards': progress.processed,
                'rollback_result': rollback_result,
            }
            raise DeckImportError(msg, context=context) from e

    def _process_batch(
        self,
        batch: list[dict[str, Any]],
        deck_name: str,
        note_type: str,
        key_field: str,
        transaction: ImportTransaction,
        progress: ImportProgress,
    ) -> None:
        """Process a batch of cards.

        Args:
            batch: Batch of card data
            deck_name: Target deck name
            note_type: Note type
            key_field: Key field for identification
            transaction: Import transaction
            progress: Progress tracker
        """
        # Find existing notes in batch if using noteId
        existing_notes = {}
        if key_field == "noteId":
            note_ids = []
            for card in batch:
                if "noteId" in card:
                    with contextlib.suppress(ValueError, TypeError):
                        note_ids.append(int(card["noteId"]))

        if note_ids:
            try:
                notes_info = self.client.notes_info(note_ids)
                for note_info in notes_info:
                    note_id = note_info.get("noteId")
                    if note_id:
                        existing_notes[str(note_id)] = note_id
            except AnkiError:
                # Continue without existing notes info
                pass

        # Process each card in batch
        for card_data in batch:
            try:
                self._process_single_card(
                    card_data=card_data,
                    deck_name=deck_name,
                    note_type=note_type,
                    key_field=key_field,
                    existing_notes=existing_notes,
                    transaction=transaction,
                )
                progress.update(created=1 if 'created' in locals() else 0,
                              updated=1 if 'updated' in locals() else 0)

            except Exception as e:
                progress.update(errors=1)
                logger.error(
                    "card_import_error",
                    error=str(e),
                    slug=card_data.get("slug", "unknown"),
                    card_data=card_data,
                )

    def _process_single_card(
        self,
        card_data: dict[str, Any],
        deck_name: str,
        note_type: str,
        key_field: str,
        existing_notes: dict[str, int],
        transaction: ImportTransaction,
    ) -> None:
        """Process a single card.

        Args:
            card_data: Card data dictionary
            deck_name: Target deck name
            note_type: Note type
            key_field: Key field
            existing_notes: Dictionary of existing note IDs
            transaction: Import transaction

        Raises:
            AnkiError: If card processing fails
        """
        # Extract fields
        metadata_fields = {"noteId", "slug", "noteType", "tags", "manifest"}
        fields = {k: v for k, v in card_data.items() if k not in metadata_fields and v}

        tags = card_data.get("tags", [])
        if isinstance(tags, str):
            tags = tags.split()

        # Determine if this is an update or create
        existing_note_id = None
        is_update = False

        if key_field == "noteId":
            note_id_str = str(card_data.get("noteId", ""))
            if note_id_str and note_id_str in existing_notes:
                existing_note_id = existing_notes[note_id_str]
                is_update = True
        elif key_field in card_data:
            # Search for existing note by key field value
            key_value = str(card_data[key_field])
            query = f'deck:"{deck_name}" {key_field}:"{key_value}"'
            try:
                existing_ids = self.client.find_notes(query)
                if existing_ids:
                    existing_note_id = existing_ids[0]
                    is_update = True
            except AnkiError:
                # Continue as create operation
                pass

        if is_update and existing_note_id is not None:
            # Store original fields for rollback
            try:
                notes_info = self.client.notes_info([existing_note_id])
                if notes_info:
                    original_fields = notes_info[0].get("fields", {})
                    transaction.add_update_operation(existing_note_id, original_fields)

                    # Update existing note
                    self.client.update_note_fields(existing_note_id, fields)
                    if tags:
                        self.client.add_tags([existing_note_id], " ".join(tags))

            except AnkiError as e:
                msg = f"Failed to update note {existing_note_id}"
                raise AnkiError(msg) from e

        else:
            # Create new note
            try:
                new_note_id = self.client.add_note(
                    deck_name=deck_name,
                    model_name=note_type,
                    fields=fields,
                    tags=tags,
                )
                transaction.add_create_operation(new_note_id, fields, tags)

            except AnkiError as e:
                msg = f"Failed to create note for slug {card_data.get('slug', 'unknown')}"
                raise AnkiError(msg) from e

    def _auto_detect_note_type(self, deck_name: str) -> str:
        """Auto-detect note type from existing notes in deck.

        Args:
            deck_name: Deck name to check

        Returns:
            Detected note type name
        """
        try:
            note_ids = self.client.find_notes(f'deck:"{deck_name}"')
            if note_ids:
                notes_info = self.client.notes_info(note_ids[:1])
                if notes_info:
                    model_name = notes_info[0].get("modelName", "APF::Simple")
                    return str(model_name) if model_name else "APF::Simple"
        except AnkiError:
            pass

        return "APF::Simple"

    def _auto_detect_key_field(self, note_type: str) -> str:
        """Auto-detect key field for note type.

        Args:
            note_type: Note type name

        Returns:
            Key field name
        """
        try:
            field_names = self.client.get_model_field_names(note_type)
            if field_names:
                return field_names[0]
        except AnkiError:
            pass

        return "slug"


# End of SafeCardImporter class


@contextlib.contextmanager
def safe_import_context(client: AnkiClient, deck_name: str):
    """Context manager for safe import operations.

    Args:
        client: AnkiClient instance
        deck_name: Target deck name

    Yields:
        SafeCardImporter: Configured importer instance
    """
    importer = SafeCardImporter(client)
    try:
        yield importer
    except Exception as e:
        logger.error("safe_import_context_failed", deck_name=deck_name, error=str(e))
        raise
