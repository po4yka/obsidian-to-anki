"""Change application component for SyncEngine.

Handles applying sync actions to Anki (create, update, delete operations).
"""

from typing import TYPE_CHECKING, Any

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.anki.field_mapper import (
    EmptyNoteError,
    map_apf_to_anki_fields,
    validate_anki_note_fields,
    validate_field_names_match_anki,
)
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import AnkiConnectError, FieldMappingError
from obsidian_anki_sync.models import Card, SyncAction
from obsidian_anki_sync.sync.change_applier.batch_ops import BatchChangeApplier
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
        fail_fast: bool | None = None,
    ):
        """Initialize change applier.

        Args:
            config: Service configuration
            state_db: State database
            anki_client: AnkiConnect client
            progress_tracker: Optional progress tracker
            stats: Statistics dictionary to update
            async_runner: Optional async runner for sync/async bridging
            fail_fast: If True, re-raise errors on first card operation failure.
                If None, uses config.fail_on_card_error.
                If False, continue processing remaining cards on error.
        """
        self.config = config
        self.db = state_db
        self.anki = anki_client
        self.progress = progress_tracker
        self.stats = stats or {}
        self._async_runner = async_runner or AsyncioRunner.get_global()
        # Determine fail-fast behavior from parameter or config
        self.fail_fast = (
            fail_fast if fail_fast is not None else config.fail_on_card_error
        )
        self.batch_applier = BatchChangeApplier(
            config=config,
            anki_client=anki_client,
            state_db=state_db,
            progress_tracker=progress_tracker,
            stats=self.stats,
            async_runner=self._async_runner,
        )

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
                    fail_fast=self.fail_fast,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")
                if self.fail_fast:
                    raise  # Re-raise to halt sync immediately
            except AnkiConnectError as e:
                logger.error(
                    "action_failed_anki_connect",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                    fail_fast=self.fail_fast,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")
                if self.fail_fast:
                    raise  # Re-raise to halt sync immediately
            except Exception as e:
                logger.error(
                    "action_failed_unexpected",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                    exc_info=True,
                    fail_fast=self.fail_fast,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                if self.progress:
                    self.progress.increment_stat("errors")
                if self.fail_fast:
                    raise  # Re-raise to halt sync immediately

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

        Raises:
            EmptyNoteError: If required fields are empty
            CardOperationError: If card creation fails
            AnkiConnectError: If AnkiConnect communication fails
        """
        logger.info("creating_card", slug=card.slug)

        # Map fields - this may raise FieldMappingError for invalid APF content
        try:
            fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
        except FieldMappingError as e:
            logger.error(
                "card_mapping_failed",
                slug=card.slug,
                error=str(e),
                apf_html_length=len(card.apf_html) if card.apf_html else 0,
            )
            error_msg = f"Failed to map APF content for card {card.slug}: {e}"
            self.db.update_card_status(card.slug, "failed", str(e))
            raise CardOperationError(error_msg) from e

        # Validate fields BEFORE sending to AnkiConnect
        # This prevents "cannot create note because it is empty" errors
        try:
            validate_anki_note_fields(fields, card.note_type, card.slug)
        except EmptyNoteError as e:
            logger.error(
                "card_validation_failed_empty_fields",
                slug=card.slug,
                error=str(e),
            )
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

        # Get the actual Anki model name and validate field names match
        model_name = self._get_anki_model_name(card.note_type)
        try:
            anki_field_names = self.anki.get_model_field_names(model_name)
            validate_field_names_match_anki(fields, model_name, anki_field_names, card.slug)
        except FieldMappingError as e:
            logger.error(
                "card_field_name_mismatch",
                slug=card.slug,
                model_name=model_name,
                error=str(e),
            )
            self.db.update_card_status(card.slug, "failed", str(e))
            raise CardOperationError(str(e)) from e

        # Log fields being sent to Anki for diagnostics
        logger.debug(
            "sending_card_to_anki",
            slug=card.slug,
            model_name=model_name,
            field_names=list(fields.keys()),
            field_values_preview={
                k: (v[:100] + "..." if len(v) > 100 else v) if v else "EMPTY"
                for k, v in fields.items()
            },
        )

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

        Raises:
            EmptyNoteError: If required fields are empty
            CardOperationError: If card update fails
            AnkiConnectError: If AnkiConnect communication fails
        """
        logger.info("updating_card", slug=card.slug, anki_guid=anki_guid)

        # Map fields - this may raise FieldMappingError for invalid APF content
        try:
            fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
        except FieldMappingError as e:
            logger.error(
                "card_update_mapping_failed",
                slug=card.slug,
                anki_guid=anki_guid,
                error=str(e),
                apf_html_length=len(card.apf_html) if card.apf_html else 0,
            )
            error_msg = f"Failed to map APF content for card {card.slug}: {e}"
            self.db.update_card_status(card.slug, "failed", str(e))
            raise CardOperationError(error_msg) from e

        # Validate fields BEFORE sending to AnkiConnect
        # This prevents "cannot create note because it is empty" errors
        try:
            validate_anki_note_fields(fields, card.note_type, card.slug)
        except EmptyNoteError as e:
            logger.error(
                "card_update_validation_failed_empty_fields",
                slug=card.slug,
                anki_guid=anki_guid,
                error=str(e),
            )
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

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
        """Create multiple cards in batch via BatchChangeApplier."""
        self.batch_applier.create_batch(actions)

    def update_batch(self, actions: list[SyncAction]) -> None:
        """Update multiple cards in batch via BatchChangeApplier."""
        self.batch_applier.update_batch(actions)

    def delete_batch(self, actions: list[SyncAction]) -> None:
        """Delete multiple cards in batch via BatchChangeApplier."""
        self.batch_applier.delete_batch(actions)

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
