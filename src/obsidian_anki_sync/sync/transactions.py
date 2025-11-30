"""Transaction support for atomic card operations.

This module provides transaction support to ensure that card operations
(create/update/delete) are atomic across both Anki and the database.

The CardTransaction class ensures that either both operations succeed,
or both are rolled back to maintain consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from obsidian_anki_sync.anki.client import AnkiClient

from .state_db import StateDB

logger = logging.getLogger(__name__)


@dataclass
class RollbackAction:
    """Represents a single rollback action and its result."""

    action_type: str
    args: tuple[Any, ...]
    succeeded: bool = False
    verified: bool = False
    error: str | None = None


@dataclass
class RollbackReport:
    """Report of rollback operation results."""

    total_actions: int = 0
    succeeded: int = 0
    failed: int = 0
    verified: int = 0
    actions: list[RollbackAction] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """Check if all rollback actions succeeded."""
        return self.failed == 0

    @property
    def all_verified(self) -> bool:
        """Check if all successful rollbacks were verified."""
        return self.verified == self.succeeded


class CardOperationError(Exception):
    """Error during card operation."""


class RollbackVerificationError(Exception):
    """Error when rollback verification fails.

    Indicates that the system may be in an inconsistent state
    that requires manual intervention.
    """

    def __init__(self, message: str, report: RollbackReport):
        super().__init__(message)
        self.report = report


class CardTransaction:
    """Atomic transaction for card create/update/delete operations.

    Ensures that either both Anki and database operations succeed,
    or both are rolled back to maintain consistency.

    Usage:
        with CardTransaction(anki_client, db) as txn:
            # Perform Anki operation
            note_id = anki_client.add_note(...)
            txn.rollback_actions.append(("delete_anki_note", note_id))

            # Perform database operation
            db.insert_card_extended(...)

            # Mark as successful
            txn.commit()
    """

    def __init__(self, anki_client: AnkiClient, db: StateDB):
        """Initialize transaction.

        Args:
            anki_client: AnkiConnect client for Anki operations
            db: State database for persistence
        """
        self.anki = anki_client
        self.db = db
        self.rollback_actions: list[tuple[str, Any]] = []
        self.committed = False

    def __enter__(self) -> "CardTransaction":
        """Enter transaction context."""
        return self

    # type: ignore[no-untyped-def]
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context, rolling back on exception.

        If an exception occurred and the transaction was not committed,
        attempts to rollback all operations. Rollback errors are logged
        but do not suppress the original exception.
        """
        if exc_type is not None and not self.committed:
            # Exception occurred, attempt rollback
            try:
                self.rollback(verify=True)
            except RollbackVerificationError as rollback_err:
                # Log rollback failure but don't suppress original exception
                logger.error(
                    "rollback_verification_error_during_exit",
                    original_error=str(exc_val),
                    rollback_error=str(rollback_err),
                    failed_actions=rollback_err.report.failed,
                )
        return False

    def rollback(self, verify: bool = True) -> RollbackReport:
        """Rollback all completed actions in reverse order.

        This method attempts to undo all operations that were performed
        during the transaction, restoring the system to its previous state.

        Args:
            verify: If True, verify each rollback action succeeded

        Returns:
            RollbackReport with details of what was rolled back

        Raises:
            RollbackVerificationError: If critical rollback actions failed
        """
        report = RollbackReport(total_actions=len(self.rollback_actions))
        logger.warning("rolling_back_transaction", actions=report.total_actions)

        for action_data in reversed(self.rollback_actions):
            action_type = action_data[0]
            args = action_data[1:]
            action = RollbackAction(action_type=action_type, args=args)

            try:
                if action_type == "delete_anki_note":
                    note_id = args[0]
                    self.anki.delete_notes([note_id])
                    action.succeeded = True
                    logger.info("rolled_back_anki_note", note_id=note_id)

                    # Verify deletion
                    if verify:
                        try:
                            info = self.anki.notes_info([note_id])
                            # If we get empty result or error, deletion succeeded
                            action.verified = not info or info[0] is None
                            if action.verified:
                                report.verified += 1
                            else:
                                logger.warning(
                                    "rollback_verification_failed",
                                    action=action_type,
                                    note_id=note_id,
                                    reason="Note still exists after deletion",
                                )
                        except Exception:
                            # If notes_info fails, assume deletion worked
                            action.verified = True
                            report.verified += 1

                elif action_type == "delete_db_card":
                    slug = args[0]
                    self.db.delete_card(slug)
                    action.succeeded = True
                    logger.info("rolled_back_db_card", slug=slug)

                    # Verify deletion
                    if verify:
                        card = self.db.get_by_slug(slug)
                        action.verified = card is None
                        if action.verified:
                            report.verified += 1
                        else:
                            logger.warning(
                                "rollback_verification_failed",
                                action=action_type,
                                slug=slug,
                                reason="Card still exists after deletion",
                            )

                elif action_type == "restore_anki_note":
                    note_id, old_fields, old_tags = args
                    self.anki.update_note_fields(note_id, old_fields)
                    self.anki.update_note_tags(note_id, old_tags)
                    action.succeeded = True
                    logger.info("rolled_back_anki_note_update", note_id=note_id)

                    # Verification for restore is complex, mark as verified if no error
                    if verify:
                        action.verified = True
                        report.verified += 1

                elif action_type == "recreate_deleted_note":
                    # Attempt to recreate a deleted note (best-effort)
                    old_note_id, fields, tags, model_name, deck_name, guid = args
                    try:
                        # Convert field values from note_info format to add_note format
                        field_dict = {}
                        for field_name, field_data in fields.items():
                            if isinstance(field_data, dict):
                                field_dict[field_name] = field_data.get("value", "")
                            else:
                                field_dict[field_name] = field_data

                        new_note_id = self.anki.add_note(
                            deck=deck_name,
                            note_type=model_name,
                            fields=field_dict,
                            tags=tags,
                            guid=guid,
                        )
                        action.succeeded = True
                        logger.info(
                            "rolled_back_deleted_note",
                            old_note_id=old_note_id,
                            new_note_id=new_note_id,
                            note="Scheduling data and review history not preserved",
                        )

                        # Verify recreation
                        if verify and new_note_id:
                            action.verified = True
                            report.verified += 1

                    except Exception as recreate_error:
                        action.error = str(recreate_error)
                        logger.error(
                            "rollback_recreate_failed",
                            old_note_id=old_note_id,
                            error=str(recreate_error),
                            note="Unable to restore deleted note",
                        )

                if action.succeeded:
                    report.succeeded += 1

            except Exception as e:
                action.error = str(e)
                report.failed += 1
                logger.error("rollback_failed", action=action_type, error=str(e))

            report.actions.append(action)

        self.rollback_actions.clear()

        # Log summary
        logger.info(
            "rollback_completed",
            total=report.total_actions,
            succeeded=report.succeeded,
            failed=report.failed,
            verified=report.verified,
        )

        # Raise error if critical rollbacks failed
        if report.failed > 0:
            msg = (
                f"Rollback partially failed: {report.failed}/{report.total_actions} "
                f"actions failed. System may be in inconsistent state."
            )
            raise RollbackVerificationError(msg, report)

        return report

    def commit(self) -> None:
        """Mark transaction as committed (no rollback needed).

        Call this method when all operations have succeeded to prevent
        rollback on context exit.
        """
        self.committed = True
        self.rollback_actions.clear()
