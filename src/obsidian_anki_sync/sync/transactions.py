"""Transaction support for atomic card operations.

This module provides transaction support to ensure that card operations
(create/update/delete) are atomic across both Anki and the database.

The CardTransaction class ensures that either both operations succeed,
or both are rolled back to maintain consistency.
"""

import logging
from typing import Any

from ..anki.client import AnkiClient
from .state_db import StateDB

logger = logging.getLogger(__name__)


class CardOperationError(Exception):
    """Error during card operation."""

    pass


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
        """Exit transaction context, rolling back on exception."""
        if exc_type is not None and not self.committed:
            # Exception occurred, rollback
            self.rollback()
        return False

    def rollback(self) -> None:
        """Rollback all completed actions in reverse order.

        This method attempts to undo all operations that were performed
        during the transaction, restoring the system to its previous state.
        """
        logger.warning("rolling_back_transaction",
                       actions=len(self.rollback_actions))

        for action_data in reversed(self.rollback_actions):
            action_type = action_data[0]
            args = action_data[1:]

            try:
                if action_type == "delete_anki_note":
                    note_id = args[0]
                    self.anki.delete_notes([note_id])
                    logger.info("rolled_back_anki_note", note_id=note_id)

                elif action_type == "delete_db_card":
                    slug = args[0]
                    self.db.delete_card(slug)
                    logger.info("rolled_back_db_card", slug=slug)

                elif action_type == "restore_anki_note":
                    note_id, old_fields, old_tags = args
                    self.anki.update_note_fields(note_id, old_fields)
                    self.anki.update_note_tags(note_id, old_tags)
                    logger.info("rolled_back_anki_note_update",
                                note_id=note_id)

                elif action_type == "recreate_deleted_note":
                    # Attempt to recreate a deleted note (best-effort, may not preserve all note state)
                    old_note_id, fields, tags, model_name, deck_name, guid = args
                    try:
                        # Convert field values from note_info format to add_note format
                        # note_info: {"FieldName": {"value": "content", "order": 0}}
                        # add_note: {"FieldName": "content"}
                        field_dict = {}
                        for field_name, field_data in fields.items():
                            if isinstance(field_data, dict):
                                field_dict[field_name] = field_data.get(
                                    "value", "")
                            else:
                                field_dict[field_name] = field_data

                        new_note_id = self.anki.add_note(
                            deck=deck_name,
                            note_type=model_name,
                            fields=field_dict,
                            tags=tags,
                            guid=guid,
                        )
                        logger.info(
                            "rolled_back_deleted_note",
                            old_note_id=old_note_id,
                            new_note_id=new_note_id,
                            note="Scheduling data and review history not preserved",
                        )
                    except Exception as recreate_error:
                        logger.error(
                            "rollback_recreate_failed",
                            old_note_id=old_note_id,
                            error=str(recreate_error),
                            note="Unable to restore deleted note",
                        )

            except Exception as e:
                logger.error("rollback_failed",
                             action=action_type, error=str(e))

        self.rollback_actions.clear()

    def commit(self) -> None:
        """Mark transaction as committed (no rollback needed).

        Call this method when all operations have succeeded to prevent
        rollback on context exit.
        """
        self.committed = True
        self.rollback_actions.clear()
