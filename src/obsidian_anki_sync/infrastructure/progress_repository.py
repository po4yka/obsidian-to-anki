"""Repository for sync progress data operations."""

import json
from datetime import datetime
from typing import Any

from obsidian_anki_sync.infrastructure.database_connection_manager import (
    DatabaseConnectionManager,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ProgressRepository:
    """Repository for sync progress data operations."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize progress repository.

        Args:
            connection_manager: Database connection manager
        """
        self._connection_manager = connection_manager

    def save_progress(self, progress: "SyncProgress") -> None:
        """Save sync progress state.

        Args:
            progress: SyncProgress instance
        """
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()

            # Serialize note progress
            note_progress_json = json.dumps(
                {
                    key: {
                        "source_path": note.source_path,
                        "card_index": note.card_index,
                        "lang": note.lang,
                        "status": note.status,
                        "error": note.error,
                        "started_at": (
                            note.started_at.isoformat() if note.started_at else None
                        ),
                        "completed_at": (
                            note.completed_at.isoformat() if note.completed_at else None
                        ),
                    }
                    for key, note in progress.note_progress.items()
                }
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO sync_progress (
                    session_id, phase, started_at, updated_at, completed_at,
                    total_notes, notes_processed, cards_generated,
                    cards_created, cards_updated, cards_deleted,
                    cards_restored, cards_skipped, errors, note_progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    progress.session_id,
                    progress.phase.value,
                    progress.started_at.isoformat(),
                    progress.updated_at.isoformat(),
                    progress.completed_at.isoformat()
                    if progress.completed_at
                    else None,
                    progress.total_notes,
                    progress.notes_processed,
                    progress.cards_generated,
                    progress.cards_created,
                    progress.cards_updated,
                    progress.cards_deleted,
                    progress.cards_restored,
                    progress.cards_skipped,
                    progress.errors,
                    note_progress_json,
                ),
            )

    def get_progress(self, session_id: str) -> "SyncProgress | None":
        """Get sync progress by session ID.

        Args:
            session_id: Session identifier

        Returns:
            SyncProgress instance or None
        """
        from obsidian_anki_sync.sync.progress import (
            NoteProgress,
            SyncPhase,
            SyncProgress,
        )

        cursor = self._connection_manager.execute_query(
            "SELECT * FROM sync_progress WHERE session_id = ?",
            (session_id,),
            "get_progress"
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize note progress
        note_progress = {}
        if row["note_progress"]:
            note_progress_data = json.loads(row["note_progress"])
            for key, data in note_progress_data.items():
                note_progress[key] = NoteProgress(
                    source_path=data["source_path"],
                    card_index=data["card_index"],
                    lang=data["lang"],
                    status=data["status"],
                    error=data.get("error"),
                    started_at=(
                        datetime.fromisoformat(data["started_at"])
                        if data.get("started_at")
                        else None
                    ),
                    completed_at=(
                        datetime.fromisoformat(data["completed_at"])
                        if data.get("completed_at")
                        else None
                    ),
                )

        return SyncProgress(
            session_id=row["session_id"],
            phase=SyncPhase(row["phase"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            total_notes=row["total_notes"],
            notes_processed=row["notes_processed"],
            cards_generated=row["cards_generated"],
            cards_created=row["cards_created"],
            cards_updated=row["cards_updated"],
            cards_deleted=row["cards_deleted"],
            cards_restored=row["cards_restored"],
            cards_skipped=row["cards_skipped"],
            errors=row["errors"],
            note_progress=note_progress,
        )

    def get_all_progress(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent sync progress records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of progress record dictionaries
        """
        cursor = self._connection_manager.execute_query(
            """
            SELECT session_id, phase, started_at, updated_at, completed_at,
                   total_notes, notes_processed, errors
            FROM sync_progress
            ORDER BY updated_at DESC
            LIMIT ?
        """,
            (limit,),
            "get_all_progress"
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "session_id": row["session_id"],
                    "phase": row["phase"],
                    "started_at": row["started_at"],
                    "updated_at": row["updated_at"],
                    "completed_at": row["completed_at"],
                    "total_notes": row["total_notes"],
                    "notes_processed": row["notes_processed"],
                    "errors": row["errors"],
                }
            )

        return results

    def get_incomplete_progress(self) -> list[dict[str, Any]]:
        """Get all incomplete sync sessions.

        Returns:
            List of progress records with incomplete syncs
        """
        cursor = self._connection_manager.execute_query(
            """
            SELECT session_id, phase, started_at, updated_at,
                   total_notes, notes_processed, errors
            FROM sync_progress
            WHERE phase IN ('initializing', 'indexing', 'scanning', 'generating', 'determining_actions', 'applying_changes', 'interrupted')
            ORDER BY updated_at DESC
        """,
            operation="get_incomplete_progress"
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "session_id": row["session_id"],
                    "phase": row["phase"],
                    "started_at": row["started_at"],
                    "updated_at": row["updated_at"],
                    "total_notes": row["total_notes"],
                    "notes_processed": row["notes_processed"],
                    "errors": row["errors"],
                }
            )

        return results

    def delete_progress(self, session_id: str) -> None:
        """Delete a progress record.

        Args:
            session_id: Session identifier
        """
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM sync_progress WHERE session_id = ?", (session_id,)
            )
