"""Repository for note index data operations."""

from datetime import datetime

from obsidian_anki_sync.infrastructure.database_connection_manager import (
    DatabaseConnectionManager,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class NoteIndexRepository:
    """Repository for note index data operations."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize note index repository.

        Args:
            connection_manager: Database connection manager
        """
        self._connection_manager = connection_manager

    def upsert_note_index(
        self,
        source_path: str,
        note_id: str | None,
        note_title: str | None,
        topic: str | None,
        language_tags: list[str],
        qa_pair_count: int,
        file_modified_at: datetime | None,
        metadata_json: str | None = None,
    ) -> None:
        """Insert or update a note in the index.

        Args:
            source_path: Relative path to note file
            note_id: Note ID from frontmatter
            note_title: Note title
            topic: Note topic
            language_tags: List of language codes
            qa_pair_count: Number of Q/A pairs in note
            file_modified_at: File modification timestamp
            metadata_json: JSON string of full metadata
        """
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO note_index (
                    source_path, note_id, note_title, topic, language_tags,
                    qa_pair_count, file_modified_at, metadata_json, last_indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_path) DO UPDATE SET
                    note_id = excluded.note_id,
                    note_title = excluded.note_title,
                    topic = excluded.topic,
                    language_tags = excluded.language_tags,
                    qa_pair_count = excluded.qa_pair_count,
                    file_modified_at = excluded.file_modified_at,
                    metadata_json = excluded.metadata_json,
                    last_indexed_at = CURRENT_TIMESTAMP
            """,
                (
                    source_path,
                    note_id,
                    note_title,
                    topic,
                    ",".join(language_tags) if language_tags else "",
                    qa_pair_count,
                    file_modified_at.isoformat() if file_modified_at else None,
                    metadata_json,
                ),
            )

    def update_note_sync_status(
        self, source_path: str, status: str, error_message: str | None = None
    ) -> None:
        """Update sync status for a note.

        Args:
            source_path: Relative path to note file
            status: Sync status (pending, processing, completed, failed)
            error_message: Optional error message if failed
        """
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE note_index
                SET sync_status = ?,
                    error_message = ?,
                    last_synced_at = CURRENT_TIMESTAMP
                WHERE source_path = ?
            """,
                (status, error_message, source_path),
            )

    def get_note_index(self, source_path: str) -> dict | None:
        """Get note index entry.

        Args:
            source_path: Relative path to note file

        Returns:
            Note index record or None
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM note_index WHERE source_path = ?",
            (source_path,),
            "get_note_index"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_notes_index(self) -> list[dict]:
        """Get all notes from index.

        Returns:
            List of note index records
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM note_index ORDER BY source_path",
            operation="get_all_notes_index"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_notes_by_status(self, status: str) -> list[dict]:
        """Get notes by sync status.

        Args:
            status: Sync status to filter by

        Returns:
            List of note index records
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM note_index WHERE sync_status = ? ORDER BY source_path",
            (status,),
            "get_notes_by_status"
        )
        return [dict(row) for row in cursor.fetchall()]

    def clear_index(self) -> None:
        """Clear all note index data."""
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM note_index")
