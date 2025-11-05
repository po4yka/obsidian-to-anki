"""SQLite database for tracking sync state.

Security Note:
    All SQL queries in this module use parameterized statements (? placeholders)
    to prevent SQL injection vulnerabilities. Never use string formatting or
    concatenation to build SQL queries. Always use parameter binding.

    Example of correct usage:
        cursor.execute("SELECT * FROM cards WHERE slug = ?", (slug,))

    Example of INCORRECT usage (vulnerable to SQL injection):
        cursor.execute(f"SELECT * FROM cards WHERE slug = '{slug}'")
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from ..models import Card


class StateDB:
    """SQLite database for tracking card state."""

    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cards (
                slug TEXT PRIMARY KEY,
                slug_base TEXT NOT NULL,
                lang TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_anchor TEXT NOT NULL,
                card_index INTEGER NOT NULL,
                anki_guid INTEGER UNIQUE,
                content_hash TEXT NOT NULL,
                note_id TEXT,
                note_title TEXT,
                note_type TEXT DEFAULT 'APF::Simple',
                card_guid TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_path, card_index, lang)
            )
        """
        )
        # Ensure card_guid column exists for legacy databases
        cursor.execute(
            """
            PRAGMA table_info(cards)
        """
        )
        columns = {row[1] for row in cursor.fetchall()}
        if "card_guid" not in columns:
            cursor.execute("ALTER TABLE cards ADD COLUMN card_guid TEXT")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source ON cards(source_path)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_guid ON cards(anki_guid)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_card_guid ON cards(card_guid)
        """
        )

        # Progress tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_progress (
                session_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                total_notes INTEGER DEFAULT 0,
                notes_processed INTEGER DEFAULT 0,
                cards_generated INTEGER DEFAULT 0,
                cards_created INTEGER DEFAULT 0,
                cards_updated INTEGER DEFAULT 0,
                cards_deleted INTEGER DEFAULT 0,
                cards_restored INTEGER DEFAULT 0,
                cards_skipped INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                note_progress TEXT  -- JSON blob of note progress
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_progress_phase ON sync_progress(phase)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_progress_updated ON sync_progress(updated_at)
        """
        )

        # Note index table - catalog of all Obsidian notes
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS note_index (
                source_path TEXT PRIMARY KEY,
                note_id TEXT,
                note_title TEXT,
                topic TEXT,
                language_tags TEXT,
                qa_pair_count INTEGER DEFAULT 0,
                file_modified_at TIMESTAMP,
                last_indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_synced_at TIMESTAMP,
                sync_status TEXT DEFAULT 'pending',
                error_message TEXT,
                metadata_json TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_note_status ON note_index(sync_status)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_note_id ON note_index(note_id)
        """
        )

        # Card index table - catalog of expected and existing cards
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS card_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                card_index INTEGER NOT NULL,
                lang TEXT NOT NULL,
                slug TEXT UNIQUE,
                anki_guid INTEGER,
                note_id TEXT,
                note_title TEXT,
                content_hash TEXT,
                status TEXT DEFAULT 'expected',
                last_indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                in_obsidian BOOLEAN DEFAULT 1,
                in_anki BOOLEAN DEFAULT 0,
                in_database BOOLEAN DEFAULT 0,
                UNIQUE(source_path, card_index, lang)
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_card_source ON card_index(source_path)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_card_status ON card_index(status)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_card_slug ON card_index(slug)
        """
        )

        self.conn.commit()

    def insert_card(self, card: Card, anki_guid: int) -> None:
        """Insert a new card record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cards (
                slug, slug_base, lang, source_path, source_anchor,
                card_index, anki_guid, content_hash, note_id, note_title, note_type,
                card_guid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                card.slug,
                card.manifest.slug_base,
                card.lang,
                card.manifest.source_path,
                card.manifest.source_anchor,
                card.manifest.card_index,
                anki_guid,
                card.content_hash,
                card.manifest.note_id,
                card.manifest.note_title,
                card.note_type,
                card.guid,
            ),
        )
        self.conn.commit()

    def update_card(self, card: Card) -> None:
        """Update existing card record."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE cards
            SET content_hash = ?,
                note_title = ?,
                note_type = ?,
                card_guid = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE slug = ?
        """,
            (
                card.content_hash,
                card.manifest.note_title,
                card.note_type,
                card.guid,
                card.slug,
            ),
        )
        self.conn.commit()

    def get_by_slug(self, slug: str) -> dict | None:
        """Get card by slug."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_guid(self, anki_guid: int) -> dict | None:
        """Get card by Anki GUID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE anki_guid = ?", (anki_guid,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_source(self, source_path: str) -> list[dict]:
        """Get all cards from a source note."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE source_path = ?", (source_path,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_cards(self) -> list[dict]:
        """Get all cards."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards")
        return [dict(row) for row in cursor.fetchall()]

    def get_processed_note_paths(self) -> set[str]:
        """Get set of all note paths that have been processed.

        Returns:
            Set of source_path values from cards that have been synced
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT source_path FROM cards")
        return {row["source_path"] for row in cursor.fetchall()}

    def delete_card(self, slug: str) -> None:
        """Delete a card record."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM cards WHERE slug = ?", (slug,))
        self.conn.commit()

    def save_progress(self, progress) -> None:
        """Save sync progress state.

        Args:
            progress: SyncProgress instance
        """
        cursor = self.conn.cursor()

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
                progress.completed_at.isoformat() if progress.completed_at else None,
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
        self.conn.commit()

    def get_progress(self, session_id: str):
        """Get sync progress by session ID.

        Args:
            session_id: Session identifier

        Returns:
            SyncProgress instance or None
        """
        from .progress import NoteProgress, SyncPhase, SyncProgress

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sync_progress WHERE session_id = ?", (session_id,)
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

    def get_all_progress(self, limit: int = 10):
        """Get recent sync progress records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of SyncProgress instances
        """

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT session_id, phase, started_at, updated_at, completed_at,
                   total_notes, notes_processed, errors
            FROM sync_progress
            ORDER BY updated_at DESC
            LIMIT ?
        """,
            (limit,),
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

    def get_incomplete_progress(self):
        """Get all incomplete sync sessions.

        Returns:
            List of progress records with incomplete syncs
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT session_id, phase, started_at, updated_at,
                   total_notes, notes_processed, errors
            FROM sync_progress
            WHERE phase IN ('scanning', 'generating', 'determining_actions', 'applying_changes', 'interrupted')
            ORDER BY updated_at DESC
        """
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
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM sync_progress WHERE session_id = ?", (session_id,))
        self.conn.commit()

    # Note Index Methods

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
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def update_note_sync_status(
        self, source_path: str, status: str, error_message: str | None = None
    ) -> None:
        """Update sync status for a note.

        Args:
            source_path: Relative path to note file
            status: Sync status (pending, processing, completed, failed)
            error_message: Optional error message if failed
        """
        cursor = self.conn.cursor()
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
        self.conn.commit()

    def get_note_index(self, source_path: str) -> dict | None:
        """Get note index entry.

        Args:
            source_path: Relative path to note file

        Returns:
            Note index record or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM note_index WHERE source_path = ?", (source_path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_notes_index(self) -> list[dict]:
        """Get all notes from index.

        Returns:
            List of note index records
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM note_index ORDER BY source_path")
        return [dict(row) for row in cursor.fetchall()]

    def get_notes_by_status(self, status: str) -> list[dict]:
        """Get notes by sync status.

        Args:
            status: Sync status to filter by

        Returns:
            List of note index records
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM note_index WHERE sync_status = ? ORDER BY source_path",
            (status,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # Card Index Methods

    def upsert_card_index(
        self,
        source_path: str,
        card_index: int,
        lang: str,
        slug: str | None = None,
        anki_guid: int | None = None,
        note_id: str | None = None,
        note_title: str | None = None,
        content_hash: str | None = None,
        status: str = "expected",
        in_obsidian: bool = True,
        in_anki: bool = False,
        in_database: bool = False,
    ) -> None:
        """Insert or update a card in the index.

        Args:
            source_path: Relative path to source note
            card_index: Card index within note (1-based)
            lang: Language code
            slug: Card slug
            anki_guid: Anki note ID
            note_id: Note ID from frontmatter
            note_title: Note title
            content_hash: Content hash
            status: Card status (expected, new, exists, modified, deleted)
            in_obsidian: Whether card exists in Obsidian vault
            in_anki: Whether card exists in Anki
            in_database: Whether card exists in sync database
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO card_index (
                source_path, card_index, lang, slug, anki_guid, note_id,
                note_title, content_hash, status, in_obsidian, in_anki,
                in_database, last_indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_path, card_index, lang) DO UPDATE SET
                slug = excluded.slug,
                anki_guid = excluded.anki_guid,
                note_id = excluded.note_id,
                note_title = excluded.note_title,
                content_hash = excluded.content_hash,
                status = excluded.status,
                in_obsidian = excluded.in_obsidian,
                in_anki = excluded.in_anki,
                in_database = excluded.in_database,
                last_indexed_at = CURRENT_TIMESTAMP
        """,
            (
                source_path,
                card_index,
                lang,
                slug,
                anki_guid,
                note_id,
                note_title,
                content_hash,
                status,
                in_obsidian,
                in_anki,
                in_database,
            ),
        )
        self.conn.commit()

    def get_card_index_by_source(self, source_path: str) -> list[dict]:
        """Get all cards for a note.

        Args:
            source_path: Relative path to note file

        Returns:
            List of card index records
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM card_index WHERE source_path = ? ORDER BY card_index, lang",
            (source_path,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_card_index_by_slug(self, slug: str) -> dict | None:
        """Get card index entry by slug.

        Args:
            slug: Card slug

        Returns:
            Card index record or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM card_index WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_index_statistics(self) -> dict:
        """Get statistics from the index.

        Returns:
            Dictionary with index statistics
        """
        cursor = self.conn.cursor()

        # Note statistics
        cursor.execute("SELECT COUNT(*) FROM note_index")
        total_notes = cursor.fetchone()[0]

        cursor.execute(
            "SELECT sync_status, COUNT(*) FROM note_index GROUP BY sync_status"
        )
        note_status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Card statistics
        cursor.execute("SELECT COUNT(*) FROM card_index")
        total_cards = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM card_index WHERE in_obsidian = 1")
        cards_in_obsidian = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM card_index WHERE in_anki = 1")
        cards_in_anki = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM card_index WHERE in_database = 1")
        cards_in_database = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) FROM card_index GROUP BY status")
        card_status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total_notes": total_notes,
            "note_status": note_status_counts,
            "total_cards": total_cards,
            "cards_in_obsidian": cards_in_obsidian,
            "cards_in_anki": cards_in_anki,
            "cards_in_database": cards_in_database,
            "card_status": card_status_counts,
        }

    def clear_index(self) -> None:
        """Clear all index data (for rebuilding)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM note_index")
        cursor.execute("DELETE FROM card_index")
        self.conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self) -> "StateDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
