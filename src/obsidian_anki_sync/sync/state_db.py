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
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .progress import SyncProgress

from obsidian_anki_sync.domain.entities.card import Card as DomainCard
from obsidian_anki_sync.domain.entities.card import CardManifest
from obsidian_anki_sync.domain.entities.note import Note as DomainNote
from obsidian_anki_sync.domain.interfaces.state_repository import IStateRepository
from obsidian_anki_sync.models import Card as ModelCard
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class StateDB(IStateRepository):
    """SQLite database for tracking card state.

    Thread Safety:
        This class is thread-safe. Each thread gets its own SQLite connection
        via thread-local storage. The database uses WAL mode for concurrent
        access. All connections are tracked and properly closed on exit.

    Async Compatibility:
        This class uses synchronous sqlite3. If used in async contexts, it will
        block the event loop. For async usage, consider using aiosqlite or
        running database operations in a thread pool executor.

    WAL Mode:
        The database uses WAL (Write-Ahead Logging) mode for better concurrency.
        WAL mode allows concurrent reads while writes are in progress.

    Usage:
        # As context manager (recommended)
        with StateDB(db_path) as db:
            db.insert_card(card, anki_guid)

        # Direct usage
        db = StateDB(db_path)
        try:
            db.insert_card(card, anki_guid)
        finally:
            db.close()
    """

    def __init__(self, db_path: Path):
        """Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file

        Note:
            Creates thread-local connections on-demand. The schema is initialized
            once using a temporary connection. Each thread that accesses the
            database will get its own connection automatically.
        """
        self._db_path = db_path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection.

        Creates a new connection for the current thread if one doesn't exist.
        Each connection is configured with WAL mode and tracked for cleanup.

        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._connections_lock:
                self._connections.append(conn)
            logger.debug(
                "db_connection_created",
                thread_id=threading.get_ident(),
                total_connections=len(self._connections),
                db_path=str(self._db_path),
            )
        return self._local.conn  # type: ignore[no-any-return]

    def _execute_query(
        self, query: str, params: tuple = (), operation: str = "query"
    ) -> sqlite3.Cursor:
        """Execute a database query with logging.

        Args:
            query: SQL query string
            params: Query parameters
            operation: Operation name for logging

        Returns:
            Cursor with results
        """
        import time

        start_time = time.time()
        try:
            conn = self._get_connection()
            cursor = conn.execute(query, params)
            duration = time.time() - start_time

            # Log slow queries (>100ms)
            if duration > 0.1:
                logger.warning(
                    "db_slow_query",
                    operation=operation,
                    duration=round(duration, 3),
                    params_count=len(params),
                    query_preview=query[:100],
                )
            else:
                logger.debug(
                    "db_query",
                    operation=operation,
                    duration=round(duration, 4),
                    params_count=len(params),
                )

            return cursor
        except sqlite3.Error as e:
            duration = time.time() - start_time
            logger.error(
                "db_query_error",
                operation=operation,
                duration=round(duration, 3),
                error=str(e),
                error_type=type(e).__name__,
                query_preview=query[:100],
            )
            raise

    def _init_schema(self) -> None:
        """Create tables if they don't exist.

        Uses a temporary connection to initialize the schema once during
        construction. This connection is not stored in thread-local storage.
        """
        # Create temporary connection for schema initialization
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        cursor = conn.cursor()
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
        # Additional indexes for performance (non-extended columns only)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source_lang ON cards(source_path, lang)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_card_index ON cards(card_index)
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

        # Add extended schema columns for atomicity support
        self._add_extended_schema_columns(cursor)

        # Checkpoint table for resumable syncs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                notes_processed INTEGER DEFAULT 0,
                cards_generated INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checkpoint_data TEXT
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON sync_checkpoints(session_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp ON sync_checkpoints(timestamp)
        """
        )

        conn.commit()
        conn.close()

    def _add_extended_schema_columns(self, cursor: sqlite3.Cursor) -> None:
        """Add extended columns for full card tracking and atomicity.

        These columns store complete card content for recovery and enable
        tracking of creation/update status for better error handling.
        """
        # Define columns to add with their SQL types
        columns_to_add = {
            "apf_html": "TEXT",
            "fields_json": "TEXT",
            "tags_json": "TEXT",
            "deck_name": "TEXT",
            "creation_status": 'TEXT DEFAULT "success"',
            "last_error": "TEXT",
            "retry_count": "INTEGER DEFAULT 0",
            "synced_at": "TIMESTAMP",
        }

        # Get existing columns
        cursor.execute("PRAGMA table_info(cards)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add missing columns
        for col_name, col_type in columns_to_add.items():
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE cards ADD COLUMN {col_name} {col_type}")
                logger.debug("added_column_to_cards_table", column=col_name)

        # Add index for creation_status (after column is created)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creation_status ON cards(creation_status)
        """
        )

    def insert_card(self, card: ModelCard, anki_guid: int) -> None:
        """Insert a new card record."""
        import time

        start_time = time.time()
        logger.debug("db_transaction_start", operation="insert_card", slug=card.slug)
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
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
            conn.commit()
            duration = time.time() - start_time
            logger.debug(
                "db_transaction_committed",
                operation="insert_card",
                slug=card.slug,
                duration=round(duration, 4),
            )
        except sqlite3.Error as e:
            duration = time.time() - start_time
            logger.error(
                "db_transaction_error",
                operation="insert_card",
                slug=card.slug,
                duration=round(duration, 3),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def update_card(self, card: ModelCard) -> None:
        """Update existing card record."""
        conn = self._get_connection()
        cursor = conn.cursor()
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
        conn.commit()

    def get_by_slug(self, slug: str) -> dict | None:
        """Get card by slug."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_guid(self, anki_guid: int) -> dict | None:
        """Get card by Anki GUID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE anki_guid = ?", (anki_guid,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_source(self, source_path: str) -> list[dict]:
        """Get all cards from a source note."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE source_path = ?", (source_path,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_cards(self) -> list[DomainCard]:
        """Get all cards."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cards")
        cards: list[DomainCard] = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Convert database row to domain Card entity
            # Extract required fields with defaults
            try:
                slug = row_dict.get("slug", "")
                lang = row_dict.get("lang", "en")
                slug_base = row_dict.get(
                    "slug_base", slug.rsplit("-", 1)[0] if "-" in slug else slug
                )
                source_path = row_dict.get("source_path", "")
                source_anchor = row_dict.get("source_anchor", "")
                note_id = row_dict.get("note_id", "")
                note_title = row_dict.get("note_title", "")
                card_index = row_dict.get("card_index", 0)
                guid = row_dict.get("guid") or row_dict.get("anki_guid")

                # Ensure required fields have defaults
                if not source_path:
                    source_path = "unknown"
                if not source_anchor:
                    source_anchor = f"qa-{card_index}"
                if not note_id:
                    note_id = "unknown"
                if not note_title:
                    note_title = "Unknown"

                # CardManifest requires guid to be str | None, not empty string
                guid_str: str | None = str(guid) if guid else None
                manifest = CardManifest(
                    slug=slug,
                    slug_base=slug_base,
                    lang=lang,
                    source_path=source_path,
                    source_anchor=source_anchor,
                    note_id=note_id,
                    note_title=note_title,
                    card_index=card_index,
                    guid=guid_str,
                    hash6=row_dict.get("hash6"),
                )
                # Get apf_html - it might not be in the database, use a default
                apf_html = row_dict.get("apf_html")
                if not apf_html:
                    # Generate minimal APF HTML if missing
                    apf_html = '<div class="front">Question</div><div class="back">Answer</div>'

                card = DomainCard(
                    slug=slug,
                    language=lang,
                    apf_html=apf_html,
                    manifest=manifest,
                    note_type=row_dict.get("note_type", "APF::Simple"),
                    tags=(
                        row_dict.get("tags", "").split()
                        if isinstance(row_dict.get("tags"), str)
                        else (row_dict.get("tags") or [])
                    ),
                    anki_guid=str(guid) if guid else None,
                )
                cards.append(card)
            except Exception as e:
                logger.warning("failed_to_convert_card", error=str(e), row=row_dict)
                continue
        return cards

    def get_processed_note_paths(self) -> set[str]:
        """Get set of all note paths that have been processed.

        Returns:
            Set of source_path values from cards that have been synced
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source_path FROM cards")
        return {row["source_path"] for row in cursor.fetchall()}

    def delete_card(self, slug: str) -> None:
        """Delete a card record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cards WHERE slug = ?", (slug,))
        conn.commit()

    def insert_card_extended(
        self,
        card: ModelCard,
        anki_guid: int,
        fields: dict[str, str],
        tags: list[str],
        deck_name: str,
        apf_html: str,
    ) -> None:
        """Insert card with full content storage for atomicity support.

        Args:
            card: Card object to insert
            anki_guid: Anki note ID
            fields: Mapped fields dict
            tags: List of tags
            deck_name: Target deck name
            apf_html: Full APF HTML content
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO cards (
                slug, slug_base, lang, source_path, source_anchor,
                card_index, anki_guid, content_hash, note_id, note_title,
                note_type, card_guid, apf_html, fields_json, tags_json,
                deck_name, creation_status, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                apf_html,
                json.dumps(fields),
                json.dumps(tags),
                deck_name,
                "success",
            ),
        )
        conn.commit()

    def update_card_extended(
        self, card: ModelCard, fields: dict[str, str], tags: list[str], apf_html: str
    ) -> None:
        """Update card with full content for atomicity support.

        Args:
            card: Card object to update
            fields: Mapped fields dict
            tags: List of tags
            apf_html: Full APF HTML content
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE cards
            SET content_hash = ?,
                note_title = ?,
                note_type = ?,
                card_guid = ?,
                apf_html = ?,
                fields_json = ?,
                tags_json = ?,
                updated_at = CURRENT_TIMESTAMP,
                synced_at = CURRENT_TIMESTAMP
            WHERE slug = ?
            """,
            (
                card.content_hash,
                card.manifest.note_title,
                card.note_type,
                card.guid,
                apf_html,
                json.dumps(fields),
                json.dumps(tags),
                card.slug,
            ),
        )
        conn.commit()

    def insert_cards_batch(
        self,
        cards_data: list[tuple[ModelCard, int, dict[str, str], list[str], str]],
        deck_name: str,
    ) -> None:
        """Insert multiple cards in a single batch operation.

        Args:
            cards_data: List of tuples (card, anki_guid, fields, tags, apf_html)
            deck_name: Target deck name
        """
        if not cards_data:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        insert_data = []
        for card, anki_guid, fields, tags, apf_html in cards_data:
            insert_data.append(
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
                    apf_html,
                    json.dumps(fields),
                    json.dumps(tags),
                    deck_name,
                    "success",
                )
            )

        cursor.executemany(
            """
            INSERT INTO cards (
                slug, slug_base, lang, source_path, source_anchor,
                card_index, anki_guid, content_hash, note_id, note_title,
                note_type, card_guid, apf_html, fields_json, tags_json,
                deck_name, creation_status, synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            insert_data,
        )
        conn.commit()
        logger.info("cards_inserted_batch", count=len(cards_data))

    def update_cards_batch(
        self,
        cards_data: list[tuple[ModelCard, dict[str, str], list[str]]],
    ) -> None:
        """Update multiple cards in a single batch operation.

        Args:
            cards_data: List of tuples (card, fields, tags)
        """
        if not cards_data:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        update_data = []
        for card, fields, tags in cards_data:
            update_data.append(
                (
                    card.content_hash,
                    card.manifest.note_title,
                    card.note_type,
                    card.guid,
                    card.apf_html,
                    json.dumps(fields),
                    json.dumps(tags),
                    card.slug,
                )
            )

        cursor.executemany(
            """
            UPDATE cards
            SET content_hash = ?,
                note_title = ?,
                note_type = ?,
                card_guid = ?,
                apf_html = ?,
                fields_json = ?,
                tags_json = ?,
                updated_at = CURRENT_TIMESTAMP,
                synced_at = CURRENT_TIMESTAMP
            WHERE slug = ?
            """,
            update_data,
        )
        conn.commit()
        logger.info("cards_updated_batch", count=len(cards_data))

    def delete_cards_batch(self, slugs: list[str]) -> None:
        """Delete multiple cards in a single batch operation.

        Args:
            slugs: List of card slugs to delete
        """
        if not slugs:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            "DELETE FROM cards WHERE slug = ?",
            [(slug,) for slug in slugs],
        )
        conn.commit()
        logger.info("cards_deleted_batch", count=len(slugs))

    def update_card_status(
        self,
        slug: str,
        status: str,
        error_message: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        """Update card creation/update status for error tracking.

        Args:
            slug: Card slug
            status: Status value (e.g., 'success', 'failed', 'pending')
            error_message: Optional error message
            increment_retry: Whether to increment retry counter
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        if increment_retry:
            cursor.execute(
                """
                UPDATE cards
                SET creation_status = ?,
                    last_error = ?,
                    retry_count = retry_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE slug = ?
                """,
                (status, error_message, slug),
            )
        else:
            cursor.execute(
                """
                UPDATE cards
                SET creation_status = ?,
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE slug = ?
                """,
                (status, error_message, slug),
            )
        conn.commit()

    def save_progress(self, progress: "SyncProgress") -> None:
        """Save sync progress state.

        Args:
            progress: SyncProgress instance
        """
        conn = self._get_connection()
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
        conn.commit()

    def get_progress(self, session_id: str) -> "SyncProgress | None":
        """Get sync progress by session ID.

        Args:
            session_id: Session identifier

        Returns:
            SyncProgress instance or None
        """
        from .progress import NoteProgress, SyncPhase, SyncProgress

        conn = self._get_connection()
        cursor = conn.cursor()
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

    def get_all_progress(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent sync progress records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of progress record dictionaries
        """

        conn = self._get_connection()
        cursor = conn.cursor()
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

    def get_incomplete_progress(self) -> list[dict[str, Any]]:
        """Get all incomplete sync sessions.

        Returns:
            List of progress records with incomplete syncs
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_id, phase, started_at, updated_at,
                   total_notes, notes_processed, errors
            FROM sync_progress
            WHERE phase IN ('initializing', 'indexing', 'scanning', 'generating', 'determining_actions', 'applying_changes', 'interrupted')
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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sync_progress WHERE session_id = ?", (session_id,))
        conn.commit()

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
        conn = self._get_connection()
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
        conn.commit()

    def update_note_sync_status(
        self, source_path: str, status: str, error_message: str | None = None
    ) -> None:
        """Update sync status for a note.

        Args:
            source_path: Relative path to note file
            status: Sync status (pending, processing, completed, failed)
            error_message: Optional error message if failed
        """
        conn = self._get_connection()
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
        conn.commit()

    def get_note_index(self, source_path: str) -> dict | None:
        """Get note index entry.

        Args:
            source_path: Relative path to note file

        Returns:
            Note index record or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM note_index WHERE source_path = ?", (source_path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_notes_index(self) -> list[dict]:
        """Get all notes from index.

        Returns:
            List of note index records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM note_index ORDER BY source_path")
        return [dict(row) for row in cursor.fetchall()]

    def get_notes_by_status(self, status: str) -> list[dict]:
        """Get notes by sync status.

        Args:
            status: Sync status to filter by

        Returns:
            List of note index records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
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
        conn = self._get_connection()
        cursor = conn.cursor()
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
        conn.commit()

    def get_card_index_by_source(self, source_path: str) -> list[dict]:
        """Get all cards for a note.

        Args:
            source_path: Relative path to note file

        Returns:
            List of card index records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM card_index WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_index_statistics(self) -> dict:
        """Get statistics from the index.

        Returns:
            Dictionary with index statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM note_index")
        cursor.execute("DELETE FROM card_index")
        conn.commit()

    # Checkpoint methods removed (were unused dead code)
    # Table sync_checkpoints kept for backward compatibility

    # IStateRepository interface implementation

    def get_note_by_id(self, note_id: str) -> DomainNote | None:
        """Retrieve a note by its ID."""
        return None

    def get_notes_by_path(self, file_path: str) -> list[DomainNote]:
        """Retrieve notes by file path."""
        return []

    def save_note(self, note: DomainNote) -> None:
        """Save a note to the repository."""

    def delete_note(self, note_id: str) -> None:
        """Delete a note from the repository."""

    def get_card_by_slug(self, slug: str) -> DomainCard | None:
        """Retrieve a card by its slug."""
        card_data = self.get_by_slug(slug)
        if card_data:
            return None
        return None

    def get_cards_by_note_id(self, note_id: str) -> list[DomainCard]:
        """Retrieve all cards for a note."""
        return []

    def save_card(self, card: DomainCard) -> None:
        """Save a card to the repository."""

    # delete_card is defined above at line 423 - removed duplicate

    def get_all_notes(self) -> list[DomainNote]:
        """Retrieve all notes."""
        return []

    # get_all_cards is defined above at line 405 - removed duplicate that caused recursion

    def get_sync_stats(self) -> dict[str, Any]:
        """Get synchronization statistics."""
        return {}

    def save_sync_session(self, session_data: dict[str, Any]) -> str:
        """Save sync session data."""
        return "session_id"

    def get_sync_session(self, session_id: str) -> dict[str, Any | None]:
        """Retrieve sync session data."""
        progress = self.get_progress(session_id)
        if progress:
            # Convert progress to dict with Any | None values
            return {"progress": progress}  # type: ignore[return-value]
        # Return empty dict instead of None to match interface
        return {}

    def update_sync_progress(
        self, session_id: str, progress_data: dict[str, Any]
    ) -> None:
        """Update sync progress for a session."""

    def get_content_hash(self, resource_id: str) -> str | None:
        """Get stored content hash for a resource."""
        return None

    def save_content_hash(self, resource_id: str, hash_value: str) -> None:
        """Save content hash for a resource."""

    def clear_expired_data(self, max_age_days: int) -> int:
        """Clear expired data from repository."""
        return 0

    def close(self) -> None:
        """Close all thread-local database connections.

        This method properly closes all connections that were created across
        different threads. It's safe to call multiple times.
        """
        with self._connections_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(
                        "error_closing_connection",
                        error=str(e),
                    )
            self._connections.clear()

        # Clear thread-local connection for current thread
        if hasattr(self._local, "conn"):
            self._local.conn = None

        logger.debug(
            "closed_all_connections",
            thread_id=threading.get_ident(),
        )

    def __enter__(self) -> "StateDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
