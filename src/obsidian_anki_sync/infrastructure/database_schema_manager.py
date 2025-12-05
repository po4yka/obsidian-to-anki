"""Database schema management for SQLite database.

Handles table creation, index creation, and schema migrations.
"""

import sqlite3
from pathlib import Path

from obsidian_anki_sync.infrastructure.database_connection_manager import DatabaseConnectionManager
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseSchemaManager:
    """Manages database schema creation and migrations."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize schema manager.

        Args:
            connection_manager: Database connection manager instance
        """
        self._connection_manager = connection_manager

    def initialize_schema(self) -> None:
        """Create tables if they don't exist.

        Uses a temporary connection to initialize the schema once during
        construction. This connection is not stored in thread-local storage.
        """
        # Create temporary connection for schema initialization
        db_path = self._connection_manager._db_path
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        cursor = conn.cursor()

        # Create tables
        self._create_cards_table(cursor)
        self._create_sync_progress_table(cursor)
        self._create_note_index_table(cursor)
        self._create_card_index_table(cursor)
        self._create_sync_checkpoints_table(cursor)

        # Create indexes
        self._create_indexes(cursor)

        # Add extended schema columns
        self._add_extended_schema_columns(cursor)

        conn.commit()
        conn.close()

    def _create_cards_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the cards table."""
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

    def _create_sync_progress_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the sync progress table."""
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

    def _create_note_index_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the note index table."""
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

    def _create_card_index_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the card index table."""
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

    def _create_sync_checkpoints_table(self, cursor: sqlite3.Cursor) -> None:
        """Create the sync checkpoints table."""
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

    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_source ON cards(source_path)",
            "CREATE INDEX IF NOT EXISTS idx_guid ON cards(anki_guid)",
            "CREATE INDEX IF NOT EXISTS idx_card_guid ON cards(card_guid)",
            "CREATE INDEX IF NOT EXISTS idx_source_lang ON cards(source_path, lang)",
            "CREATE INDEX IF NOT EXISTS idx_card_index ON cards(card_index)",
            "CREATE INDEX IF NOT EXISTS idx_progress_phase ON sync_progress(phase)",
            "CREATE INDEX IF NOT EXISTS idx_progress_updated ON sync_progress(updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_note_status ON note_index(sync_status)",
            "CREATE INDEX IF NOT EXISTS idx_note_id ON note_index(note_id)",
            "CREATE INDEX IF NOT EXISTS idx_card_source ON card_index(source_path)",
            "CREATE INDEX IF NOT EXISTS idx_card_status ON card_index(status)",
            "CREATE INDEX IF NOT EXISTS idx_card_slug ON card_index(slug)",
            "CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON sync_checkpoints(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_checkpoint_timestamp ON sync_checkpoints(timestamp)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

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
                cursor.execute(
                    f"ALTER TABLE cards ADD COLUMN {col_name} {col_type}")
                logger.debug("added_column_to_cards_table", column=col_name)

        # Add index for creation_status (after column is created)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_creation_status ON cards(creation_status)
        """
        )
