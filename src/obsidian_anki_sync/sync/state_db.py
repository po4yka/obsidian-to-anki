"""SQLite database for tracking sync state."""

import sqlite3
from pathlib import Path
from typing import Optional

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

    def get_by_slug(self, slug: str) -> Optional[dict]:
        """Get card by slug."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM cards WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_guid(self, anki_guid: int) -> Optional[dict]:
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

    def delete_card(self, slug: str) -> None:
        """Delete a card record."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM cards WHERE slug = ?", (slug,))
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
