"""Repository for card index data operations."""

from obsidian_anki_sync.infrastructure.database_connection_manager import (
    DatabaseConnectionManager,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class CardIndexRepository:
    """Repository for card index data operations."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize card index repository.

        Args:
            connection_manager: Database connection manager
        """
        self._connection_manager = connection_manager

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
        with self._connection_manager.transaction() as conn:
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

    def get_card_index_by_source(self, source_path: str) -> list[dict]:
        """Get all cards for a note.

        Args:
            source_path: Relative path to note file

        Returns:
            List of card index records
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM card_index WHERE source_path = ? ORDER BY card_index, lang",
            (source_path,),
            "get_card_index_by_source",
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_card_index_by_slug(self, slug: str) -> dict | None:
        """Get card index entry by slug.

        Args:
            slug: Card slug

        Returns:
            Card index record or None
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM card_index WHERE slug = ?", (slug,), "get_card_index_by_slug"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_index_statistics(self) -> dict:
        """Get statistics from the index.

        Returns:
            Dictionary with index statistics
        """
        cursor = self._connection_manager.execute_query(
            "SELECT COUNT(*) FROM note_index", operation="get_index_statistics_notes"
        )
        total_notes = cursor.fetchone()[0]

        cursor = self._connection_manager.execute_query(
            "SELECT sync_status, COUNT(*) FROM note_index GROUP BY sync_status",
            operation="get_index_statistics_note_status",
        )
        note_status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = self._connection_manager.execute_query(
            "SELECT COUNT(*) FROM card_index", operation="get_index_statistics_cards"
        )
        total_cards = cursor.fetchone()[0]

        cursor = self._connection_manager.execute_query(
            "SELECT COUNT(*) FROM card_index WHERE in_obsidian = 1",
            operation="get_index_statistics_obsidian",
        )
        cards_in_obsidian = cursor.fetchone()[0]

        cursor = self._connection_manager.execute_query(
            "SELECT COUNT(*) FROM card_index WHERE in_anki = 1",
            operation="get_index_statistics_anki",
        )
        cards_in_anki = cursor.fetchone()[0]

        cursor = self._connection_manager.execute_query(
            "SELECT COUNT(*) FROM card_index WHERE in_database = 1",
            operation="get_index_statistics_database",
        )
        cards_in_database = cursor.fetchone()[0]

        cursor = self._connection_manager.execute_query(
            "SELECT status, COUNT(*) FROM card_index GROUP BY status",
            operation="get_index_statistics_card_status",
        )
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
        """Clear all card index data."""
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM card_index")
