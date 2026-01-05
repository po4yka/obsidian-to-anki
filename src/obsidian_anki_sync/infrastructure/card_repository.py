"""Repository for card data operations."""

import json
from typing import Any

from obsidian_anki_sync.domain.entities.card import Card as DomainCard
from obsidian_anki_sync.domain.entities.card import CardManifest
from obsidian_anki_sync.infrastructure.database_connection_manager import (
    DatabaseConnectionManager,
)
from obsidian_anki_sync.models import Card as ModelCard
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class CardRepository:
    """Repository for card data operations."""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """Initialize card repository.

        Args:
            connection_manager: Database connection manager
        """
        self._connection_manager = connection_manager

    def insert_card(self, card: ModelCard, anki_guid: int) -> None:
        """Insert a new card record."""
        import time

        start_time = time.time()
        logger.debug("db_transaction_start", operation="insert_card", slug=card.slug)
        try:
            with self._connection_manager.transaction() as conn:
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
            duration = time.time() - start_time
            logger.debug(
                "db_transaction_committed",
                operation="insert_card",
                slug=card.slug,
                duration=round(duration, 4),
            )
        except Exception as e:
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
        with self._connection_manager.transaction() as conn:
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

    def get_by_slug(self, slug: str) -> dict | None:
        """Get card by slug."""
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards WHERE slug = ?", (slug,), "get_by_slug"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_guid(self, anki_guid: int) -> dict | None:
        """Get card by Anki GUID."""
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards WHERE anki_guid = ?", (anki_guid,), "get_by_guid"
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_source(self, source_path: str) -> list[dict]:
        """Get all cards from a source note."""
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards WHERE source_path = ?", (source_path,), "get_by_source"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_cards(self) -> list[DomainCard]:
        """Get all cards."""
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards", operation="get_all_cards"
        )
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

    def get_all_cards_raw(self) -> list[dict]:
        """Get all cards as raw dictionary records.

        Use this for operations that need database-specific fields
        like creation_status, retry_count, etc. that aren't part
        of the domain Card entity.

        Returns:
            List of card records as dictionaries
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards", operation="get_all_cards_raw"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_processed_note_paths(self) -> set[str]:
        """Get set of all note paths that have been processed.

        Returns:
            Set of source_path values from cards that have been synced
        """
        cursor = self._connection_manager.execute_query(
            "SELECT DISTINCT source_path FROM cards",
            operation="get_processed_note_paths",
        )
        return {row["source_path"] for row in cursor.fetchall()}

    def delete_card(self, slug: str) -> None:
        """Delete a card record."""
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cards WHERE slug = ?", (slug,))

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
        with self._connection_manager.transaction() as conn:
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
        with self._connection_manager.transaction() as conn:
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

    def upsert_card_extended(
        self,
        card: ModelCard,
        anki_guid: int | None,
        fields: dict[str, str],
        tags: list[str],
        deck_name: str,
        apf_html: str,
        creation_status: str = "success",
    ) -> None:
        """Insert or update card with full content.

        Args:
            card: Card object
            anki_guid: Anki note ID (can be None for pending cards)
            fields: Mapped fields dict
            tags: List of tags
            deck_name: Target deck name
            apf_html: Full APF HTML content
            creation_status: Status of creation (success, pending, failed)
        """
        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cards (
                    slug, slug_base, lang, source_path, source_anchor,
                    card_index, anki_guid, content_hash, note_id, note_title,
                    note_type, card_guid, apf_html, fields_json, tags_json,
                    deck_name, creation_status, synced_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(slug) DO UPDATE SET
                    anki_guid = excluded.anki_guid,
                    content_hash = excluded.content_hash,
                    note_title = excluded.note_title,
                    note_type = excluded.note_type,
                    card_guid = excluded.card_guid,
                    apf_html = excluded.apf_html,
                    fields_json = excluded.fields_json,
                    tags_json = excluded.tags_json,
                    deck_name = excluded.deck_name,
                    creation_status = excluded.creation_status,
                    synced_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
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
                    creation_status,
                ),
            )

    def upsert_batch_extended(
        self,
        cards_data: list[dict[str, Any]],
    ) -> None:
        """Insert or update multiple cards in a single transaction.

        Args:
            cards_data: List of dicts containing:
                - card: ModelCard
                - anki_guid: int | None
                - fields: dict[str, str]
                - tags: list[str]
                - deck_name: str
                - apf_html: str
                - creation_status: str (optional, default="success")
        """
        if not cards_data:
            return

        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()

            # Prepare data for executemany
            params_list = []
            for item in cards_data:
                card = item["card"]
                params_list.append(
                    (
                        card.slug,
                        card.manifest.slug_base,
                        card.lang,
                        card.manifest.source_path,
                        card.manifest.source_anchor,
                        card.manifest.card_index,
                        item.get("anki_guid"),
                        card.content_hash,
                        card.manifest.note_id,
                        card.manifest.note_title,
                        card.note_type,
                        card.guid,
                        item["apf_html"],
                        json.dumps(item["fields"]),
                        json.dumps(item["tags"]),
                        item["deck_name"],
                        item.get("creation_status", "success"),
                    )
                )

            cursor.executemany(
                """
                INSERT INTO cards (
                    slug, slug_base, lang, source_path, source_anchor,
                    card_index, anki_guid, content_hash, note_id, note_title,
                    note_type, card_guid, apf_html, fields_json, tags_json,
                    deck_name, creation_status, synced_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(slug) DO UPDATE SET
                    anki_guid = excluded.anki_guid,
                    content_hash = excluded.content_hash,
                    note_title = excluded.note_title,
                    note_type = excluded.note_type,
                    card_guid = excluded.card_guid,
                    apf_html = excluded.apf_html,
                    fields_json = excluded.fields_json,
                    tags_json = excluded.tags_json,
                    deck_name = excluded.deck_name,
                    creation_status = excluded.creation_status,
                    synced_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
            """,
                params_list,
            )
            logger.debug(
                "db_batch_committed",
                operation="upsert_batch_extended",
                count=len(cards_data),
            )

    def get_pending_cards(self) -> list[dict[str, Any]]:
        """Get all cards with 'pending' creation status.

        Returns:
            List of pending card records
        """
        cursor = self._connection_manager.execute_query(
            "SELECT * FROM cards WHERE creation_status = 'pending'",
            operation="get_pending_cards",
        )
        return [dict(row) for row in cursor.fetchall()]

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

        with self._connection_manager.transaction() as conn:
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
                    deck_name, creation_status, synced_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(slug) DO UPDATE SET
                    anki_guid = excluded.anki_guid,
                    content_hash = excluded.content_hash,
                    note_title = excluded.note_title,
                    note_type = excluded.note_type,
                    card_guid = excluded.card_guid,
                    apf_html = excluded.apf_html,
                    fields_json = excluded.fields_json,
                    tags_json = excluded.tags_json,
                    deck_name = excluded.deck_name,
                    creation_status = excluded.creation_status,
                    synced_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
            """,
                insert_data,
            )
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

        with self._connection_manager.transaction() as conn:
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
            logger.info("cards_updated_batch", count=len(cards_data))

    def delete_cards_batch(self, slugs: list[str]) -> None:
        """Delete multiple cards in a single batch operation.

        Args:
            slugs: List of card slugs to delete
        """
        if not slugs:
            return

        with self._connection_manager.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "DELETE FROM cards WHERE slug = ?",
                [(slug,) for slug in slugs],
            )
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
        with self._connection_manager.transaction() as conn:
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
