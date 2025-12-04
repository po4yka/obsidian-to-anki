"""Recovery utilities for orphaned and inconsistent cards.

This module provides utilities to detect and recover from inconsistent
states where cards exist in Anki but not in the database, or vice versa.

These inconsistencies can occur due to:
- Interrupted syncs
- Database failures after Anki operations
- Manual card deletions in Anki
- Database corruption
"""

from typing import Any

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.exceptions import AnkiConnectError
from obsidian_anki_sync.utils.logging import get_logger

from .state_db import StateDB

logger = get_logger(__name__)


class CardRecovery:
    """Utilities for recovering from inconsistent card states.

    This class provides methods to detect orphaned cards (cards that exist
    in one system but not the other) and verify consistency between Anki
    and the database.
    """

    def __init__(self, anki_client: AnkiClient, db: StateDB):
        """Initialize recovery utility.

        Args:
            anki_client: AnkiConnect client for Anki operations
            db: State database for persistence
        """
        self.anki = anki_client
        self.db = db

    def find_orphaned_cards(self) -> dict[str, list[Any]]:
        """Find cards in inconsistent states.

        Scans both Anki and the database to identify:
        - Cards in Anki but not in database (orphaned in Anki)
        - Cards in database but not in Anki (orphaned in DB)
        - Cards with different content in Anki vs DB (inconsistent)

        Returns:
            Dictionary with three lists:
            - 'orphaned_in_anki': Anki note IDs not in database
            - 'orphaned_in_db': Card slugs in DB but not in Anki
            - 'inconsistent': Card slugs with different content (future use)
        """
        results: dict[str, list[str]] = {
            "orphaned_in_anki": [],  # In Anki but not in DB
            "orphaned_in_db": [],  # In DB but not in Anki
            "inconsistent": [],  # Different content in Anki vs DB
        }

        # Get all cards from database (returns Card domain objects)
        db_cards = self.db.get_all_cards()
        db_by_guid = {c.anki_guid: c for c in db_cards if c.anki_guid}

        logger.info("checking_for_orphaned_cards", db_cards=len(db_cards))

        # Get all cards from Anki
        try:
            all_anki_guids = self.anki.find_notes("deck:*")
            logger.info("found_anki_notes", count=len(all_anki_guids))

            for guid in all_anki_guids:
                if guid not in db_by_guid:
                    results["orphaned_in_anki"].append(guid)

        except AnkiConnectError as e:
            logger.error("failed_to_fetch_anki_cards", error=str(e))

        # Check DB cards that should be in Anki
        for card in db_cards:
            if not card.anki_guid:
                # Card has no Anki GUID recorded
                results["orphaned_in_db"].append(card.slug)
                continue

            try:
                anki_info = self.anki.notes_info([card.anki_guid])
                if not anki_info:
                    # Card deleted from Anki but still in DB
                    results["orphaned_in_db"].append(card.slug)
            except AnkiConnectError:
                # Failed to query - assume orphaned
                results["orphaned_in_db"].append(card.slug)

        logger.info(
            "orphaned_cards_found",
            orphaned_in_anki=len(results["orphaned_in_anki"]),
            orphaned_in_db=len(results["orphaned_in_db"]),
        )

        return results

    def verify_card_consistency(self, slug: str) -> bool:
        """Verify card is consistent between Anki and database.

        Args:
            slug: Card slug to verify

        Returns:
            True if card is consistent, False otherwise
        """
        db_card = self.db.get_by_slug(slug)
        if not db_card or not db_card.get("anki_guid"):
            logger.warning("card_not_found_in_db", slug=slug)
            return False

        try:
            anki_info = self.anki.notes_info([db_card["anki_guid"]])
            if not anki_info:
                logger.warning(
                    "card_not_found_in_anki", slug=slug, anki_guid=db_card["anki_guid"]
                )
                return False

            logger.info("card_is_consistent", slug=slug)
            return True

        except AnkiConnectError as e:
            logger.error("failed_to_verify_card", slug=slug, error=str(e))
            return False

    def get_failed_cards(self) -> list[dict[str, Any]]:
        """Get list of cards with failed creation/update status.

        Returns:
            List of card records with 'failed' status
        """
        # Use raw method to access database-specific fields like creation_status
        all_cards = self.db.get_all_cards_raw()
        failed_cards = [c for c in all_cards if c.get("creation_status") == "failed"]

        logger.info("found_failed_cards", count=len(failed_cards))
        return failed_cards

    def get_cards_needing_retry(self, max_retries: int = 3) -> list[dict[str, Any]]:
        """Get list of cards that failed but haven't exceeded retry limit.

        Args:
            max_retries: Maximum number of retries allowed

        Returns:
            List of card records eligible for retry
        """
        failed_cards = self.get_failed_cards()
        retry_cards = [c for c in failed_cards if c.get("retry_count", 0) < max_retries]

        logger.info("found_cards_needing_retry", count=len(retry_cards))
        return retry_cards
