"""Anki state management component for SyncEngine.

Handles fetching Anki state and determining sync actions.
"""

from ..agents.models import GeneratedCard
from ..anki.client import AnkiClient
from ..config import Config
from ..models import Card, SyncAction
from ..sync.state_db import StateDB
from ..utils.guid import deterministic_guid
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AnkiStateManager:
    """Handles Anki state fetching and action determination."""

    def __init__(
        self,
        config: Config,
        state_db: StateDB,
        anki_client: AnkiClient,
    ):
        """Initialize Anki state manager.

        Args:
            config: Service configuration
            state_db: State database
            anki_client: AnkiConnect client
        """
        self.config = config
        self.db = state_db
        self.anki = anki_client

    def fetch_state(self) -> dict[str, int]:
        """Fetch current Anki state with detailed logging.

        Returns:
            Dict of slug -> anki_note_id
        """
        import time

        start_time = time.time()
        logger.info("fetching_anki_state", deck=self.config.anki_deck_name)

        # Find all notes in target deck
        try:
            from ..exceptions import AnkiConnectError

            note_ids = self.anki.find_notes(f"deck:{self.config.anki_deck_name}")
        except AnkiConnectError as e:
            elapsed = time.time() - start_time
            logger.error(
                "anki_query_failed_connect_error",
                deck=self.config.anki_deck_name,
                error=str(e),
                elapsed_seconds=round(elapsed, 2),
            )
            return {}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "anki_query_failed_unexpected",
                deck=self.config.anki_deck_name,
                error=str(e),
                elapsed_seconds=round(elapsed, 2),
                exc_info=True,
            )
            return {}

        if not note_ids:
            elapsed = time.time() - start_time
            logger.info(
                "anki_state_empty",
                deck=self.config.anki_deck_name,
                elapsed_seconds=round(elapsed, 2),
            )
            return {}

        # Get note info in batches
        batch_size = 100
        anki_cards: dict[str, int] = {}
        total_batches = (len(note_ids) + batch_size - 1) // batch_size

        logger.info(
            "fetching_anki_note_info",
            total_notes=len(note_ids),
            batch_size=batch_size,
            total_batches=total_batches,
        )

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(note_ids))
            batch_note_ids = note_ids[batch_start:batch_end]

            try:
                notes_info = self.anki.notes_info(batch_note_ids)

                for note_info in notes_info:
                    manifest_field = note_info.get("fields", {}).get("Manifest", {})
                    if isinstance(manifest_field, dict):
                        manifest_field = manifest_field.get("value", "")

                    if not manifest_field:
                        continue

                    # Parse manifest
                    manifest = self._parse_manifest_field(manifest_field)
                    if manifest and manifest.slug:
                        note_id = note_info.get("noteId")
                        if note_id:
                            anki_cards[manifest.slug] = note_id

                logger.debug(
                    "anki_batch_fetched",
                    batch=batch_idx + 1,
                    total_batches=total_batches,
                    notes_in_batch=len(batch_note_ids),
                    slugs_found=len(anki_cards),
                )

            except Exception as e:
                logger.error(
                    "anki_batch_fetch_failed",
                    batch=batch_idx + 1,
                    error=str(e),
                    exc_info=True,
                )
                continue

        elapsed = time.time() - start_time
        logger.info(
            "anki_state_fetched",
            deck=self.config.anki_deck_name,
            total_notes=len(note_ids),
            cards_found=len(anki_cards),
            elapsed_seconds=round(elapsed, 2),
        )

        return anki_cards

    def determine_actions(
        self,
        obsidian_cards: dict[str, Card],
        anki_cards: dict[str, int],
        changes: list[SyncAction],
    ) -> None:
        """Determine what actions to take.

        Args:
            obsidian_cards: Cards from Obsidian
            anki_cards: Current Anki state (slug -> note_id)
            changes: List to populate with sync actions
        """
        logger.info("determining_actions")

        # Get database state
        logger.debug("getting_db_cards")
        db_cards = {c["slug"]: c for c in self.db.get_all_cards()}
        logger.debug("got_db_cards", count=len(db_cards))

        # Check each Obsidian card
        for slug, obs_card in obsidian_cards.items():
            db_card = db_cards.get(slug)
            anki_id = anki_cards.get(slug)

            if not db_card and not anki_id:
                # New card - create
                changes.append(
                    SyncAction(
                        type="create",
                        card=obs_card,
                        reason="New card not in database or Anki",
                    )
                )

            elif db_card and obs_card.content_hash != db_card["content_hash"]:
                # Updated card - update
                changes.append(
                    SyncAction(
                        type="update",
                        card=obs_card,
                        anki_guid=db_card["anki_guid"],
                        reason=f"Content changed (old hash: {db_card['content_hash'][:8]}...)",
                    )
                )

            else:
                # No changes - skip
                changes.append(
                    SyncAction(
                        type="skip",
                        card=obs_card,
                        anki_guid=db_card.get("anki_guid") if db_card else None,
                        reason="No changes detected",
                    )
                )

        # Check for deletions in Obsidian
        for slug, db_card in db_cards.items():
            if slug not in obsidian_cards and slug in anki_cards:
                # Card deleted in Obsidian but still in Anki
                from ..models import Card as CardModel
                from ..models import Manifest

                # Reconstruct minimal card for deletion
                card = CardModel(
                    slug=slug,
                    lang=db_card["lang"],
                    apf_html="",
                    manifest=Manifest(
                        slug=slug,
                        slug_base=db_card["slug_base"],
                        lang=db_card["lang"],
                        source_path=db_card["source_path"],
                        source_anchor=db_card["source_anchor"],
                        note_id=db_card["note_id"],
                        note_title=db_card["note_title"],
                        card_index=db_card["card_index"],
                        guid=db_card.get("card_guid")
                        or deterministic_guid(
                            [
                                db_card.get("note_id", ""),
                                db_card["source_path"],
                                str(db_card["card_index"]),
                                db_card["lang"],
                            ]
                        ),
                    ),
                    content_hash=db_card["content_hash"],
                    note_type=db_card.get("note_type", "APF::Simple"),
                    tags=[],
                    guid=db_card.get("card_guid")
                    or deterministic_guid(
                        [
                            db_card.get("note_id", ""),
                            db_card["source_path"],
                            str(db_card["card_index"]),
                            db_card["lang"],
                        ]
                    ),
                )

                changes.append(
                    SyncAction(
                        type="delete",
                        card=card,
                        anki_guid=db_card["anki_guid"],
                        reason="Card removed from Obsidian",
                    )
                )

        # Check for deletions in Anki (restore)
        for slug in db_cards.keys():
            if slug not in anki_cards and slug in obsidian_cards:
                # Card deleted in Anki but still in Obsidian - restore
                changes.append(
                    SyncAction(
                        type="restore",
                        card=obsidian_cards[slug],
                        reason="Card deleted in Anki, restoring from Obsidian",
                    )
                )

        # Count actions
        action_counts: dict[str, int] = {}
        for action in changes:
            action_counts[action.type] = action_counts.get(action.type, 0) + 1

        logger.info("actions_determined", actions=action_counts)

    def _parse_manifest_field(self, manifest_field: str):
        """Parse and validate manifest field from Anki card.

        Args:
            manifest_field: JSON string from Manifest field

        Returns:
            Validated ManifestData or None if invalid
        """
        import json

        from pydantic import ValidationError

        from ..models import ManifestData

        try:
            manifest_dict = json.loads(manifest_field)
        except json.JSONDecodeError as e:
            logger.warning(
                "invalid_manifest_json",
                manifest_field=manifest_field[:100],
                error=str(e),
            )
            return None

        if not isinstance(manifest_dict, dict):
            logger.warning(
                "manifest_not_dict",
                manifest_type=type(manifest_dict).__name__,
            )
            return None

        try:
            manifest = ManifestData(**manifest_dict)
            return manifest
        except ValidationError as e:
            logger.warning(
                "manifest_validation_failed",
                manifest_dict=manifest_dict,
                errors=e.errors(),
            )
            return None

    def fetch_existing_cards_for_duplicate_detection(self) -> list[GeneratedCard]:
        """Fetch existing cards from Anki for duplicate detection.

        Returns:
            List of GeneratedCard instances from Anki
        """
        import time

        start_time = time.time()
        logger.info(
            "fetching_existing_cards_for_duplicate_detection",
            deck=self.config.anki_deck_name,
        )

        try:
            # Find all notes in target deck
            note_ids = self.anki.find_notes(f"deck:{self.config.anki_deck_name}")

            if not note_ids:
                logger.info("no_existing_cards_found", deck=self.config.anki_deck_name)
                return []

            # Get note info for all notes
            notes_info = self.anki.notes_info(note_ids)

            existing_cards: list[GeneratedCard] = []

            for note_info in notes_info:
                try:
                    # Extract fields from note
                    fields = note_info.get("fields", {})

                    # Try to get APF HTML from fields
                    apf_html = ""
                    for field_name, field_data in fields.items():
                        if "apf" in field_name.lower() or "html" in field_name.lower():
                            apf_html = field_data.get("value", "")
                            break

                    if not apf_html:
                        continue

                    # Extract slug from manifest or generate from note ID
                    slug = f"anki-{note_info['noteId']}"

                    # Try to parse manifest from HTML
                    manifest_data = self._extract_manifest_from_html(apf_html)
                    if manifest_data and hasattr(manifest_data, "slug"):
                        slug = manifest_data.slug

                    # Create GeneratedCard
                    card = GeneratedCard(
                        slug=slug,
                        lang="en",  # Default, could be detected
                        apf_html=apf_html,
                        card_index=0,  # Default for existing cards
                        content_hash="",  # Will be computed if needed
                    )

                    existing_cards.append(card)

                except Exception as e:
                    logger.warning(
                        "failed_to_process_existing_card",
                        note_id=note_info.get("noteId"),
                        error=str(e),
                    )
                    continue

            elapsed = time.time() - start_time
            logger.info(
                "fetched_existing_cards_for_duplicate_detection",
                deck=self.config.anki_deck_name,
                card_count=len(existing_cards),
                elapsed_seconds=round(elapsed, 2),
            )

            return existing_cards

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "failed_to_fetch_existing_cards_for_duplicate_detection",
                deck=self.config.anki_deck_name,
                error=str(e),
                elapsed_seconds=round(elapsed, 2),
            )
            return []
