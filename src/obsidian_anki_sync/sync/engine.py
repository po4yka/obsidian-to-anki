"""Synchronization engine for Obsidian to Anki sync."""

import random

import yaml  # type: ignore

from ..anki.client import AnkiClient
from ..anki.field_mapper import map_apf_to_anki_fields
from ..apf.generator import APFGenerator
from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..config import Config
from ..models import Card, NoteMetadata, QAPair, SyncAction
from ..obsidian.parser import discover_notes, parse_note, ParserError
from ..sync.slug_generator import create_manifest, generate_slug
from ..sync.state_db import StateDB
from ..utils.logging import get_logger
from ..utils.guid import deterministic_guid

logger = get_logger(__name__)


class SyncEngine:
    """Orchestrate synchronization between Obsidian and Anki."""

    def __init__(self, config: Config, state_db: StateDB, anki_client: AnkiClient):
        """
        Initialize sync engine.

        Args:
            config: Service configuration
            state_db: State database
            anki_client: AnkiConnect client
        """
        self.config = config
        self.db = state_db
        self.anki = anki_client
        self.apf_gen = APFGenerator(config)
        self.changes: list[SyncAction] = []
        self.stats = {
            'processed': 0,
            'created': 0,
            'updated': 0,
            'deleted': 0,
            'restored': 0,
            'skipped': 0,
            'errors': 0,
        }

    def sync(self, dry_run: bool = False, sample_size: int | None = None) -> dict:
        """
        Perform synchronization.

        Args:
            dry_run: If True, preview changes without applying
            sample_size: Optional number of notes to randomly sample

        Returns:
            Statistics dict
        """
        logger.info("sync_started", dry_run=dry_run, sample_size=sample_size)

        try:
            # Step 1: Scan Obsidian notes and generate cards
            obsidian_cards = self._scan_obsidian_notes(sample_size=sample_size)

            # Step 2: Fetch Anki state
            anki_cards = self._fetch_anki_state()

            # Step 3: Determine sync actions
            self._determine_actions(obsidian_cards, anki_cards)

            # Step 4: Apply or preview
            if dry_run:
                self._print_plan()
            else:
                self._apply_changes()

            logger.info("sync_completed", stats=self.stats)
            return self.stats

        except Exception as e:
            logger.error("sync_failed", error=str(e))
            raise

    def _scan_obsidian_notes(self, sample_size: int | None = None) -> dict[str, Card]:
        """
        Scan Obsidian vault and generate cards.

        Args:
            sample_size: Optional number of notes to randomly process

        Returns:
            Dict of slug -> Card
        """
        logger.info(
            "scanning_obsidian",
            path=str(self.config.vault_path),
            sample_size=sample_size
        )

        note_files = discover_notes(
            self.config.vault_path, self.config.source_dir
        )

        if sample_size and sample_size > 0 and len(note_files) > sample_size:
            note_files = random.sample(note_files, sample_size)
            logger.info("sampling_notes", count=sample_size)

        obsidian_cards: dict[str, Card] = {}
        existing_slugs: set[str] = set()

        for file_path, relative_path in note_files:
            try:
                # Parse note
                metadata, qa_pairs = parse_note(file_path)
            except (
                ParserError,
                yaml.YAMLError,
                OSError,
                UnicodeDecodeError,
            ) as e:
                logger.error(
                    "note_parsing_failed",
                    file=relative_path,
                    error=str(e),
                    error_type=type(e).__name__
                )
                self.stats['errors'] += 1
                continue
            except Exception:
                logger.exception(
                    "unexpected_parsing_error",
                    file=relative_path
                )
                self.stats['errors'] += 1
                continue

            try:

                logger.debug(
                    "processing_note",
                    file=relative_path,
                    pairs=len(qa_pairs)
                )

                # Generate cards for each Q/A pair and language
                for qa_pair in qa_pairs:
                    for lang in metadata.language_tags:
                        try:
                            card = self._generate_card(
                                qa_pair, metadata, relative_path,
                                lang, existing_slugs
                            )
                            obsidian_cards[card.slug] = card
                            existing_slugs.add(card.slug)

                        except Exception as e:
                            logger.error(
                                "card_generation_failed",
                                file=relative_path,
                                pair=qa_pair.card_index,
                                lang=lang,
                                error=str(e)
                            )
                            self.stats['errors'] += 1

                self.stats['processed'] += 1

            except Exception:
                logger.exception(
                    "card_generation_failed",
                    file=relative_path
                )
                self.stats['errors'] += 1

        logger.info(
            "obsidian_scan_completed",
            notes=len(note_files),
            cards=len(obsidian_cards)
        )

        return obsidian_cards

    def _generate_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        relative_path: str,
        lang: str,
        existing_slugs: set[str]
    ) -> Card:
        """Generate a single card."""
        # Generate slug
        slug, slug_base, hash6 = generate_slug(
            relative_path,
            qa_pair.card_index,
            lang,
            existing_slugs
        )

        # Compute deterministic GUID for the note
        guid = deterministic_guid([
            metadata.id,
            relative_path,
            str(qa_pair.card_index),
            lang
        ])

        # Create manifest
        manifest = create_manifest(
            slug, slug_base, lang, relative_path,
            qa_pair.card_index, metadata, guid, hash6
        )

        # Generate APF card via LLM
        card = self.apf_gen.generate_card(qa_pair, metadata, manifest, lang)

        # Validate APF format
        validation = validate_apf(card.apf_html, slug)
        if validation.errors:
            logger.warning(
                "apf_validation_errors",
                slug=slug,
                errors=validation.errors
            )
        if validation.warnings:
            logger.debug(
                "apf_validation_warnings",
                slug=slug,
                warnings=validation.warnings
            )

        html_errors = validate_card_html(card.apf_html)
        if html_errors:
            logger.error(
                "apf_html_invalid",
                slug=slug,
                errors=html_errors
            )
            raise ValueError(f"Invalid HTML formatting for {slug}: {html_errors[0]}")

        return card

    def _fetch_anki_state(self) -> dict[str, int]:
        """
        Fetch current Anki state.

        Returns:
            Dict of slug -> anki_note_id
        """
        logger.info("fetching_anki_state", deck=self.config.anki_deck_name)

        # Find all notes in target deck
        try:
            note_ids = self.anki.find_notes(f"deck:{self.config.anki_deck_name}")
        except Exception as e:
            logger.error("anki_query_failed", error=str(e))
            return {}

        if not note_ids:
            logger.info("no_anki_notes_found")
            return {}

        # Get note info (batch fetch to avoid N+1)
        try:
            notes_info = self.anki.notes_info(note_ids)
        except Exception as e:
            logger.error("anki_notes_info_failed", error=str(e))
            return {}

        # Extract slugs from manifests
        anki_cards = {}
        for note_info in notes_info:
            try:
                # Look for Manifest field
                fields = note_info.get('fields', {})
                manifest_field = fields.get('Manifest', {}).get('value', '{}')

                import json
                manifest = json.loads(manifest_field)
                slug = manifest.get('slug')

                if slug:
                    anki_cards[slug] = note_info['noteId']

            except Exception as e:
                logger.warning(
                    "failed_to_parse_manifest",
                    note_id=note_info.get('noteId'),
                    error=str(e)
                )

        logger.info("anki_state_fetched", count=len(anki_cards))
        return anki_cards

    def _determine_actions(
        self,
        obsidian_cards: dict[str, Card],
        anki_cards: dict[str, int]
    ) -> None:
        """
        Determine what actions to take.

        Args:
            obsidian_cards: Cards from Obsidian
            anki_cards: Current Anki state (slug -> note_id)
        """
        logger.info("determining_actions")

        # Get database state
        db_cards = {c['slug']: c for c in self.db.get_all_cards()}

        # Check each Obsidian card
        for slug, obs_card in obsidian_cards.items():
            db_card = db_cards.get(slug)
            anki_id = anki_cards.get(slug)

            if not db_card and not anki_id:
                # New card - create
                self.changes.append(SyncAction(
                    type='create',
                    card=obs_card,
                    reason="New card not in database or Anki"
                ))

            elif db_card and obs_card.content_hash != db_card['content_hash']:
                # Updated card - update
                self.changes.append(SyncAction(
                    type='update',
                    card=obs_card,
                    anki_guid=db_card['anki_guid'],
                    reason=f"Content changed (old hash: {db_card['content_hash'][:8]}...)"
                ))

            else:
                # No changes - skip
                self.changes.append(SyncAction(
                    type='skip',
                    card=obs_card,
                    anki_guid=db_card.get('anki_guid') if db_card else None,
                    reason="No changes detected"
                ))

        # Check for deletions in Obsidian
        for slug, db_card in db_cards.items():
            if slug not in obsidian_cards and slug in anki_cards:
                # Card deleted in Obsidian but still in Anki
                from ..models import Card as CardModel, Manifest

                # Reconstruct minimal card for deletion
                card = CardModel(
                    slug=slug,
                    lang=db_card['lang'],
                    apf_html="",
                    manifest=Manifest(
                        slug=slug,
                        slug_base=db_card['slug_base'],
                        lang=db_card['lang'],
                        source_path=db_card['source_path'],
                        source_anchor=db_card['source_anchor'],
                        note_id=db_card['note_id'],
                        note_title=db_card['note_title'],
                        card_index=db_card['card_index'],
                        guid=db_card.get('card_guid') or deterministic_guid([
                            db_card.get('note_id', ''),
                            db_card['source_path'],
                            str(db_card['card_index']),
                            db_card['lang']
                        ]),
                    ),
                    content_hash=db_card['content_hash'],
                    note_type=db_card.get('note_type', 'APF::Simple'),
                    tags=[],
                    guid=db_card.get('card_guid') or deterministic_guid([
                        db_card.get('note_id', ''),
                        db_card['source_path'],
                        str(db_card['card_index']),
                        db_card['lang']
                    ]),
                )

                self.changes.append(SyncAction(
                    type='delete',
                    card=card,
                    anki_guid=db_card['anki_guid'],
                    reason="Card removed from Obsidian"
                ))

        # Check for deletions in Anki (restore)
        for slug in db_cards.keys():
            if slug not in anki_cards and slug in obsidian_cards:
                # Card deleted in Anki but still in Obsidian - restore
                self.changes.append(SyncAction(
                    type='restore',
                    card=obsidian_cards[slug],
                    reason="Card deleted in Anki, restoring from Obsidian"
                ))

        # Count actions
        action_counts: dict[str, int] = {}
        for action in self.changes:
            action_counts[action.type] = (
                action_counts.get(action.type, 0) + 1
            )

        logger.info("actions_determined", actions=action_counts)

    def _print_plan(self) -> None:
        """Print sync plan for dry-run."""
        print("\n=== Sync Plan (Dry Run) ===\n")

        for action in self.changes:
            if action.type == 'skip':
                continue

            print(f"[{action.type.upper()}] {action.card.slug}")
            if action.reason:
                print(f"  Reason: {action.reason}")
            print()

        # Print summary
        action_counts: dict[str, int] = {}
        for action in self.changes:
            action_counts[action.type] = (
                action_counts.get(action.type, 0) + 1
            )

        print("=== Summary ===")
        for action_type, count in sorted(action_counts.items()):
            print(f"{action_type}: {count}")
        print()

    def _apply_changes(self) -> None:
        """Apply sync actions to Anki."""
        logger.info("applying_changes", count=len(self.changes))

        for action in self.changes:
            try:
                if action.type == 'create':
                    self._create_card(action.card)
                    self.stats['created'] += 1

                elif action.type == 'update':
                    if action.anki_guid:
                        self._update_card(action.card, action.anki_guid)
                        self.stats['updated'] += 1

                elif action.type == 'delete':
                    if action.anki_guid:
                        self._delete_card(action.card, action.anki_guid)
                        self.stats['deleted'] += 1

                elif action.type == 'restore':
                    self._create_card(action.card)
                    self.stats['restored'] += 1

                elif action.type == 'skip':
                    self.stats['skipped'] += 1

            except Exception as e:
                logger.error(
                    "action_failed",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e)
                )
                self.stats['errors'] += 1

    def _create_card(self, card: Card) -> None:
        """Create card in Anki."""
        logger.info("creating_card", slug=card.slug)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        # Add note to Anki
        note_id = self.anki.add_note(
            deck=self.config.anki_deck_name,
            note_type=card.note_type,
            fields=fields,
            tags=card.tags,
            guid=card.guid
        )

        # Save to database
        self.db.insert_card(card, anki_guid=note_id)

    def _update_card(self, card: Card, anki_guid: int) -> None:
        """Update card in Anki."""
        logger.info("updating_card", slug=card.slug, anki_guid=anki_guid)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        # Update note in Anki
        self.anki.update_note_fields(anki_guid, fields)
        self.anki.update_note_tags(anki_guid, card.tags)

        # Update database
        self.db.update_card(card)

    def _delete_card(self, card: Card, anki_guid: int) -> None:
        """Delete card from Anki."""
        logger.info("deleting_card", slug=card.slug, anki_guid=anki_guid)

        if self.config.delete_mode == 'delete':
            # Actually delete from Anki
            self.anki.delete_notes([anki_guid])
        # else: archive mode - just remove from database

        # Remove from database
        self.db.delete_card(card.slug)
