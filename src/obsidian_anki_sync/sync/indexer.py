"""Indexing system for Obsidian vault and Anki cards."""

import json
from collections import defaultdict
from contextlib import nullcontext, suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from obsidian_anki_sync.anki.client import AnkiClient

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.models import ManifestData
from obsidian_anki_sync.obsidian.parser import (
    ParserError,
    discover_notes,
    parse_note,
    parse_note_with_repair,
    temporarily_disable_llm_extraction,
)
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

logger = get_logger(__name__)


class VaultIndexer:
    """Index Obsidian vault notes."""

    def __init__(
        self,
        config: Config,
        db: StateDB,
        archiver: ProblematicNotesArchiver | None = None,
    ):
        """
        Initialize vault indexer.

        Args:
            config: Service configuration
            db: State database
            archiver: Optional problematic notes archiver
        """
        self.config = config
        self.db = db
        self.archiver = archiver or ProblematicNotesArchiver(
            archive_dir=config.problematic_notes_dir,
            enabled=config.enable_problematic_notes_archival,
        )

    def index_vault(self, incremental: bool = False) -> dict:
        """
        Index all notes in the Obsidian vault.

        Args:
            incremental: If True, only index new/modified notes

        Returns:
            Statistics dict
        """
        logger.info(
            "indexing_vault_started",
            path=str(self.config.vault_path),
            incremental=incremental,
        )

        # Use source_subdirs if configured, otherwise use source_dir
        source_dirs = (
            self.config.source_subdirs
            if self.config.source_subdirs
            else self.config.source_dir
        )
        note_files = discover_notes(self.config.vault_path, source_dirs)

        stats = {
            "total_discovered": len(note_files),
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Collect errors for aggregated logging
        error_by_type: defaultdict[str, int] = defaultdict(int)
        error_samples: defaultdict[str, list[str]] = defaultdict(
            list
        )  # Store sample errors per type

        llm_context = (
            temporarily_disable_llm_extraction()
            if not self.config.index_use_llm_extraction
            else nullcontext()
        )

        with llm_context:
            for file_path, relative_path in note_files:
                try:
                    # Check if we should skip (incremental mode)
                    if incremental:
                        existing_index = self.db.get_note_index(relative_path)
                        if existing_index:
                            # Check file modification time
                            file_mtime = datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            )
                            indexed_mtime = (
                                datetime.fromisoformat(
                                    existing_index["file_modified_at"]
                                )
                                if existing_index["file_modified_at"]
                                else None
                            )

                            if indexed_mtime and file_mtime <= indexed_mtime:
                                # File hasn't changed, skip
                                stats["skipped"] += 1
                                logger.debug(
                                    "skipping_unchanged_note", path=relative_path
                                )
                                continue

                    # Parse the note with repair if enabled
                    repair_enabled = getattr(self.config, "parser_repair_enabled", True)
                    if repair_enabled:
                        # Try to get LLM provider for repair
                        llm_provider_for_repair = None
                        try:
                            from obsidian_anki_sync.providers.factory import (
                                ProviderFactory,
                            )

                            llm_provider_for_repair = (
                                ProviderFactory.create_from_config(self.config)
                            )
                        except Exception as e:
                            logger.debug(
                                "repair_provider_unavailable",
                                error=str(e),
                                note=str(relative_path),
                            )
                            # Continue without repair provider

                        repair_model = self.config.get_model_for_agent("parser_repair")
                        tolerant_parsing = getattr(
                            self.config, "tolerant_parsing", True
                        )
                        enable_content_generation = getattr(
                            self.config, "enable_content_generation", True
                        )
                        repair_missing_sections = getattr(
                            self.config, "repair_missing_sections", True
                        )

                        metadata, qa_pairs = parse_note_with_repair(
                            file_path=file_path,
                            ollama_client=llm_provider_for_repair,
                            repair_model=repair_model,
                            enable_repair=repair_enabled
                            and llm_provider_for_repair is not None,
                            tolerant_parsing=tolerant_parsing,
                            enable_content_generation=enable_content_generation,
                            repair_missing_sections=repair_missing_sections,
                        )
                    else:
                        # Parse without repair
                        metadata, qa_pairs = parse_note(file_path)

                    # Get file modification time
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    # Serialize metadata for storage
                    metadata_json = json.dumps(
                        {
                            "id": metadata.id,
                            "title": metadata.title,
                            "topic": metadata.topic,
                            "language_tags": metadata.language_tags,
                            "created": metadata.created.isoformat(),
                            "updated": metadata.updated.isoformat(),
                            "aliases": metadata.aliases,
                            "subtopics": metadata.subtopics,
                            "question_kind": metadata.question_kind,
                            "difficulty": metadata.difficulty,
                            "status": metadata.status,
                        }
                    )

                    # Insert/update note index
                    self.db.upsert_note_index(
                        source_path=relative_path,
                        note_id=metadata.id,
                        note_title=metadata.title,
                        topic=metadata.topic,
                        language_tags=metadata.language_tags,
                        qa_pair_count=len(qa_pairs),
                        file_modified_at=file_mtime,
                        metadata_json=metadata_json,
                    )

                    # Index expected cards for this note
                    for qa_pair in qa_pairs:
                        for lang in metadata.language_tags:
                            self.db.upsert_card_index(
                                source_path=relative_path,
                                card_index=qa_pair.card_index,
                                lang=lang,
                                note_id=metadata.id,
                                note_title=metadata.title,
                                status="expected",
                                in_obsidian=True,
                                in_anki=False,
                                in_database=False,
                            )

                    stats["indexed"] += 1
                    logger.debug(
                        "note_indexed",
                        path=relative_path,
                        qa_pairs=len(qa_pairs),
                        languages=len(metadata.language_tags),
                    )

                except (ParserError, OSError, Exception) as e:
                    error_type_name = type(e).__name__
                    error_message = str(e)

                    # Archive problematic note
                    try:
                        note_content = ""
                        with suppress(Exception):
                            note_content = file_path.read_text(encoding="utf-8")

                        self.archiver.archive_note(
                            note_path=file_path,
                            error=e,
                            error_type=error_type_name,
                            processing_stage="indexing",
                            note_content=note_content if note_content else None,
                            context={
                                "relative_path": relative_path,
                            },
                        )
                    except Exception as archive_error:
                        logger.warning(
                            "failed_to_archive_problematic_note",
                            note_path=str(file_path),
                            archive_error=str(archive_error),
                        )

                    # Aggregate errors with defaultdict
                    error_by_type[error_type_name] += 1

                    # Store sample errors (up to 3 per type)
                    if len(error_samples[error_type_name]) < 3:
                        error_samples[error_type_name].append(
                            f"{relative_path}: {error_message}"
                        )

                    stats["errors"] += 1

        # Log aggregated error summary
        if error_by_type:
            logger.warning(
                "indexing_errors_summary",
                total_errors=stats["errors"],
                error_breakdown=error_by_type,
            )
            # Log sample errors for each type
            for err_type, samples in error_samples.items():
                for i, sample in enumerate(samples):
                    logger.warning(
                        "error_sample",
                        error_type=err_type,
                        sample_num=i + 1,
                        error=sample,
                    )

        logger.info("indexing_vault_completed", stats=stats)
        return stats


class AnkiIndexer:
    """Index Anki cards."""

    def __init__(self, db: StateDB, anki_client: "AnkiClient"):
        """
        Initialize Anki indexer.

        Args:
            db: State database
            anki_client: AnkiConnect client
        """
        self.db = db
        self.anki = anki_client

    def _parse_manifest_field(self, manifest_field: str) -> ManifestData | None:
        """Parse and validate manifest field from Anki card.

        Args:
            manifest_field: JSON string from Manifest field

        Returns:
            Validated ManifestData or None if invalid
        """
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

    def index_anki_cards(self, deck_name: str) -> dict:
        """
        Index all cards in Anki deck.

        Args:
            deck_name: Anki deck name to index

        Returns:
            Statistics dict
        """
        logger.info("indexing_anki_started", deck=deck_name)

        stats = {
            "total_discovered": 0,
            "indexed": 0,
            "matched": 0,
            "unmatched": 0,
            "errors": 0,
        }

        try:
            # Find all notes in deck
            note_ids = self.anki.find_notes(f"deck:{deck_name}")
            stats["total_discovered"] = len(note_ids)

            if not note_ids:
                logger.info("no_anki_notes_found")
                return stats

            # Get note info
            notes_info = self.anki.notes_info(note_ids)

            for note_info in notes_info:
                # Extract manifest from note
                fields = note_info.get("fields", {})
                manifest_field = fields.get("Manifest", {}).get("value", "{}")

                # Parse and validate manifest, keep dict for optional fields
                try:
                    manifest_dict = json.loads(manifest_field)
                except json.JSONDecodeError:
                    manifest_dict = {}

                manifest = self._parse_manifest_field(manifest_field)
                if manifest is None:
                    logger.warning(
                        "skipping_card_invalid_manifest",
                        note_id=note_info.get("noteId"),
                    )
                    stats["unmatched"] += 1
                    continue

                # Extract validated fields
                slug = manifest.slug
                source_path = manifest.source_path
                card_index = manifest.card_index
                lang = manifest.lang

                try:
                    # Update card index with Anki information
                    existing_card = self.db.get_card_index_by_slug(slug)

                    # Get optional fields from raw manifest dict for backward compatibility
                    note_id = manifest_dict.get("note_id")
                    note_title = manifest_dict.get("note_title")

                    if existing_card:
                        # Update existing index entry
                        self.db.upsert_card_index(
                            source_path=source_path,
                            card_index=card_index,
                            lang=lang,
                            slug=slug,
                            anki_guid=note_info["noteId"],
                            note_id=note_id,
                            note_title=note_title,
                            in_obsidian=existing_card.get("in_obsidian", False),
                            in_anki=True,
                            in_database=True,
                        )
                        stats["matched"] += 1
                    else:
                        # Card exists in Anki but not in our index (orphaned)
                        self.db.upsert_card_index(
                            source_path=source_path,
                            card_index=card_index,
                            lang=lang,
                            slug=slug,
                            anki_guid=note_info["noteId"],
                            note_id=note_id,
                            note_title=note_title,
                            status="orphaned",
                            in_obsidian=False,
                            in_anki=True,
                            in_database=True,
                        )
                        stats["unmatched"] += 1

                    stats["indexed"] += 1

                except Exception as e:
                    logger.error(
                        "card_indexing_failed",
                        note_id=note_info.get("noteId"),
                        error=str(e),
                    )
                    stats["errors"] += 1

        except Exception as e:
            logger.error("anki_indexing_failed", error=str(e))
            stats["errors"] += 1

        logger.info("indexing_anki_completed", stats=stats)
        return stats


class SyncIndexer:
    """Index sync database state."""

    def __init__(self, db: StateDB):
        """
        Initialize sync indexer.

        Args:
            db: State database
        """
        self.db = db

    def index_database_cards(self) -> dict:
        """
        Index cards from sync database.

        Returns:
            Statistics dict
        """
        logger.info("indexing_database_started")

        stats = {
            "total_discovered": 0,
            "updated": 0,
            "errors": 0,
        }

        try:
            # Get all cards from database
            db_cards = self.db.get_all_cards()
            stats["total_discovered"] = len(db_cards)

            for db_card in db_cards:
                try:
                    slug = db_card["slug"]

                    # Update card index with database information
                    existing_card = self.db.get_card_index_by_slug(slug)

                    if existing_card:
                        # Update with database info
                        self.db.upsert_card_index(
                            source_path=db_card["source_path"],
                            card_index=db_card["card_index"],
                            lang=db_card["lang"],
                            slug=slug,
                            anki_guid=db_card["anki_guid"],
                            note_id=db_card.get("note_id"),
                            note_title=db_card.get("note_title"),
                            content_hash=db_card["content_hash"],
                            in_obsidian=existing_card.get("in_obsidian", False),
                            in_anki=existing_card.get("in_anki", False),
                            in_database=True,
                        )
                    else:
                        # Card in database but not indexed yet
                        self.db.upsert_card_index(
                            source_path=db_card["source_path"],
                            card_index=db_card["card_index"],
                            lang=db_card["lang"],
                            slug=slug,
                            anki_guid=db_card["anki_guid"],
                            note_id=db_card.get("note_id"),
                            note_title=db_card.get("note_title"),
                            content_hash=db_card["content_hash"],
                            status="synced",
                            in_obsidian=False,
                            in_anki=False,
                            in_database=True,
                        )

                    stats["updated"] += 1

                except Exception as e:
                    logger.error(
                        "database_card_indexing_failed",
                        slug=db_card.get("slug"),
                        error=str(e),
                    )
                    stats["errors"] += 1

        except Exception as e:
            logger.error("database_indexing_failed", error=str(e))
            stats["errors"] += 1

        logger.info("indexing_database_completed", stats=stats)
        return stats


def build_full_index(
    config: Config, db: StateDB, anki_client: "AnkiClient", incremental: bool = False
) -> dict[str, Any]:
    """
    Build full index of vault, Anki, and database.

    Args:
        config: Service configuration
        db: State database
        anki_client: AnkiConnect client
        incremental: If True, only index changed items

    Returns:
        Combined statistics dict
    """
    logger.info("building_full_index", incremental=incremental)

    combined_stats: dict[str, Any] = {
        "vault": {},
        "anki": {},
        "database": {},
        "overall": {},
    }

    # Index vault notes
    # Create archiver for problematic notes
    from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

    archiver = ProblematicNotesArchiver(
        archive_dir=config.problematic_notes_dir,
        enabled=config.enable_problematic_notes_archival,
    )
    vault_indexer = VaultIndexer(config, db, archiver=archiver)
    combined_stats["vault"] = vault_indexer.index_vault(incremental=incremental)

    # Index database cards
    sync_indexer = SyncIndexer(db)
    combined_stats["database"] = sync_indexer.index_database_cards()

    # Index Anki cards
    anki_indexer = AnkiIndexer(db, anki_client)
    combined_stats["anki"] = anki_indexer.index_anki_cards(config.anki_deck_name)

    # Get overall statistics
    combined_stats["overall"] = db.get_index_statistics()

    logger.info("full_index_built", stats=combined_stats["overall"])

    return combined_stats
