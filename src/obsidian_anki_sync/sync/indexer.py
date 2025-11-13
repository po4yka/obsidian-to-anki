"""Indexing system for Obsidian vault and Anki cards."""

import json
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..anki.client import AnkiClient

from ..config import Config
from ..obsidian.parser import ParserError, discover_notes, parse_note
from ..sync.state_db import StateDB
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VaultIndexer:
    """Index Obsidian vault notes."""

    def __init__(self, config: Config, db: StateDB):
        """
        Initialize vault indexer.

        Args:
            config: Service configuration
            db: State database
        """
        self.config = config
        self.db = db

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

        note_files = discover_notes(self.config.vault_path, self.config.source_dir)

        stats = {
            "total_discovered": len(note_files),
            "indexed": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Collect topic mismatches for aggregated logging
        topic_mismatches: dict[str, int] = {}

        # Collect errors for aggregated logging
        error_by_type: defaultdict[str, int] = defaultdict(int)
        error_samples: defaultdict[str, list[str]] = defaultdict(
            list
        )  # Store sample errors per type

        for file_path, relative_path in note_files:
            try:
                # Check if we should skip (incremental mode)
                if incremental:
                    existing_index = self.db.get_note_index(relative_path)
                    if existing_index:
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        indexed_mtime = (
                            datetime.fromisoformat(existing_index["file_modified_at"])
                            if existing_index["file_modified_at"]
                            else None
                        )

                        if indexed_mtime and file_mtime <= indexed_mtime:
                            # File hasn't changed, skip
                            stats["skipped"] += 1
                            logger.debug("skipping_unchanged_note", path=relative_path)
                            continue

                # Parse the note
                metadata, qa_pairs = parse_note(file_path)

                # Check for topic mismatch and collect for summary
                expected_topic = file_path.parent.name
                if metadata.topic != expected_topic:
                    key = f"{expected_topic} -> {metadata.topic}"
                    topic_mismatches[key] = topic_mismatches.get(key, 0) + 1

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

        # Log aggregated topic mismatch summary
        if topic_mismatches:
            total_mismatches = sum(topic_mismatches.values())
            logger.info(
                "topic_mismatches_summary",
                total=total_mismatches,
                patterns=topic_mismatches,
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
                try:
                    # Extract manifest from note
                    fields = note_info.get("fields", {})
                    manifest_field = fields.get("Manifest", {}).get("value", "{}")

                    manifest = json.loads(manifest_field)
                    slug = manifest.get("slug")
                    source_path = manifest.get("source_path")
                    card_index = manifest.get("card_index")
                    lang = manifest.get("lang")

                    if not all([slug, source_path, card_index, lang]):
                        logger.warning(
                            "incomplete_manifest",
                            note_id=note_info.get("noteId"),
                            slug=slug,
                        )
                        stats["unmatched"] += 1
                        continue

                    # Update card index with Anki information
                    existing_card = self.db.get_card_index_by_slug(slug)

                    if existing_card:
                        # Update existing index entry
                        self.db.upsert_card_index(
                            source_path=source_path,
                            card_index=card_index,
                            lang=lang,
                            slug=slug,
                            anki_guid=note_info["noteId"],
                            note_id=manifest.get("note_id"),
                            note_title=manifest.get("note_title"),
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
                            note_id=manifest.get("note_id"),
                            note_title=manifest.get("note_title"),
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
    vault_indexer = VaultIndexer(config, db)
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
