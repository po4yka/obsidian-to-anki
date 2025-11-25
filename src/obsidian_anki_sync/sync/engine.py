"""Synchronization engine for Obsidian to Anki sync."""

import contextlib
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import diskcache
import yaml  # type: ignore
from pydantic import ValidationError

from ..anki.client import AnkiClient
from ..anki.field_mapper import map_apf_to_anki_fields
from ..apf.generator import APFGenerator
from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..config import Config
from ..exceptions import AnkiConnectError
from ..models import Card, ManifestData, NoteMetadata, QAPair, SyncAction
from ..obsidian.parser import (
    ParserError,
    create_qa_extractor,
    discover_notes,
    parse_note,
    parse_note_with_repair,
)
from ..sync.indexer import build_full_index
from ..sync.slug_generator import create_manifest, generate_slug
from ..sync.state_db import StateDB
from ..sync.transactions import CardOperationError, CardTransaction
from ..utils.content_hash import compute_content_hash
from ..utils.guid import deterministic_guid
from ..utils.logging import get_logger
from ..utils.problematic_notes import ProblematicNotesArchiver

if TYPE_CHECKING:
    from ..sync.progress import ProgressTracker

logger = get_logger(__name__)

# Import agent orchestrator (optional dependency)
try:
    from ..agents.orchestrator import AgentOrchestrator

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    AgentOrchestrator = None  # type: ignore
    logger.warning("agent_system_not_available", reason="Import failed")

# Import progress display (optional)
try:
    from ..utils.progress_display import ProgressDisplay

    PROGRESS_DISPLAY_AVAILABLE = True
except ImportError:
    PROGRESS_DISPLAY_AVAILABLE = False
    ProgressDisplay = None  # type: ignore


class SyncEngine:
    """Orchestrate synchronization between Obsidian and Anki."""

    def __init__(
        self,
        config: Config,
        state_db: StateDB,
        anki_client: AnkiClient,
        progress_tracker: "ProgressTracker | None" = None,
    ):
        """
        Initialize sync engine.

        Args:
            config: Service configuration
            state_db: State database
            anki_client: AnkiConnect client
            progress_tracker: Optional progress tracker for resumable syncs
        """
        self.config = config
        self.db = state_db
        self.anki = anki_client
        self.progress = progress_tracker
        self.progress_display = None  # Will be set via set_progress_display()

        # Initialize problematic notes archiver
        self.archiver = ProblematicNotesArchiver(
            archive_dir=config.problematic_notes_dir,
            enabled=config.enable_problematic_notes_archival,
        )

        # Log configuration at startup
        logger.info(
            "sync_engine_configuration",
            llm_provider=config.llm_provider,
            llm_timeout=config.llm_timeout,
            use_agent_system=config.use_agent_system,
            vault_path=str(config.vault_path),
            anki_deck=config.anki_deck_name,
            run_mode=config.run_mode,
            delete_mode=config.delete_mode,
        )

        # Initialize card generator (APFGenerator or AgentOrchestrator)
        if config.use_agent_system:
            if not AGENTS_AVAILABLE:
                raise RuntimeError(
                    "Agent system requested but not available. "
                    "Please ensure agent dependencies are installed."
                )
            logger.info("initializing_agent_orchestrator")
            self.agent_orchestrator: AgentOrchestrator | None = AgentOrchestrator(
                config
            )  # type: ignore
            # Still keep for backward compat
            self.apf_gen = APFGenerator(config)
            self.use_agents = True

            # Configure LLM-based Q&A extraction when using agents
            # Use the same provider as the orchestrator
            # Resolve model from config (handles empty strings and presets)
            qa_extractor_model = config.get_model_for_agent("qa_extractor")
            qa_extractor_temp = getattr(config, "qa_extractor_temperature", None)
            if qa_extractor_temp is None:
                # Get temperature from model config if not explicitly set
                model_config = config.get_model_config_for_task("qa_extraction")
                qa_extractor_temp = model_config.get("temperature", 0.0)
            reasoning_enabled = getattr(config, "llm_reasoning_enabled", False)

            logger.info(
                "configuring_llm_qa_extraction",
                model=qa_extractor_model,
                temperature=qa_extractor_temp,
                reasoning_enabled=reasoning_enabled,
            )
            # Create QA extractor agent for LLM-based extraction
            self.qa_extractor = create_qa_extractor(
                llm_provider=self.agent_orchestrator.provider,
                model=qa_extractor_model,
                temperature=qa_extractor_temp,
                reasoning_enabled=reasoning_enabled,
                enable_content_generation=True,
                repair_missing_sections=getattr(
                    config, "enforce_bilingual_validation", True
                ),
            )
        else:
            self.apf_gen = APFGenerator(config)
            self.agent_orchestrator: AgentOrchestrator | None = None  # type: ignore
            self.use_agents = False
            # No LLM extraction when not using agents
            self.qa_extractor = None

        self.changes: list[SyncAction] = []
        self.stats = {
            # Error metrics
            "parser_warnings": 0,
            "llm_truncations": 0,
            "validation_errors": 0,
            "auto_fix_attempts": 0,
            "auto_fix_successes": 0,
            "processed": 0,
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "restored": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Thread-safe slug generation
        import threading

        self._slug_counter_lock = threading.Lock()
        self._slug_counters: dict[str, int] = {}  # base_slug -> next_index

        # Initialize persistent disk caches
        # Cache directory is placed next to the database file
        cache_base_dir = config.db_path.parent / ".cache"
        cache_base_dir.mkdir(parents=True, exist_ok=True)

        # Cache for agent-generated cards (cache_key -> list of Card)
        # Cache key format: f"{metadata.id}:{relative_path}:{content_hash}"
        agent_cache_dir = cache_base_dir / "agent_cards"
        self._agent_card_cache = diskcache.Cache(
            directory=str(agent_cache_dir),
            size_limit=2**30,  # 1GB limit
            eviction_policy="least-recently-used",
        )

        # Cache for non-agent generated cards (cache_key -> Card)
        # Cache key format: f"{relative_path}:{qa_pair.card_index}:{lang}:{content_hash}"
        apf_cache_dir = cache_base_dir / "apf_cards"
        self._apf_card_cache = diskcache.Cache(
            directory=str(apf_cache_dir),
            size_limit=2**30,  # 1GB limit
            eviction_policy="least-recently-used",
        )

        self._cache_hits = 0
        self._cache_misses = 0

        # Cache statistics tracking
        self._cache_stats = {"hits": 0, "misses": 0, "generation_times": []}

        logger.info(
            "persistent_cache_initialized",
            agent_cache_dir=str(agent_cache_dir),
            apf_cache_dir=str(apf_cache_dir),
            cache_size_limit_gb=1,
        )

    def set_progress_display(self, progress_display: "ProgressDisplay | None") -> None:
        """Set progress display for real-time updates.

        Args:
            progress_display: ProgressDisplay instance or None
        """
        self.progress_display = progress_display
        # Pass to agents if available
        if (
            self.progress_display
            and hasattr(self, "agent_orchestrator")
            and self.agent_orchestrator
        ):
            if hasattr(self.agent_orchestrator, "set_progress_display"):
                self.agent_orchestrator.set_progress_display(self.progress_display)

    def _close_caches(self) -> None:
        """Close disk caches to ensure data is flushed to disk."""
        try:
            self._agent_card_cache.close()
            self._apf_card_cache.close()
            logger.debug("caches_closed")
        except Exception as e:
            logger.warning("cache_close_error", error=str(e))
            # Non-critical, continue

    def __enter__(self) -> "SyncEngine":
        """Context manager entry for guaranteed cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure caches are closed."""
        self._close_caches()
        # Reset cache stats for next sync
        self._cache_stats = {"hits": 0, "misses": 0, "generation_times": []}

    def _archive_note_safely(
        self,
        file_path: Path,
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        context: dict[str, Any] | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Safely archive a problematic note with consistent error handling.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred (parsing, card_generation, processing)
            note_content: Optional note content (will be read if not provided)
            context: Optional additional context for archiving
            card_index: Optional card index if error occurred during card generation
            language: Optional language if error occurred during card generation
        """
        try:
            # Read content if not provided
            if note_content is None:
                try:
                    note_content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as read_err:
                    logger.debug(
                        "unable_to_read_note_for_archiving",
                        file=relative_path,
                        error=str(read_err),
                    )
                    note_content = ""

            # Build context dict
            archive_context = {"relative_path": relative_path}
            if context:
                archive_context.update(context)

            self.archiver.archive_note(
                note_path=file_path,
                error=error,
                error_type=type(error).__name__,
                processing_stage=processing_stage,
                card_index=card_index,
                language=language,
                note_content=note_content if note_content else None,
                context=archive_context,
            )
        except Exception as archive_error:
            logger.warning(
                "failed_to_archive_problematic_note",
                note_path=str(file_path),
                archive_error=str(archive_error),
            )

    def __del__(self) -> None:
        """Cleanup caches on destruction."""
        try:
            self._close_caches()
        except OSError as e:
            logger.warning("cache_close_failed_on_destruction", error=str(e))
        except Exception as e:
            logger.error(
                "unexpected_cache_close_error_on_destruction",
                error=str(e),
                exc_info=True,
            )

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

    @contextlib.contextmanager
    def _get_progress_bar(self, total: int, description: str = "Processing"):
        """Yield progress bar context manager.

        Args:
            total: Total number of items
            description: Progress bar description

        Yields:
            Tuple of (progress_bar, task_id) or None if progress display unavailable
        """
        if not self.progress_display or not PROGRESS_DISPLAY_AVAILABLE:
            yield None
            return

        progress_bar = self.progress_display.create_progress_bar(total)
        progress_task_id = progress_bar.add_task(f"[cyan]{description}...", total=total)

        try:
            with progress_bar:
                yield (progress_bar, progress_task_id)
        finally:
            pass

    def _generate_thread_safe_slug(
        self, base_slug: str, card_index: int, lang: str
    ) -> str:
        """Generate unique slug using thread-safe counter.

        This method prevents race conditions in parallel note processing by using
        a thread-safe counter to track slug collisions. Unlike the standard
        generate_slug() which copies the slug set, this ensures modifications
        are visible across all threads.

        Args:
            base_slug: Base slug from note path (without card index or lang)
            card_index: Card index within note
            lang: Language code

        Returns:
            Unique slug with collision counter if needed
        """
        # Initial slug follows format: base-lang (card_index already in base_slug)
        initial_slug = f"{base_slug}-{lang}"

        with self._slug_counter_lock:
            if initial_slug not in self._slug_counters:
                # First time seeing this slug
                self._slug_counters[initial_slug] = 0
                return initial_slug
            else:
                # Collision, use counter
                self._slug_counters[initial_slug] += 1
                collision_count = self._slug_counters[initial_slug]
                return f"{initial_slug}-{collision_count}"

    def sync(
        self,
        dry_run: bool = False,
        sample_size: int | None = None,
        incremental: bool = False,
        build_index: bool = True,
    ) -> dict:
        """
        Perform synchronization.

        Args:
            dry_run: If True, preview changes without applying
            sample_size: Optional number of notes to randomly sample
            incremental: If True, only process new notes not yet in database
            build_index: If True, build index of vault and Anki before sync

        Returns:
            Statistics dict
        """
        logger.info(
            "sync_started",
            dry_run=dry_run,
            sample_size=sample_size,
            incremental=incremental,
            build_index=build_index,
        )

        # Install signal handlers if progress tracking is enabled
        if self.progress:
            self.progress.install_signal_handlers()

        try:
            # Step 0: Build index (if enabled)
            index_stats = None
            if build_index:
                if self.progress:
                    from .progress import SyncPhase

                    self.progress.set_phase(SyncPhase.INDEXING)

                logger.info("building_index")
                index_stats = build_full_index(
                    self.config, self.db, self.anki, incremental=incremental
                )
                logger.info("index_built", stats=index_stats["overall"])

                # Check for interruption
                if self.progress and self.progress.is_interrupted():
                    return self.progress.get_stats()

            # Step 1: Scan Obsidian notes and generate cards
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.SCANNING)

            obsidian_cards = self._scan_obsidian_notes(
                sample_size=sample_size, incremental=incremental
            )

            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                return cast(dict[Any, Any], self.progress.get_stats())

            # Step 2: Fetch Anki state
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.DETERMINING_ACTIONS)

            anki_cards = self._fetch_anki_state()

            # Step 3: Determine sync actions
            self._determine_actions(obsidian_cards, anki_cards)

            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                return cast(dict[Any, Any], self.progress.get_stats())

            # Step 4: Apply or preview
            if dry_run:
                self._print_plan()
            else:
                if self.progress:
                    from .progress import SyncPhase

                    self.progress.set_phase(SyncPhase.APPLYING_CHANGES)

                self._apply_changes()

            # Mark as completed
            if self.progress:
                self.progress.complete(success=True)

            logger.info("sync_completed", stats=self.stats)

            # Log error metrics summary
            if any(
                self.stats.get(metric, 0) > 0
                for metric in [
                    "parser_warnings",
                    "llm_truncations",
                    "validation_errors",
                    "auto_fix_attempts",
                ]
            ):
                logger.info(
                    "error_metrics_summary",
                    parser_warnings=self.stats.get("parser_warnings", 0),
                    llm_truncations=self.stats.get("llm_truncations", 0),
                    validation_errors=self.stats.get("validation_errors", 0),
                    auto_fix_attempts=self.stats.get("auto_fix_attempts", 0),
                    auto_fix_successes=self.stats.get("auto_fix_successes", 0),
                    auto_fix_success_rate=(
                        round(
                            self.stats.get("auto_fix_successes", 0)
                            / self.stats.get("auto_fix_attempts", 1)
                            * 100,
                            1,
                        )
                        if self.stats.get("auto_fix_attempts", 0) > 0
                        else 0.0
                    ),
                )

            # Log LLM session summary
            from ..utils.llm_logging import log_session_summary

            if self.progress:
                log_session_summary(session_id=self.progress.session_id)

            # Log final cache statistics
            if self._cache_stats["hits"] + self._cache_stats["misses"] > 0:
                cache_hit_ratio = self._cache_stats["hits"] / (
                    self._cache_stats["hits"] + self._cache_stats["misses"]
                )
                avg_generation_time = (
                    sum(self._cache_stats["generation_times"])
                    / len(self._cache_stats["generation_times"])
                    if self._cache_stats["generation_times"]
                    else 0
                )
                logger.info(
                    "cache_statistics",
                    hits=self._cache_stats["hits"],
                    misses=self._cache_stats["misses"],
                    hit_ratio=round(cache_hit_ratio, 3),
                    avg_generation_seconds=round(avg_generation_time, 2),
                )

            # Build result dict
            result = self.progress.get_stats() if self.progress else self.stats
            if index_stats:
                result["index"] = index_stats["overall"]

            return result

        except Exception as e:
            logger.error("sync_failed", error=str(e))
            if self.progress:
                self.progress.complete(success=False)
            raise
        finally:
            # Close caches to ensure data is flushed to disk
            self._close_caches()

    def _scan_obsidian_notes(
        self, sample_size: int | None = None, incremental: bool = False
    ) -> dict[str, Card]:
        """
        Scan Obsidian vault and generate cards.

        Args:
            sample_size: Optional number of notes to randomly process
            incremental: If True, only process new notes not yet in database

        Returns:
            Dict of slug -> Card
        """
        logger.info(
            "scanning_obsidian",
            path=str(self.config.vault_path),
            sample_size=sample_size,
            incremental=incremental,
        )

        # Use source_subdirs if configured, otherwise use source_dir
        source_dirs = (
            self.config.source_subdirs
            if self.config.source_subdirs
            else self.config.source_dir
        )
        note_files = discover_notes(self.config.vault_path, source_dirs)

        # Filter for incremental mode
        if incremental:
            processed_paths = self.db.get_processed_note_paths()
            original_count = len(note_files)
            note_files = [
                (file_path, rel_path)
                for file_path, rel_path in note_files
                if rel_path not in processed_paths
            ]
            filtered_count = original_count - len(note_files)
            logger.info(
                "incremental_mode",
                total_notes=original_count,
                new_notes=len(note_files),
                filtered_out=filtered_count,
            )

        if sample_size and sample_size > 0 and len(note_files) > sample_size:
            note_files = random.sample(note_files, sample_size)
            logger.info("sampling_notes", count=sample_size)

        # Set total notes in progress tracker
        if self.progress:
            # Calculate total work units (notes * qa_pairs * languages)
            # For now we'll just count files, will update as we parse
            self.progress.set_total_notes(len(note_files))

        obsidian_cards: dict[str, Card] = {}
        existing_slugs: set[str] = set()

        # Collect topic mismatches for aggregated logging
        topic_mismatches: defaultdict[str, int] = defaultdict(int)

        # Collect errors for aggregated logging
        error_by_type: defaultdict[str, int] = defaultdict(int)
        error_samples: defaultdict[str, list[str]] = defaultdict(list)

        # Progress tracking
        batch_start_time = time.time()
        notes_processed = 0

        # Use parallel processing if enabled and max_concurrent_generations > 1
        use_parallel = (
            self.config.max_concurrent_generations > 1 and len(note_files) > 1
        )

        if use_parallel:
            return self._scan_obsidian_notes_parallel(
                note_files,
                obsidian_cards,
                existing_slugs,
                topic_mismatches,
                error_by_type,
                error_samples,
            )

        # Sequential processing (original implementation)
        # Consecutive error tracking for early termination
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Use context manager for progress bar
        with self._get_progress_bar(
            len(note_files), "Processing notes"
        ) as progress_context:
            if progress_context:
                progress_bar, progress_task_id = progress_context
            else:
                progress_bar = None
                progress_task_id = None
            for file_path, relative_path in note_files:
                # Check for interruption
                if self.progress and self.progress.is_interrupted():
                    break

                # Check for too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "too_many_consecutive_errors",
                        consecutive_errors=consecutive_errors,
                        max_allowed=max_consecutive_errors,
                        last_file=relative_path,
                    )
                    logger.error(
                        "terminating_sync_due_to_errors",
                        message=f"Stopping sync after {consecutive_errors} consecutive errors. "
                        f"This usually indicates a systemic issue (e.g., model unavailable, API errors). "
                        f"Please check the logs and fix the underlying problem before retrying.",
                    )
                    break

                try:
                    # Parse note with repair support
                    # Get LLM provider for repair if available
                    llm_provider_for_repair = None
                    if self.use_agents and self.agent_orchestrator:
                        llm_provider_for_repair = self.agent_orchestrator.provider

                    # Get repair configuration
                    repair_enabled = getattr(self.config, "parser_repair_enabled", True)
                    repair_model = self.config.get_model_for_agent("parser_repair")
                    tolerant_parsing = getattr(self.config, "tolerant_parsing", True)
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
                        enable_repair=repair_enabled,
                        tolerant_parsing=tolerant_parsing,
                        enable_content_generation=enable_content_generation,
                        repair_missing_sections=repair_missing_sections,
                    )

                    # Check for topic mismatch and collect for summary
                    expected_topic = file_path.parent.name
                    if metadata.topic != expected_topic:
                        key = f"{expected_topic} -> {metadata.topic}"
                        topic_mismatches[key] += 1

                    # Read full note content if using agent system
                    note_content = ""
                    if self.use_agents:
                        try:
                            note_content = file_path.read_text(encoding="utf-8")
                        except UnicodeDecodeError as e:
                            logger.warning(
                                "failed_to_read_note_content_encoding_error",
                                file=relative_path,
                                error=str(e),
                            )
                        except OSError as e:
                            logger.warning(
                                "failed_to_read_note_content_io_error",
                                file=relative_path,
                                error=str(e),
                            )
                        except Exception as e:
                            logger.error(
                                "failed_to_read_note_content_unexpected",
                                file=relative_path,
                                error=str(e),
                                exc_info=True,
                            )

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
                        error_type=type(e).__name__,
                    )
                    self.stats["errors"] += 1
                    consecutive_errors += 1
                    continue
                except Exception as e:
                    # Catch any unexpected errors during parsing to prevent full sync failure
                    logger.exception(
                        "unexpected_parsing_error",
                        file=relative_path,
                        error_type=type(e).__name__,
                        error_msg=str(e),
                    )
                    self.stats["errors"] += 1
                    consecutive_errors += 1
                    continue

                try:
                    logger.debug(
                        "processing_note", file=relative_path, pairs=len(qa_pairs)
                    )

                    # Use batch generation if multiple Q/A pairs and batch operations enabled
                    use_batch_generation = (
                        self.config.enable_batch_operations
                        and len(qa_pairs) > 1
                        and not self.use_agents
                    )

                    if use_batch_generation:
                        # Generate all cards for each language in batch
                        for lang in metadata.language_tags:
                            # Prepare manifests for all Q/A pairs
                            manifests = []
                            lang_qa_pairs = []
                            lang_slugs = []

                            for qa_pair in qa_pairs:
                                # Check if already processed
                                if self.progress and self.progress.is_note_completed(
                                    relative_path, qa_pair.card_index, lang
                                ):
                                    continue

                                # Track progress
                                if self.progress:
                                    self.progress.start_note(
                                        relative_path, qa_pair.card_index, lang
                                    )

                                # Generate slug
                                slug, slug_base, hash6 = generate_slug(
                                    relative_path,
                                    qa_pair.card_index,
                                    lang,
                                    existing_slugs,
                                )
                                lang_slugs.append(slug)
                                existing_slugs.add(slug)

                                # Compute GUID
                                guid = deterministic_guid(
                                    [
                                        metadata.id,
                                        relative_path,
                                        str(qa_pair.card_index),
                                        lang,
                                    ]
                                )

                                # Create manifest
                                manifest = create_manifest(
                                    slug,
                                    slug_base,
                                    lang,
                                    relative_path,
                                    qa_pair.card_index,
                                    metadata,
                                    guid,
                                    hash6,
                                )
                                manifests.append(manifest)
                                lang_qa_pairs.append(qa_pair)

                            if not lang_qa_pairs:
                                continue

                            # Generate cards in batch
                            try:
                                batch_cards = self.apf_gen.generate_cards(
                                    lang_qa_pairs, metadata, manifests, lang
                                )

                                for card in batch_cards:
                                    obsidian_cards[card.slug] = card
                                    # Mark as completed
                                    if self.progress:
                                        self.progress.complete_note(
                                            relative_path,
                                            card.manifest.card_index,
                                            lang,
                                            1,
                                        )

                            except Exception as e:
                                logger.error(
                                    "batch_generation_failed",
                                    file=relative_path,
                                    lang=lang,
                                    error=str(e),
                                )
                                # Fall back to individual generation
                                for qa_pair, manifest in zip(lang_qa_pairs, manifests):
                                    try:
                                        card = self._generate_card(
                                            qa_pair=qa_pair,
                                            metadata=metadata,
                                            relative_path=relative_path,
                                            lang=lang,
                                            existing_slugs=existing_slugs,
                                            note_content=note_content,
                                            all_qa_pairs=qa_pairs,
                                        )
                                        obsidian_cards[card.slug] = card
                                        existing_slugs.add(card.slug)

                                        if self.progress:
                                            self.progress.complete_note(
                                                relative_path,
                                                qa_pair.card_index,
                                                lang,
                                                1,
                                            )
                                    except Exception as card_error:
                                        error_type_name = type(card_error).__name__
                                        error_message = str(card_error)
                                        error_by_type[error_type_name] += 1
                                        if len(error_samples[error_type_name]) < 3:
                                            error_samples[error_type_name].append(
                                                f"{relative_path} (pair {qa_pair.card_index}, {lang}): {error_message}"
                                            )
                                        self.stats["errors"] += 1
                                        if self.progress:
                                            self.progress.fail_note(
                                                relative_path,
                                                qa_pair.card_index,
                                                lang,
                                                str(card_error),
                                            )
                    else:
                        # Original sequential generation
                        # Generate cards for each Q/A pair and language
                        for qa_pair in qa_pairs:
                            for lang in metadata.language_tags:
                                # Check if already processed (for resume)
                                if self.progress and self.progress.is_note_completed(
                                    relative_path, qa_pair.card_index, lang
                                ):
                                    logger.debug(
                                        "skipping_completed_note",
                                        file=relative_path,
                                        card_index=qa_pair.card_index,
                                        lang=lang,
                                    )
                                    continue

                                # Track progress
                                if self.progress:
                                    self.progress.start_note(
                                        relative_path, qa_pair.card_index, lang
                                    )

                                try:
                                    card = self._generate_card(
                                        qa_pair=qa_pair,
                                        metadata=metadata,
                                        relative_path=relative_path,
                                        lang=lang,
                                        existing_slugs=existing_slugs,
                                        note_content=note_content,
                                        all_qa_pairs=qa_pairs,
                                    )
                                    obsidian_cards[card.slug] = card
                                    existing_slugs.add(card.slug)

                                    # Reset consecutive errors on success
                                    consecutive_errors = 0

                                    # Mark as completed
                                    if self.progress:
                                        self.progress.complete_note(
                                            relative_path, qa_pair.card_index, lang, 1
                                        )

                                except Exception as e:
                                    error_type_name = type(e).__name__
                                    error_message = str(e)

                                    # Archive problematic note
                                    try:
                                        note_content = ""
                                        try:
                                            note_content = file_path.read_text(
                                                encoding="utf-8"
                                            )
                                        except (
                                            UnicodeDecodeError,
                                            OSError,
                                        ) as read_err:
                                            logger.debug(
                                                "unable_to_read_note_for_archiving",
                                                file=relative_path,
                                                error=str(read_err),
                                            )

                                        self.archiver.archive_note(
                                            note_path=file_path,
                                            error=e,
                                            error_type=error_type_name,
                                            processing_stage="card_generation",
                                            card_index=qa_pair.card_index,
                                            language=lang,
                                            note_content=(
                                                note_content if note_content else None
                                            ),
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
                                            f"{relative_path} (pair {qa_pair.card_index}, {lang}): {error_message}"
                                        )

                                    self.stats["errors"] += 1
                                    consecutive_errors += 1

                                    if self.progress:
                                        self.progress.fail_note(
                                            relative_path,
                                            qa_pair.card_index,
                                            lang,
                                            str(e),
                                        )

                    self.stats["processed"] += 1

                except Exception as e:
                    # Catch unexpected errors during card generation to prevent full sync failure
                    logger.exception(
                        "card_generation_failed",
                        file=relative_path,
                        error_type=type(e).__name__,
                        error_msg=str(e),
                    )

                    # Archive problematic note
                    try:
                        note_content = ""
                        try:
                            note_content = file_path.read_text(encoding="utf-8")
                        except (UnicodeDecodeError, OSError) as read_err:
                            logger.debug(
                                "unable_to_read_note_for_archiving",
                                file=relative_path,
                                error=str(read_err),
                            )

                        self.archiver.archive_note(
                            note_path=file_path,
                            error=e,
                            error_type=type(e).__name__,
                            processing_stage="card_generation",
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

                    self.stats["errors"] += 1
                    consecutive_errors += 1

            # Update progress bar
            notes_processed += 1
            if progress_bar and progress_task_id is not None:
                progress_bar.update(
                    progress_task_id,
                    advance=1,
                    description=f"[cyan]Processing notes... ({notes_processed}/{len(note_files)})",
                )
                # Update status panel
                if self.progress_display:
                    note_name = Path(relative_path).stem
                    self.progress_display.update_operation(
                        f"Processing note {notes_processed}/{len(note_files)}",
                        note_name,
                    )

            # Progress indicator (log every 10 notes or on completion)
            if notes_processed % 10 == 0 or notes_processed == len(note_files):
                elapsed_time = time.time() - batch_start_time
                avg_time_per_note = (
                    elapsed_time / notes_processed if notes_processed > 0 else 0
                )
                remaining_notes = len(note_files) - notes_processed
                estimated_remaining = avg_time_per_note * remaining_notes

                logger.info(
                    "batch_progress",
                    processed=notes_processed,
                    total=len(note_files),
                    percent=f"{(notes_processed / len(note_files) * 100):.1f}%",
                    elapsed_seconds=round(elapsed_time, 1),
                    avg_seconds_per_note=round(avg_time_per_note, 2),
                    estimated_remaining_seconds=round(estimated_remaining, 1),
                    cards_generated=len(obsidian_cards),
                )

        # Log if sync was terminated early due to consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            logger.error(
                "sync_terminated_early",
                reason="too_many_consecutive_errors",
                consecutive_errors=consecutive_errors,
                notes_processed=notes_processed,
                total_notes=len(note_files),
                notes_remaining=len(note_files) - notes_processed,
            )

        # Log aggregated error summary
        if error_by_type:
            logger.warning(
                "card_generation_errors_summary",
                total_errors=self.stats["errors"],
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

        # Log cache statistics
        if self._cache_hits > 0 or self._cache_misses > 0:
            total_cache_requests = self._cache_hits + self._cache_misses
            hit_rate = (
                (self._cache_hits / total_cache_requests * 100)
                if total_cache_requests > 0
                else 0
            )
            # Get cache sizes (diskcache provides volume() for size in bytes)
            agent_cache_size_bytes = self._agent_card_cache.volume()
            apf_cache_size_bytes = self._apf_card_cache.volume()
            logger.info(
                "cache_statistics",
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                hit_rate=f"{hit_rate:.1f}%",
                agent_cache_size_mb=f"{agent_cache_size_bytes / (1024 * 1024):.2f}",
                apf_cache_size_mb=f"{apf_cache_size_bytes / (1024 * 1024):.2f}",
            )

        logger.info(
            "obsidian_scan_completed", notes=len(note_files), cards=len(obsidian_cards)
        )

        return obsidian_cards

    def _process_single_note(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file and generate cards.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)

        Returns:
            Tuple of (cards_dict, new_slugs_set, result_info)
        """
        cards: dict[str, Card] = {}
        new_slugs: set[str] = set()
        result_info = {
            "success": False,
            "error": None,
            "error_type": None,
            "cards_count": 0,
        }

        try:
            # Parse note with optional QA extractor
            metadata, qa_pairs = parse_note(file_path, qa_extractor=self.qa_extractor)

            # Read full note content if using agent system
            note_content = ""
            if self.use_agents:
                try:
                    note_content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(
                        "failed_to_read_note_content_encoding_error",
                        file=relative_path,
                        error=str(e),
                    )
                except OSError as e:
                    logger.warning(
                        "failed_to_read_note_content_io_error",
                        file=relative_path,
                        error=str(e),
                    )
                except Exception as e:
                    logger.error(
                        "failed_to_read_note_content_unexpected",
                        file=relative_path,
                        error=str(e),
                        exc_info=True,
                    )

            # Generate cards for each Q/A pair and language
            for qa_pair in qa_pairs:
                for lang in metadata.language_tags:
                    # Check if already processed (for resume)
                    if self.progress and self.progress.is_note_completed(
                        relative_path, qa_pair.card_index, lang
                    ):
                        continue

                    # Track progress
                    if self.progress:
                        self.progress.start_note(
                            relative_path, qa_pair.card_index, lang
                        )

                    try:
                        card = self._generate_card(
                            qa_pair=qa_pair,
                            metadata=metadata,
                            relative_path=relative_path,
                            lang=lang,
                            existing_slugs=existing_slugs.copy(),  # Copy to avoid race conditions
                            note_content=note_content,
                            all_qa_pairs=qa_pairs,
                        )
                        cards[card.slug] = card
                        new_slugs.add(card.slug)

                        # Mark as completed
                        if self.progress:
                            self.progress.complete_note(
                                relative_path, qa_pair.card_index, lang, 1
                            )

                    except Exception as e:
                        error_type_name = type(e).__name__
                        error_message = str(e)

                        # Archive problematic note using helper
                        self._archive_note_safely(
                            file_path=file_path,
                            relative_path=relative_path,
                            error=e,
                            processing_stage="card_generation",
                            note_content=note_content if note_content else None,
                            card_index=qa_pair.card_index,
                            language=lang,
                            context={
                                "metadata_id": (
                                    metadata.id if "metadata" in locals() else None
                                ),
                                "qa_pairs_count": (
                                    len(qa_pairs) if "qa_pairs" in locals() else None
                                ),
                            },
                        )

                        if self.progress:
                            self.progress.fail_note(
                                relative_path, qa_pair.card_index, lang, error_message
                            )

                        result_info["error"] = error_message
                        result_info["error_type"] = error_type_name
                        # Continue processing other cards in this note

            result_info["success"] = True
            result_info["cards_count"] = len(cards)

        except (
            ParserError,
            yaml.YAMLError,
            OSError,
            UnicodeDecodeError,
        ) as e:
            # Archive problematic note using helper
            self._archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="parsing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        except Exception as e:
            # Archive problematic note using helper
            self._archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="processing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        return cards, new_slugs, result_info

    def _scan_obsidian_notes_parallel(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        topic_mismatches: defaultdict[str, int],
        error_by_type: defaultdict[str, int],
        error_samples: defaultdict[str, list[str]],
    ) -> dict[str, Card]:
        """Scan Obsidian notes using parallel processing.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs (thread-safe updates)
            topic_mismatches: Dict to collect topic mismatches
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors

        Returns:
            Dict of slug -> Card
        """
        max_workers = min(self.config.max_concurrent_generations, len(note_files))
        logger.info(
            "parallel_scan_started",
            total_notes=len(note_files),
            max_workers=max_workers,
        )

        # Initialize slug counters from existing slugs to prevent collisions
        with self._slug_counter_lock:
            for slug in existing_slugs:
                self._slug_counters[slug] = 0

        # Thread-safe tracking using locks
        import threading

        # Lock for slug and card updates
        slugs_lock = threading.Lock()
        shared_slugs = set(existing_slugs)

        # Lock for stats and error tracking (separate to reduce contention)
        stats_lock = threading.Lock()

        # Process notes in parallel
        notes_processed = 0
        batch_start_time = time.time()

        # Use context manager for progress bar (parallel processing)
        with self._get_progress_bar(
            len(note_files), f"Processing notes (parallel, {max_workers} workers)"
        ) as progress_context:
            if progress_context:
                progress_bar, progress_task_id = progress_context
            else:
                progress_bar = None
                progress_task_id = None
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_note = {
                    executor.submit(
                        self._process_single_note,
                        file_path,
                        relative_path,
                        shared_slugs,
                    ): (file_path, relative_path)
                    for file_path, relative_path in note_files
                }

                # Process results as they complete
                for future in as_completed(future_to_note):
                    # Check for interruption
                    if self.progress and self.progress.is_interrupted():
                        # Cancel remaining tasks
                        for f in future_to_note:
                            f.cancel()
                        break

                    file_path, relative_path = future_to_note[future]

                    try:
                        cards, new_slugs, result_info = future.result()

                        # Thread-safe update of slug/card state
                        with slugs_lock:
                            obsidian_cards.update(cards)
                            shared_slugs.update(new_slugs)
                            existing_slugs.update(new_slugs)

                        # Thread-safe update of stats and error tracking
                        with stats_lock:
                            if result_info["success"]:
                                self.stats["processed"] += 1
                            else:
                                self.stats["errors"] += 1
                                if result_info["error_type"]:
                                    error_by_type[result_info["error_type"]] += 1
                                    if len(error_samples[result_info["error_type"]]) < 3:
                                        error_samples[result_info["error_type"]].append(
                                            f"{relative_path}: {result_info['error']}"
                                        )

                    except Exception as e:
                        logger.exception(
                            "parallel_note_processing_failed",
                            file=relative_path,
                            error=str(e),
                        )
                        # Thread-safe update of stats and error tracking
                        with stats_lock:
                            self.stats["errors"] += 1
                            error_type_name = type(e).__name__
                            error_by_type[error_type_name] += 1
                            if len(error_samples[error_type_name]) < 3:
                                error_samples[error_type_name].append(
                                    f"{relative_path}: {str(e)}"
                                )

                    # Progress indicator
                    notes_processed += 1
                    if notes_processed % 10 == 0 or notes_processed == len(note_files):
                        elapsed_time = time.time() - batch_start_time
                        avg_time_per_note = (
                            elapsed_time / notes_processed if notes_processed > 0 else 0
                        )
                        remaining_notes = len(note_files) - notes_processed
                        estimated_remaining = avg_time_per_note * remaining_notes

                        logger.info(
                            "parallel_batch_progress",
                            processed=notes_processed,
                            total=len(note_files),
                            percent=f"{(notes_processed / len(note_files) * 100):.1f}%",
                            elapsed_seconds=round(elapsed_time, 1),
                            avg_seconds_per_note=round(avg_time_per_note, 2),
                            estimated_remaining_seconds=round(estimated_remaining, 1),
                            cards_generated=len(obsidian_cards),
                            active_workers=max_workers,
                        )

                    # Update progress bar
                    if progress_bar and progress_task_id is not None:
                        progress_bar.update(
                            progress_task_id,
                            advance=1,
                            description=f"[cyan]Processing notes ({notes_processed}/{len(note_files)}, {max_workers} workers)...",
                        )
                        # Update status panel
                        if self.progress_display:
                            note_name = (
                                Path(relative_path).stem if relative_path else "unknown"
                            )
                            self.progress_display.update_operation(
                                f"Processing note {notes_processed}/{len(note_files)} (parallel)",
                                note_name,
                            )

        logger.info(
            "parallel_scan_completed",
            notes=len(note_files),
            cards=len(obsidian_cards),
            workers_used=max_workers,
        )

        return obsidian_cards

    def _generate_cards_with_agents(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        relative_path: str,
    ) -> list[Card]:
        """Generate all cards for a note using the agent system.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: List of Q/A pairs
            relative_path: Relative path to note

        Returns:
            List of generated cards
        """
        if not self.use_agents or not self.agent_orchestrator:
            raise RuntimeError("Agent system not initialized")

        # Compute content hash for the entire note
        # Use a hash of all Q/A pairs and metadata to detect changes
        import hashlib

        content_components = [
            metadata.id,
            metadata.title,
            metadata.topic,
            ",".join(sorted(metadata.subtopics)),
            ",".join(sorted(metadata.tags)),
            note_content,
        ]
        note_content_hash = hashlib.sha256(
            "\n".join(str(c) for c in content_components).encode("utf-8")
        ).hexdigest()[
            :16
        ]  # Use first 16 chars for shorter keys

        # Check cache with content hash
        cache_key = f"{metadata.id}:{relative_path}:{note_content_hash}"
        try:
            cached_cards = self._agent_card_cache.get(cache_key)
            if cached_cards is not None:
                # Verify content hash matches (double-check)
                self._cache_hits += 1
                logger.info(
                    "agent_cache_hit",
                    cache_key=cache_key,
                    note=relative_path,
                    cards_returned=len(cached_cards),
                    content_hash=note_content_hash,
                )
                return cached_cards
        except Exception as e:
            logger.warning(
                "agent_cache_read_error",
                cache_key=cache_key,
                error=str(e),
            )
            # Continue with cache miss on error

        self._cache_misses += 1
        logger.info(
            "generating_cards_with_agents",
            note=relative_path,
            qa_pairs=len(qa_pairs),
            cache_miss=True,
        )

        # Run agent pipeline
        import asyncio
        from pathlib import Path

        file_path = self.config.vault_path / relative_path

        # Validate file path exists before processing
        if not file_path.exists():
            logger.warning(
                "file_path_not_found_for_agent_processing",
                relative_path=relative_path,
                computed_path=str(file_path),
            )
            # Continue with None file_path, agent can handle it

        # Handle both sync (AgentOrchestrator) and async (LangGraphOrchestrator) orchestrators
        if hasattr(self.agent_orchestrator, "process_note"):
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(self.agent_orchestrator.process_note):
                # Async orchestrator (LangGraphOrchestrator)
                # Use asyncio.run() for clean loop management
                result = asyncio.run(
                    self.agent_orchestrator.process_note(
                        note_content=note_content,
                        metadata=metadata,
                        qa_pairs=qa_pairs,
                        file_path=Path(file_path) if file_path.exists() else None,
                    )
                )
            else:
                # Sync orchestrator (AgentOrchestrator)
                result = self.agent_orchestrator.process_note(
                    note_content=note_content,
                    metadata=metadata,
                    qa_pairs=qa_pairs,
                    file_path=Path(file_path) if file_path.exists() else None,
                )
        else:
            raise RuntimeError("Orchestrator does not have process_note method")

        # Track metrics from agent pipeline
        if result.post_validation:
            if not result.post_validation.is_valid:
                self.stats["validation_errors"] += 1
        if result.retry_count > 0:
            self.stats["auto_fix_attempts"] += result.retry_count
            if result.success:
                self.stats["auto_fix_successes"] += 1

        if not result.success or not result.generation:
            error_msg = (
                result.post_validation.error_details
                if result.post_validation
                else "Unknown error"
            )
            raise ValueError(f"Agent pipeline failed: {error_msg}")

        # Convert GeneratedCard to Card instances
        cards = self.agent_orchestrator.convert_to_cards(
            result.generation.cards, metadata, qa_pairs
        )

        # Update card metadata with proper paths and GUIDs
        for card in cards:
            card.manifest.source_path = relative_path
            card.manifest.note_id = metadata.id
            card.manifest.note_title = metadata.title

        # Cache the results
        try:
            self._agent_card_cache.set(cache_key, cards)
        except Exception as e:
            logger.warning(
                "agent_cache_write_error",
                cache_key=cache_key,
                error=str(e),
            )
            # Continue even if cache write fails

        logger.info(
            "agent_generation_success",
            note=relative_path,
            cards_generated=len(cards),
            time=result.total_time,
        )

        return cards

    def _generate_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        relative_path: str,
        lang: str,
        existing_slugs: set[str],
        note_content: str = "",
        all_qa_pairs: list[QAPair] | None = None,
    ) -> Card:
        """Generate a single card.

        Args:
            qa_pair: Q/A pair to generate card for
            metadata: Note metadata
            relative_path: Relative path to note
            lang: Language code
            existing_slugs: Set of existing slugs
            note_content: Full note content (required for agent system)
            all_qa_pairs: All Q/A pairs from note (required for agent system)

        Returns:
            Generated card
        """
        start_time = time.time()

        # Use agent system if enabled
        if self.use_agents:
            if not note_content or all_qa_pairs is None:
                raise ValueError(
                    "note_content and all_qa_pairs required when using agent system"
                )

            # Generate all cards for the note (cached)
            all_cards = self._generate_cards_with_agents(
                note_content, metadata, all_qa_pairs, relative_path
            )

            # Find the specific card for this qa_pair and lang
            for card in all_cards:
                if card.manifest.card_index == qa_pair.card_index and card.lang == lang:
                    # Track cache hit for agent system (already counted in _generate_cards_with_agents)
                    elapsed_ms = round((time.time() - start_time) * 1000, 2)
                    self._cache_stats["hits"] += 1
                    logger.debug(
                        "card_generation_cache_hit_agent",
                        slug=card.slug,
                        elapsed_ms=elapsed_ms,
                    )
                    return card

            raise ValueError(
                f"Agent system did not generate card for index={qa_pair.card_index}, lang={lang}"
            )

        # Check cache for non-agent generation
        content_hash = compute_content_hash(qa_pair, metadata, lang)
        cache_key = f"{relative_path}:{qa_pair.card_index}:{lang}:{content_hash}"

        try:
            cached_card = self._apf_card_cache.get(cache_key)
            if cached_card is not None:
                # Verify content hash matches
                if cached_card.content_hash == content_hash:
                    elapsed_ms = round((time.time() - start_time) * 1000, 2)
                    self._cache_hits += 1
                    self._cache_stats["hits"] += 1
                    logger.debug(
                        "card_generation_cache_hit",
                        slug=cached_card.slug,
                        elapsed_ms=elapsed_ms,
                    )
                    return cached_card
        except Exception as e:
            logger.warning(
                "apf_cache_read_error",
                cache_key=cache_key,
                error=str(e),
            )
            # Continue with cache miss on error

        self._cache_misses += 1
        self._cache_stats["misses"] += 1

        # Original APFGenerator logic
        # Generate slug - use thread-safe method in parallel mode if counters are initialized
        if self._slug_counters:
            # Thread-safe mode - compute base slug from path
            import re
            import unicodedata
            from pathlib import Path

            # Normalize path to slug base (same logic as slug_generator)
            path_parts = Path(relative_path).with_suffix("").parts
            slug_parts = []
            for part in path_parts:
                normalized = unicodedata.normalize("NFKD", part)
                ascii_segment = normalized.encode("ascii", "ignore").decode("ascii")
                ascii_segment = re.sub(r"[^a-z0-9-]", "-", ascii_segment.lower())
                ascii_segment = re.sub(r"-+", "-", ascii_segment).strip("-")
                if ascii_segment:
                    slug_parts.append(ascii_segment)
            sanitized = "-".join(slug_parts) or "note"
            base_without_suffix = f"{sanitized}-p{qa_pair.card_index:02d}"
            slug_base = base_without_suffix[:70]  # MAX_SLUG_LENGTH

            # Use thread-safe counter for collision resolution
            slug = self._generate_thread_safe_slug(slug_base, qa_pair.card_index, lang)
            hash6 = None
        else:
            # Sequential mode - use standard generate_slug
            slug, slug_base, hash6 = generate_slug(
                relative_path, qa_pair.card_index, lang, existing_slugs
            )

        # Compute deterministic GUID for the note
        guid = deterministic_guid(
            [metadata.id, relative_path, str(qa_pair.card_index), lang]
        )

        # Create manifest
        manifest = create_manifest(
            slug,
            slug_base,
            lang,
            relative_path,
            qa_pair.card_index,
            metadata,
            guid,
            hash6,
        )

        # Generate APF card via LLM
        card = cast(Card, self.apf_gen.generate_card(qa_pair, metadata, manifest, lang))

        # Ensure content hash is set
        if not card.content_hash:
            card.content_hash = content_hash

        # Validate APF format
        validation = validate_apf(card.apf_html, slug)
        if validation.errors:
            self.stats["validation_errors"] += len(validation.errors)
            logger.error("apf_validation_errors", slug=slug, errors=validation.errors)
            raise ValueError(
                f"APF validation failed for {slug}: {validation.errors[0]}"
            )
        if validation.warnings:
            logger.debug(
                "apf_validation_warnings", slug=slug, warnings=validation.warnings
            )

        html_errors = validate_card_html(card.apf_html)
        if html_errors:
            logger.error("apf_html_invalid", slug=slug, errors=html_errors)
            raise ValueError(f"Invalid HTML formatting for {slug}: {html_errors[0]}")

        # Cache the generated card
        try:
            self._apf_card_cache.set(cache_key, card)
            logger.debug(
                "apf_cache_stored",
                cache_key=cache_key,
                slug=slug,
                content_hash=content_hash[:8],
            )
        except Exception as e:
            logger.warning(
                "apf_cache_write_error",
                cache_key=cache_key,
                error=str(e),
            )
            # Continue even if cache write fails

        # Log generation time
        elapsed = time.time() - start_time
        self._cache_stats["generation_times"].append(elapsed)
        logger.info("card_generated", slug=slug, elapsed_seconds=round(elapsed, 2))

        return card

    def _fetch_anki_state(self) -> dict[str, int]:
        """
        Fetch current Anki state with detailed logging.

        Returns:
            Dict of slug -> anki_note_id
        """
        start_time = time.time()
        logger.info("fetching_anki_state", deck=self.config.anki_deck_name)

        # Find all notes in target deck
        try:
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
                "no_anki_notes_found",
                deck=self.config.anki_deck_name,
                elapsed_seconds=round(elapsed, 2),
            )
            return {}

        # Fetch info with progress
        logger.info("fetching_note_info", note_count=len(note_ids))
        try:
            notes_info = self.anki.notes_info(note_ids)
        except AnkiConnectError as e:
            elapsed = time.time() - start_time
            logger.error(
                "anki_notes_info_failed_connect_error",
                note_count=len(note_ids),
                error=str(e),
                elapsed_seconds=round(elapsed, 2),
            )
            return {}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "anki_notes_info_failed_unexpected",
                note_count=len(note_ids),
                error=str(e),
                elapsed_seconds=round(elapsed, 2),
                exc_info=True,
            )
            return {}

        # Extract slugs from manifests
        anki_cards = {}
        invalid_manifest_count = 0
        for note_info in notes_info:
            # Look for Manifest field
            fields = note_info.get("fields", {})
            manifest_field = fields.get("Manifest", {}).get("value", "{}")

            manifest = self._parse_manifest_field(manifest_field)
            if manifest is None:
                invalid_manifest_count += 1
                logger.warning(
                    "skipping_card_invalid_manifest",
                    note_id=note_info.get("noteId"),
                )
                continue

            slug = manifest.slug  # Now guaranteed to be a valid string
            anki_cards[slug] = note_info["noteId"]

        elapsed = time.time() - start_time
        logger.info(
            "anki_state_fetched",
            deck=self.config.anki_deck_name,
            cards_found=len(anki_cards),
            notes_processed=len(notes_info),
            invalid_manifests=invalid_manifest_count,
            elapsed_seconds=round(elapsed, 2),
        )
        return anki_cards

    def _determine_actions(
        self, obsidian_cards: dict[str, Card], anki_cards: dict[str, int]
    ) -> None:
        """
        Determine what actions to take.

        Args:
            obsidian_cards: Cards from Obsidian
            anki_cards: Current Anki state (slug -> note_id)
        """
        logger.info("determining_actions")

        # Get database state
        db_cards = {c["slug"]: c for c in self.db.get_all_cards()}

        # Check each Obsidian card
        for slug, obs_card in obsidian_cards.items():
            db_card = db_cards.get(slug)
            anki_id = anki_cards.get(slug)

            if not db_card and not anki_id:
                # New card - create
                self.changes.append(
                    SyncAction(
                        type="create",
                        card=obs_card,
                        reason="New card not in database or Anki",
                    )
                )

            elif db_card and obs_card.content_hash != db_card["content_hash"]:
                # Updated card - update
                self.changes.append(
                    SyncAction(
                        type="update",
                        card=obs_card,
                        anki_guid=db_card["anki_guid"],
                        reason=f"Content changed (old hash: {db_card['content_hash'][:8]}...)",
                    )
                )

            else:
                # No changes - skip
                self.changes.append(
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

                self.changes.append(
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
                self.changes.append(
                    SyncAction(
                        type="restore",
                        card=obsidian_cards[slug],
                        reason="Card deleted in Anki, restoring from Obsidian",
                    )
                )

        # Count actions
        action_counts: dict[str, int] = {}
        for action in self.changes:
            action_counts[action.type] = action_counts.get(action.type, 0) + 1

        logger.info("actions_determined", actions=action_counts)

    def _print_plan(self) -> None:
        """Print sync plan for dry-run."""
        print("\n=== Sync Plan (Dry Run) ===\n")

        for action in self.changes:
            if action.type == "skip":
                continue

            print(f"[{action.type.upper()}] {action.card.slug}")
            if action.reason:
                print(f"  Reason: {action.reason}")
            print()

        # Print summary
        action_counts: dict[str, int] = {}
        for action in self.changes:
            action_counts[action.type] = action_counts.get(action.type, 0) + 1

        print("=== Summary ===")
        for action_type, count in sorted(action_counts.items()):
            print(f"{action_type}: {count}")
        print()

    def _apply_changes(self) -> None:
        """Apply sync actions to Anki."""
        logger.info("applying_changes", count=len(self.changes))

        # Use batch operations if enabled
        if self.config.enable_batch_operations:
            self._apply_changes_batched()
        else:
            self._apply_changes_sequential()

    def _apply_changes_sequential(self) -> None:
        """Apply changes sequentially (original implementation)."""
        for action in self.changes:
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            try:
                if action.type == "create":
                    self._create_card(action.card)
                    self.stats["created"] += 1
                    if self.progress:
                        self.progress.increment_stat("created")

                elif action.type == "update":
                    if action.anki_guid:
                        self._update_card(action.card, action.anki_guid)
                        self.stats["updated"] += 1
                        if self.progress:
                            self.progress.increment_stat("updated")

                elif action.type == "delete":
                    if action.anki_guid:
                        self._delete_card(action.card, action.anki_guid)
                        self.stats["deleted"] += 1
                        if self.progress:
                            self.progress.increment_stat("deleted")

                elif action.type == "restore":
                    self._create_card(action.card)
                    self.stats["restored"] += 1
                    if self.progress:
                        self.progress.increment_stat("restored")

                elif action.type == "skip":
                    self.stats["skipped"] += 1
                    if self.progress:
                        self.progress.increment_stat("skipped")

            except CardOperationError as e:
                logger.error(
                    "action_failed_card_operation",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                )
                self.stats["errors"] += 1
                if self.progress:
                    self.progress.increment_stat("errors")
            except AnkiConnectError as e:
                logger.error(
                    "action_failed_anki_connect",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                )
                self.stats["errors"] += 1
                if self.progress:
                    self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "action_failed_unexpected",
                    action=action.type,
                    slug=action.card.slug,
                    error=str(e),
                    exc_info=True,
                )
                self.stats["errors"] += 1
                if self.progress:
                    self.progress.increment_stat("errors")

    def _apply_changes_batched(self) -> None:
        """Apply changes using batch operations for better performance."""
        # Group actions by type
        creates: list[SyncAction] = []
        updates: list[SyncAction] = []
        deletes: list[SyncAction] = []
        restores: list[SyncAction] = []

        for action in self.changes:
            if action.type == "create":
                creates.append(action)
            elif action.type == "update" and action.anki_guid:
                updates.append(action)
            elif action.type == "delete" and action.anki_guid:
                deletes.append(action)
            elif action.type == "restore":
                restores.append(action)
            elif action.type == "skip":
                self.stats["skipped"] += 1
                if self.progress:
                    self.progress.increment_stat("skipped")

        # Process creates in batches
        if creates or restores:
            all_creates = creates + restores
            self._create_cards_batch(all_creates)

        # Process updates in batches
        if updates:
            self._update_cards_batch(updates)

        # Process deletes (already batched in delete_notes)
        if deletes:
            self._delete_cards_batch(deletes)

    def _create_card(self, card: Card) -> None:
        """Create card in Anki with atomic transaction."""
        logger.info("creating_card", slug=card.slug)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        try:
            with CardTransaction(self.anki, self.db) as txn:
                # Step 1: Add to Anki
                note_id = self.anki.add_note(
                    deck=self.config.anki_deck_name,
                    note_type=card.note_type,
                    fields=fields,
                    tags=card.tags,
                    guid=card.guid,
                )

                # Register rollback action
                txn.rollback_actions.append(("delete_anki_note", note_id))

                # Step 2: Save to database with full content
                self.db.insert_card_extended(
                    card=card,
                    anki_guid=note_id,
                    fields=fields,
                    tags=card.tags,
                    deck_name=self.config.anki_deck_name,
                    apf_html=card.apf_html,
                )

                # Mark as committed (no rollback needed)
                txn.commit()

                logger.info(
                    "card_created_successfully", slug=card.slug, anki_guid=note_id
                )

                # Verify card creation if enabled
                if getattr(self.config, "verify_card_creation", True):
                    self._verify_card_creation(card, note_id, fields, card.tags)

        except AnkiConnectError as e:
            logger.error("anki_create_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

        except Exception as e:
            logger.error("card_create_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise CardOperationError(f"Failed to create card {card.slug}: {e}")

    def _verify_card_creation(
        self,
        card: Card,
        note_id: int,
        expected_fields: dict[str, str],
        expected_tags: list[str],
    ) -> None:
        """Verify that a card was successfully created in Anki.

        Args:
            card: Card object that was created
            note_id: Anki note ID returned from creation
            expected_fields: Expected field values
            expected_tags: Expected tags

        Raises:
            CardOperationError: If verification fails critically
        """
        try:
            # Get note info from Anki
            notes_info = self.anki.notes_info([note_id])
            if not notes_info:
                logger.error(
                    "card_verification_failed_not_found",
                    slug=card.slug,
                    note_id=note_id,
                )
                raise CardOperationError(
                    f"Card {card.slug} (note_id={note_id}) not found in Anki after creation"
                )

            note_info = notes_info[0]

            # Verify note exists
            if note_info.get("noteId") != note_id:
                logger.error(
                    "card_verification_failed_id_mismatch",
                    slug=card.slug,
                    expected_note_id=note_id,
                    actual_note_id=note_info.get("noteId"),
                )
                raise CardOperationError(
                    f"Card {card.slug} verification failed: note ID mismatch"
                )

            # Verify deck
            actual_deck = note_info.get("deckName", "")
            expected_deck = self.config.anki_deck_name
            if actual_deck != expected_deck:
                logger.warning(
                    "card_verification_deck_mismatch",
                    slug=card.slug,
                    expected_deck=expected_deck,
                    actual_deck=actual_deck,
                )

            # Verify note type
            actual_note_type = note_info.get("modelName", "")
            if actual_note_type != card.note_type:
                logger.warning(
                    "card_verification_note_type_mismatch",
                    slug=card.slug,
                    expected_note_type=card.note_type,
                    actual_note_type=actual_note_type,
                )

            # Verify fields (check key fields only to avoid false positives from formatting)
            actual_fields = note_info.get("fields", {})
            field_mismatches = []
            for field_name, expected_value in expected_fields.items():
                actual_value = actual_fields.get(field_name, {}).get("value", "")
                # Normalize whitespace for comparison
                expected_normalized = " ".join(expected_value.split())
                actual_normalized = " ".join(actual_value.split())
                if expected_normalized != actual_normalized:
                    field_mismatches.append(field_name)
                    logger.debug(
                        "card_verification_field_mismatch",
                        slug=card.slug,
                        field=field_name,
                        expected_length=len(expected_value),
                        actual_length=len(actual_value),
                    )

            if field_mismatches:
                logger.warning(
                    "card_verification_field_mismatches",
                    slug=card.slug,
                    mismatched_fields=field_mismatches,
                )

            # Verify tags
            actual_tags = set(note_info.get("tags", []))
            expected_tags_set = set(expected_tags)
            if actual_tags != expected_tags_set:
                missing_tags = expected_tags_set - actual_tags
                extra_tags = actual_tags - expected_tags_set
                if missing_tags or extra_tags:
                    logger.warning(
                        "card_verification_tag_mismatch",
                        slug=card.slug,
                        missing_tags=list(missing_tags),
                        extra_tags=list(extra_tags),
                    )

            logger.debug(
                "card_verification_succeeded",
                slug=card.slug,
                note_id=note_id,
            )

        except AnkiConnectError as e:
            logger.error(
                "card_verification_failed_anki_error",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            # Don't raise - verification failure shouldn't break the sync
            # The card was created, verification is just a safety check

        except CardOperationError:
            # Re-raise critical verification failures
            raise

        except Exception as e:
            logger.error(
                "card_verification_failed_unexpected",
                slug=card.slug,
                note_id=note_id,
                error=str(e),
            )
            # Don't raise - verification failure shouldn't break the sync

    def _update_card(self, card: Card, anki_guid: int) -> None:
        """Update card in Anki with atomic transaction."""
        logger.info("updating_card", slug=card.slug, anki_guid=anki_guid)

        # Map fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        try:
            with CardTransaction(self.anki, self.db) as txn:
                # Get current state for potential rollback
                current_info = self.anki.notes_info([anki_guid])
                old_fields = {}
                old_tags = []
                if current_info:
                    old_fields = current_info[0].get("fields", {})
                    old_tags = current_info[0].get("tags", [])

                # Step 1: Update in Anki
                self.anki.update_note_fields(anki_guid, fields)
                self.anki.update_note_tags(anki_guid, card.tags)

                # Register rollback (restore old state)
                if current_info:
                    txn.rollback_actions.append(
                        ("restore_anki_note", anki_guid, old_fields, old_tags)
                    )

                # Step 2: Update database with full content
                self.db.update_card_extended(
                    card=card, fields=fields, tags=card.tags, apf_html=card.apf_html
                )

                txn.commit()
                logger.info("card_updated_successfully", slug=card.slug)

        except AnkiConnectError as e:
            logger.error("anki_update_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise

        except Exception as e:
            logger.error("card_update_failed", slug=card.slug, error=str(e))
            self.db.update_card_status(card.slug, "failed", str(e))
            raise CardOperationError(f"Failed to update card {card.slug}: {e}")

    def _delete_card(self, card: Card, anki_guid: int) -> None:
        """Delete card from Anki."""
        logger.info("deleting_card", slug=card.slug, anki_guid=anki_guid)

        if self.config.delete_mode == "delete":
            # Actually delete from Anki
            self.anki.delete_notes([anki_guid])
        # else: archive mode - just remove from database

        # Remove from database
        self.db.delete_card(card.slug)

    def _create_cards_batch(self, actions: list[SyncAction]) -> None:
        """Create multiple cards in batch."""
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("creating_cards_batch", total=total, batch_size=batch_size)

        for batch_start in range(0, total, batch_size):
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            # Prepare batch payloads
            note_payloads = []
            card_data = []  # Store card, fields, tags, apf_html for DB insertion

            for action in batch_actions:
                card = action.card
                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

                note_payload = {
                    "deckName": self.config.anki_deck_name,
                    "modelName": card.note_type,
                    "fields": fields,
                    "tags": card.tags,
                    "options": {"allowDuplicate": False},
                }
                if card.guid:
                    note_payload["guid"] = card.guid

                note_payloads.append(note_payload)
                card_data.append((card, fields, card.tags, card.apf_html))

            # Batch create in Anki
            try:
                with CardTransaction(self.anki, self.db) as txn:
                    note_ids = self.anki.add_notes(note_payloads)

                    # Process results and handle partial failures
                    successful_cards = []
                    for i, (note_id, (card, fields, tags, apf_html)) in enumerate(
                        zip(note_ids, card_data)
                    ):
                        if note_id is not None:
                            # Register rollback
                            txn.rollback_actions.append(("delete_anki_note", note_id))
                            successful_cards.append(
                                (card, note_id, fields, tags, apf_html)
                            )
                        else:
                            # Failed card
                            logger.error(
                                "batch_create_failed",
                                slug=card.slug,
                                index=i,
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                            self.db.update_card_status(
                                card.slug, "failed", "Anki batch create returned None"
                            )

                    # Batch insert into database
                    if successful_cards:
                        self.db.insert_cards_batch(
                            successful_cards, self.config.anki_deck_name
                        )

                    txn.commit()

                    # Update stats
                    created_count = len(successful_cards)
                    self.stats["created"] += created_count
                    if self.progress:
                        for _ in range(created_count):
                            self.progress.increment_stat("created")

                    logger.info(
                        "batch_create_completed",
                        batch_start=batch_start,
                        batch_end=batch_end,
                        successful=created_count,
                        failed=len(batch_actions) - created_count,
                    )

                    # Verify card creation if enabled
                    if getattr(self.config, "verify_card_creation", True):
                        for card, note_id, fields, tags, _ in successful_cards:
                            self._verify_card_creation(card, note_id, fields, tags)

            except AnkiConnectError as e:
                logger.error(
                    "batch_create_failed_anki_connect",
                    error=str(e),
                    batch_start=batch_start,
                )
                # Fall back to individual creates
                for action in batch_actions:
                    try:
                        self._create_card(action.card)
                        if action.type == "restore":
                            self.stats["restored"] += 1
                            if self.progress:
                                self.progress.increment_stat("restored")
                        else:
                            self.stats["created"] += 1
                            if self.progress:
                                self.progress.increment_stat("created")
                    except CardOperationError as card_error:
                        logger.error(
                            "individual_create_failed_after_batch_operation",
                            slug=action.card.slug,
                            error=str(card_error),
                        )
                        self.stats["errors"] += 1
                        if self.progress:
                            self.progress.increment_stat("errors")
                    except AnkiConnectError as card_error:
                        logger.error(
                            "individual_create_failed_after_batch_anki",
                            slug=action.card.slug,
                            error=str(card_error),
                        )
                        self.stats["errors"] += 1
                        if self.progress:
                            self.progress.increment_stat("errors")
                    except Exception as card_error:
                        logger.error(
                            "individual_create_failed_after_batch_unexpected",
                            slug=action.card.slug,
                            error=str(card_error),
                            exc_info=True,
                        )
                        self.stats["errors"] += 1
                        if self.progress:
                            self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "batch_create_failed_unexpected",
                    error=str(e),
                    batch_start=batch_start,
                    exc_info=True,
                )
                # Fall back to individual creates
                for action in batch_actions:
                    try:
                        self._create_card(action.card)
                        if action.type == "restore":
                            self.stats["restored"] += 1
                            if self.progress:
                                self.progress.increment_stat("restored")
                        else:
                            self.stats["created"] += 1
                            if self.progress:
                                self.progress.increment_stat("created")
                    except Exception as card_error:
                        logger.error(
                            "individual_create_failed_after_batch_fallback",
                            slug=action.card.slug,
                            error=str(card_error),
                        )
                        self.stats["errors"] += 1
                        if self.progress:
                            self.progress.increment_stat("errors")

    def _update_cards_batch(self, actions: list[SyncAction]) -> None:
        """Update multiple cards in batch."""
        batch_size = self.config.batch_size
        total = len(actions)

        logger.info("updating_cards_batch", total=total, batch_size=batch_size)

        for batch_start in range(0, total, batch_size):
            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                break

            batch_end = min(batch_start + batch_size, total)
            batch_actions = actions[batch_start:batch_end]

            # Prepare batch updates
            field_updates = []
            tag_updates = []
            card_data = []  # Store card, fields, tags for DB update

            for action in batch_actions:
                if not action.anki_guid:
                    continue

                card = action.card
                fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

                field_updates.append({"id": action.anki_guid, "fields": fields})
                tag_updates.append((action.anki_guid, card.tags))
                card_data.append((card, fields, card.tags))

            if not field_updates:
                continue

            # Batch update in Anki
            try:
                with CardTransaction(self.anki, self.db) as txn:
                    # Get current state for rollback
                    anki_guids = [update["id"] for update in field_updates]
                    current_info = self.anki.notes_info(anki_guids)

                    # Register rollback actions
                    for info in current_info:
                        note_id = info["noteId"]
                        old_fields = info.get("fields", {})
                        old_tags = info.get("tags", [])
                        txn.rollback_actions.append(
                            ("restore_anki_note", note_id, old_fields, old_tags)
                        )

                    # Batch update fields
                    field_results = self.anki.update_notes_fields(field_updates)

                    # Batch update tags
                    tag_results = self.anki.update_notes_tags(tag_updates)

                    # Process results
                    successful_cards = []
                    for i, (
                        field_success,
                        tag_success,
                        (card, fields, tags),
                    ) in enumerate(zip(field_results, tag_results, card_data)):
                        if field_success and tag_success:
                            successful_cards.append((card, fields, tags))
                        else:
                            logger.error(
                                "batch_update_failed",
                                slug=card.slug,
                                field_success=field_success,
                                tag_success=tag_success,
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                            self.db.update_card_status(
                                card.slug,
                                "failed",
                                f"Batch update failed: field={field_success}, tag={tag_success}",
                            )

                    # Batch update database
                    if successful_cards:
                        self.db.update_cards_batch(successful_cards)

                    txn.commit()

                    # Update stats
                    updated_count = len(successful_cards)
                    self.stats["updated"] += updated_count
                    if self.progress:
                        for _ in range(updated_count):
                            self.progress.increment_stat("updated")

                    logger.info(
                        "batch_update_completed",
                        batch_start=batch_start,
                        batch_end=batch_end,
                        successful=updated_count,
                        failed=len(batch_actions) - updated_count,
                    )

            except AnkiConnectError as e:
                logger.error(
                    "batch_update_failed_anki_connect",
                    error=str(e),
                    batch_start=batch_start,
                )
                # Fall back to individual updates
                for action in batch_actions:
                    if action.anki_guid:
                        try:
                            self._update_card(action.card, action.anki_guid)
                            self.stats["updated"] += 1
                            if self.progress:
                                self.progress.increment_stat("updated")
                        except CardOperationError as card_error:
                            logger.error(
                                "individual_update_failed_after_batch_operation",
                                slug=action.card.slug,
                                error=str(card_error),
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                        except AnkiConnectError as card_error:
                            logger.error(
                                "individual_update_failed_after_batch_anki",
                                slug=action.card.slug,
                                error=str(card_error),
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                        except Exception as card_error:
                            logger.error(
                                "individual_update_failed_after_batch_unexpected",
                                slug=action.card.slug,
                                error=str(card_error),
                                exc_info=True,
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
            except Exception as e:
                logger.error(
                    "batch_update_failed_unexpected",
                    error=str(e),
                    batch_start=batch_start,
                    exc_info=True,
                )
                # Fall back to individual updates
                for action in batch_actions:
                    if action.anki_guid:
                        try:
                            self._update_card(action.card, action.anki_guid)
                            self.stats["updated"] += 1
                            if self.progress:
                                self.progress.increment_stat("updated")
                        except Exception as card_error:
                            logger.error(
                                "individual_update_failed_after_batch_fallback",
                                slug=action.card.slug,
                                error=str(card_error),
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")

    def _delete_cards_batch(self, actions: list[SyncAction]) -> None:
        """Delete multiple cards in batch."""
        if not actions:
            return

        # Group by delete mode
        anki_guids_to_delete = []
        slugs_to_delete = []

        for action in actions:
            if action.anki_guid:
                slugs_to_delete.append(action.card.slug)
                if self.config.delete_mode == "delete":
                    anki_guids_to_delete.append(action.anki_guid)

        # Batch delete from Anki
        if anki_guids_to_delete:
            try:
                self.anki.delete_notes(anki_guids_to_delete)
                logger.info("batch_delete_anki", count=len(anki_guids_to_delete))
            except AnkiConnectError as e:
                logger.error("batch_delete_anki_failed_connect_error", error=str(e))
                # Fall back to individual deletes
                for action in actions:
                    if action.anki_guid:
                        try:
                            self._delete_card(action.card, action.anki_guid)
                            self.stats["deleted"] += 1
                            if self.progress:
                                self.progress.increment_stat("deleted")
                        except AnkiConnectError as del_error:
                            logger.error(
                                "individual_delete_failed_after_batch_anki",
                                slug=action.card.slug,
                                error=str(del_error),
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                        except Exception as del_error:
                            logger.error(
                                "individual_delete_failed_after_batch_unexpected",
                                slug=action.card.slug,
                                error=str(del_error),
                                exc_info=True,
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                return
            except Exception as e:
                logger.error(
                    "batch_delete_anki_failed_unexpected", error=str(e), exc_info=True
                )
                # Fall back to individual deletes
                for action in actions:
                    if action.anki_guid:
                        try:
                            self._delete_card(action.card, action.anki_guid)
                            self.stats["deleted"] += 1
                            if self.progress:
                                self.progress.increment_stat("deleted")
                        except Exception as del_error:
                            logger.error(
                                "individual_delete_failed_after_batch_fallback",
                                slug=action.card.slug,
                                error=str(del_error),
                            )
                            self.stats["errors"] += 1
                            if self.progress:
                                self.progress.increment_stat("errors")
                return

        # Batch delete from database
        if slugs_to_delete:
            try:
                self.db.delete_cards_batch(slugs_to_delete)
                deleted_count = len(slugs_to_delete)
                self.stats["deleted"] += deleted_count
                if self.progress:
                    for _ in range(deleted_count):
                        self.progress.increment_stat("deleted")
                logger.info("batch_delete_db", count=deleted_count)
            except Exception as e:
                logger.error("batch_delete_db_failed", error=str(e), exc_info=True)
                # Fall back to individual deletes
                for slug in slugs_to_delete:
                    try:
                        self.db.delete_card(slug)
                    except Exception as db_error:
                        logger.warning(
                            "individual_db_delete_failed_after_batch",
                            slug=slug,
                            error=str(db_error),
                        )
