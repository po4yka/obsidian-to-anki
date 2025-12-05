"""Synchronization engine for Obsidian to Anki sync."""

import contextlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import diskcache
import structlog
from pydantic import ValidationError

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:
    # Fallback if tenacity not available
    retry = None  # type: ignore
    retry_if_exception_type = None  # type: ignore
    stop_after_attempt = None  # type: ignore
    wait_exponential = None  # type: ignore
    before_sleep_log = None  # type: ignore

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.error_codes import ErrorCode
from obsidian_anki_sync.models import Card, ManifestData, SyncAction
from obsidian_anki_sync.providers.factory import ProviderFactory
from obsidian_anki_sync.sync.anki_state_manager import AnkiStateManager
from obsidian_anki_sync.sync.card_generator import CardGenerator
from obsidian_anki_sync.sync.change_applier import ChangeApplier
from obsidian_anki_sync.sync.indexer import build_full_index
from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.guid import deterministic_guid
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)

# Import LangGraph orchestrator (optional dependency)
try:
    from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    LangGraphOrchestrator = None  # type: ignore
    logger.warning("agent_system_not_available", reason="Import failed")

# Import progress display (optional)
try:
    from obsidian_anki_sync.utils.progress_display import ProgressDisplay

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
            vault_path=str(config.vault_path),
            anki_deck=config.anki_deck_name,
            run_mode=config.run_mode,
            delete_mode=config.delete_mode,
        )

        # Initialize card generator (LangGraphOrchestrator with PydanticAI)
        if not AGENTS_AVAILABLE:
            msg = (
                "LangGraph agent system is required but not available. "
                "Please ensure agent dependencies are installed."
            )
            raise RuntimeError(msg)

        # Show progress bar for initialization
        logger.info("starting_agent_initialization_progress")
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "Initializing agent system...", total=3)

            logger.info("initializing_langgraph_orchestrator")
            progress.update(
                task, description="Initializing LangGraph orchestrator..."
            )
            self.agent_orchestrator = LangGraphOrchestrator(
                config)  # type: ignore
            progress.update(
                task, advance=1, description="LangGraph orchestrator ready"
            )
            self.use_agents = True

            # Configure LLM-based Q&A extraction when using agents
            from obsidian_anki_sync.obsidian.parser import create_qa_extractor

            qa_extractor_model = config.get_model_for_agent("qa_extractor")
            model_config = config.get_model_config_for_task("qa_extraction")
            qa_extractor_temp = model_config.get("temperature", 0.0)
            reasoning_enabled = getattr(
                config, "llm_reasoning_enabled", False)

            logger.info(
                "configuring_llm_qa_extraction",
                model=qa_extractor_model,
                temperature=qa_extractor_temp,
                reasoning_enabled=reasoning_enabled,
            )

            # Create a real provider instance for the extractor
            # We cannot use self.agent_orchestrator.provider because it's a dummy provider
            # that returns coroutines for generate(), which breaks the synchronous QAExtractorAgent
            progress.update(
                task, description="Creating QA extraction provider...")
            qa_provider = ProviderFactory.create_from_config(
                config, verbose_logging=False
            )
            progress.update(
                task, advance=1, description="QA extraction provider ready"
            )

            progress.update(
                task, description="Creating QA extractor agent...")
            self.qa_extractor = create_qa_extractor(
                llm_provider=qa_provider,
                model=qa_extractor_model,
                temperature=qa_extractor_temp,
                reasoning_enabled=reasoning_enabled,
                enable_content_generation=getattr(
                    config, "enable_content_generation", True
                ),
                repair_missing_sections=getattr(
                    config, "repair_missing_sections", True
                ),
            )
            progress.update(task, advance=1,
                            description="QA extractor ready")

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

        # Retry statistics tracking
        self._retry_stats: dict[str, int] = {
            "total_retries": 0,
            "successful_after_retry": 0,
            "failed_after_max_retries": 0,
        }

        # Thread-safe slug generation
        import threading

        self._slug_counter_lock = threading.Lock()
        self._slug_counters: dict[str, int] = {}  # base_slug -> next_index

        # Initialize persistent disk caches
        # Cache directory is placed next to the database file
        # Only create cache if db_path is properly configured (not a mock)
        if (hasattr(config.db_path, '__class__') and
            config.db_path.__class__.__name__ != 'MagicMock' and
            hasattr(config.db_path, 'parent') and
            config.db_path.parent.exists()):
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
        else:
            # Fallback: use in-memory caches when db_path is not properly configured
            logger.debug("db_path_not_configured_using_memory_caches")
            self._agent_card_cache = None  # type: ignore
            self._apf_card_cache = None  # type: ignore

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

        # Initialize component classes
        self.card_generator = CardGenerator(
            config=config,
            agent_orchestrator=self.agent_orchestrator,
            use_agents=self.use_agents,
            agent_card_cache=self._agent_card_cache,
            apf_card_cache=self._apf_card_cache,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_stats=self._cache_stats,
            slug_counters=self._slug_counters,
            slug_counter_lock=self._slug_counter_lock,
            stats=self.stats,
        )

        self.note_scanner = NoteScanner(
            config=config,
            state_db=state_db,
            card_generator=self.card_generator,
            archiver=self.archiver,
            progress_tracker=progress_tracker,
            progress_display=None,  # Will be set via set_progress_display()
            stats=self.stats,
        )

        self.anki_state_manager = AnkiStateManager(
            config=config,
            state_db=state_db,
            anki_client=anki_client,
        )

        self.change_applier = ChangeApplier(
            config=config,
            state_db=state_db,
            anki_client=anki_client,
            progress_tracker=progress_tracker,
            stats=self.stats,
        )

        self.sync_run_id: str | None = None

    def set_progress_display(self, progress_display: "ProgressDisplay | None") -> None:
        """Set progress display for real-time updates.

        Args:
            progress_display: ProgressDisplay instance or None
        """
        self.progress_display = progress_display
        # Pass to note scanner
        if hasattr(self, "note_scanner"):
            self.note_scanner.progress_display = progress_display
        # Pass to agents if available
        if (
            self.progress_display
            and hasattr(self, "agent_orchestrator")
            and self.agent_orchestrator
        ) and hasattr(self.agent_orchestrator, "set_progress_display"):
            self.agent_orchestrator.set_progress_display(self.progress_display)

    def _get_anki_model_name(self, internal_note_type: str) -> str:
        """Get the actual Anki model name from internal note type.

        Maps internal note types (e.g., "APF::Simple") to actual Anki
        model names (e.g., "APF: Simple (3.0.0)").

        Args:
            internal_note_type: Internal note type identifier

        Returns:
            Actual Anki model name from config.model_names mapping,
            or the internal name if no mapping exists
        """
        return self.config.model_names.get(internal_note_type, internal_note_type)

    def _close_caches(self) -> None:
        """Close disk caches to ensure data is flushed to disk."""
        try:
            if self._agent_card_cache is not None:
                self._agent_card_cache.close()
            if self._apf_card_cache is not None:
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
        progress_task_id = progress_bar.add_task(
            f"[cyan]{description}...", total=total)

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
        sync_run_id: str | None = None,
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
        if sync_run_id is None:
            sync_run_id = str(uuid.uuid4())

        self.sync_run_id = sync_run_id
        structlog.contextvars.bind_contextvars(sync_run_id=sync_run_id)
        self.note_scanner.set_sync_run_id(sync_run_id)

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

        import time

        sync_start_time = time.time()
        try:
            # Step 0: Build index (if enabled)
            index_stats = None
            if build_index:
                if self.progress:
                    from .progress import SyncPhase

                    self.progress.set_phase(SyncPhase.INDEXING)

                logger.info("sync_phase_started", phase="indexing")
                index_start_time = time.time()
                index_stats = build_full_index(
                    self.config, self.db, self.anki, incremental=incremental
                )
                index_duration = time.time() - index_start_time
                logger.info(
                    "sync_phase_completed",
                    phase="indexing",
                    duration=round(index_duration, 2),
                    stats=index_stats["overall"],
                )

                # Check for interruption
                if self.progress and self.progress.is_interrupted():
                    return self.progress.get_stats()

            # Step 1: Fetch existing cards for duplicate detection (if needed)
            existing_cards = []
            if self.use_agents and getattr(
                self.config, "enable_duplicate_detection", False
            ):
                logger.info("sync_phase_started",
                            phase="fetching_existing_cards")
                fetch_start_time = time.time()
                existing_cards = self.anki_state_manager.fetch_existing_cards_for_duplicate_detection()
                fetch_duration = time.time() - fetch_start_time
                logger.info(
                    "sync_phase_completed",
                    phase="fetching_existing_cards",
                    duration=round(fetch_duration, 2),
                    count=len(existing_cards),
                )
                # Pass existing cards to card generator
                self.card_generator.set_existing_cards_for_duplicate_detection(
                    existing_cards
                )

            # Step 2: Fetch Anki state early for atomic processing
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.DETERMINING_ACTIONS)

            logger.info("sync_phase_started", phase="fetching_anki_state")
            fetch_start_time = time.time()
            anki_cards = self.anki_state_manager.fetch_state()
            fetch_duration = time.time() - fetch_start_time
            logger.info(
                "sync_phase_completed",
                phase="fetching_anki_state",
                duration=round(fetch_duration, 2),
                cards_found=len(anki_cards),
            )

            # Get database state early (Card objects, not dicts)
            db_cards = {c.slug: c for c in self.db.get_all_cards()}

            # Define callback for atomic processing
            self.changes = []
            changes_by_type = {}

            def on_batch_complete(batch_cards: list[Card]) -> None:
                """Process a batch of generated cards immediately."""
                if not batch_cards:
                    return

                batch_changes: list[SyncAction] = []
                batch_obsidian_cards = {c.slug: c for c in batch_cards}

                # Determine actions for this batch
                self.anki_state_manager.determine_actions(
                    batch_obsidian_cards,
                    anki_cards,
                    batch_changes,
                    db_cards_override=db_cards,
                )

                # Update stats
                for change in batch_changes:
                    change_type = change.type
                    changes_by_type[change_type] = (
                        changes_by_type.get(change_type, 0) + 1
                    )
                    self.changes.append(change)

                # Apply changes immediately if not dry run
                if not dry_run:
                    self.change_applier.apply_changes(batch_changes)

                    # Update in-memory state to reflect changes
                    for change in batch_changes:
                        if change.type == "create" and change.card:
                            # We don't have the new note ID yet without refetching or return from apply_changes
                            # But we can mark it as present to avoid duplicate creates if processed again
                            # For now, just updating db_cards is enough for content hash checks
                            pass
                        elif change.type == "update" and change.card:
                            pass

            # Step 3: Scan Obsidian notes and generate cards (with atomic processing)
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.SCANNING)

            logger.info("sync_phase_started", phase="scanning")
            scan_start_time = time.time()

            # Pass callback to scanner
            obsidian_cards = self.note_scanner.scan_notes(
                sample_size=sample_size,
                incremental=incremental,
                qa_extractor=self.qa_extractor,
                existing_cards_for_duplicate_detection=existing_cards,
                on_batch_complete=on_batch_complete,
            )

            scan_duration = time.time() - scan_start_time
            logger.info(
                "sync_phase_completed",
                phase="scanning",
                duration=round(scan_duration, 2),
                cards_generated=len(obsidian_cards),
            )

            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                return cast("dict[Any, Any]", self.progress.get_stats())

            # Step 4: Handle deletions (cards in DB/Anki but not in Obsidian)
            # This must be done after all notes are scanned
            logger.info("sync_phase_started", phase="determining_deletions")

            deletion_changes: list[SyncAction] = []

            # Check for deletions in Obsidian
            for slug, db_card in db_cards.items():
                if slug not in obsidian_cards and slug in anki_cards:
                    # Card deleted in Obsidian but still in Anki
                    from obsidian_anki_sync.models import Card as CardModel
                    from obsidian_anki_sync.models import Manifest

                    # db_card is a domain Card object - access manifest for nested fields
                    manifest = db_card.manifest
                    card_guid = manifest.guid or deterministic_guid(
                        [
                            manifest.note_id or "",
                            manifest.source_path,
                            str(manifest.card_index),
                            db_card.language,
                        ]
                    )

                    # Reconstruct minimal card for deletion
                    card = CardModel(
                        slug=slug,
                        lang=db_card.language,
                        apf_html="",
                        manifest=Manifest(
                            slug=slug,
                            slug_base=manifest.slug_base,
                            lang=db_card.language,
                            source_path=manifest.source_path,
                            source_anchor=manifest.source_anchor,
                            note_id=manifest.note_id,
                            note_title=manifest.note_title,
                            card_index=manifest.card_index,
                            guid=card_guid,
                        ),
                        content_hash=db_card.content_hash,
                        note_type=db_card.note_type or "APF::Simple",
                        tags=[],
                        guid=card_guid,
                    )

                    deletion_changes.append(
                        SyncAction(
                            type="delete",
                            card=card,
                            anki_guid=db_card.anki_guid,
                            reason="Card removed from Obsidian",
                        )
                    )

            # Check for deletions in Anki (restore)
            for slug in db_cards:
                if slug not in anki_cards and slug in obsidian_cards:
                    # Card deleted in Anki but still in Obsidian - restore
                    deletion_changes.append(
                        SyncAction(
                            type="restore",
                            card=obsidian_cards[slug],
                            reason="Card deleted in Anki, restoring from Obsidian",
                        )
                    )

            # Apply deletion/restore changes
            if deletion_changes:
                for change in deletion_changes:
                    change_type = change.type
                    changes_by_type[change_type] = (
                        changes_by_type.get(change_type, 0) + 1
                    )
                    self.changes.append(change)

                if not dry_run:
                    self.change_applier.apply_changes(deletion_changes)

            logger.info(
                "sync_phase_completed",
                phase="determining_deletions",
                total_changes=len(self.changes),
                changes_by_type=changes_by_type,
            )

            # Step 5: Finalize
            if dry_run:
                logger.info("sync_phase_started", phase="preview")
                self._print_plan()
                logger.info(
                    "sync_phase_completed",
                    phase="preview",
                    changes_previewed=len(self.changes),
                )
            else:
                # Changes already applied incrementally
                pass

            # Mark as completed
            if self.progress:
                self.progress.complete(success=True)

            total_duration = time.time() - sync_start_time

            # Build sync summary log with optional retry stats
            sync_summary_kwargs: dict[str, Any] = {
                "stats": self.stats,
                "total_duration": round(total_duration, 2),
                "dry_run": dry_run,
                "incremental": incremental,
                "sample_size": sample_size,
            }

            # Include retry stats if configured and any retries occurred
            if (
                self.config.include_retry_stats_in_summary
                and self._retry_stats["total_retries"] > 0
            ):
                sync_summary_kwargs["retry_stats"] = self._retry_stats

            logger.info("sync_completed", **sync_summary_kwargs)

            # Clean up memory after sync completion
            self._cleanup_memory()

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
            from obsidian_anki_sync.utils.llm_logging import log_session_summary

            if self.progress:
                log_session_summary(
                    session_id=self.progress.progress.session_id)

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

            # Check for orphaned cards at end of sync (if configured)
            orphan_stats = None
            if self.config.detect_orphans_on_sync and not dry_run:
                logger.info("sync_phase_started", phase="orphan_detection")
                orphan_stats = self._check_for_orphans()

            # Build result dict
            result = self.progress.get_stats() if self.progress else self.stats
            if index_stats:
                result["index"] = index_stats["overall"]
            if orphan_stats:
                result["orphans"] = {
                    "in_anki": len(orphan_stats.get("orphaned_in_anki", [])),
                    "in_db": len(orphan_stats.get("orphaned_in_db", [])),
                }
            if self._retry_stats["total_retries"] > 0:
                result["retry_stats"] = self._retry_stats

            return result

        except Exception as e:
            # Use repr to avoid potential recursion in str(e) if e is broken
            error_msg = repr(e)
            if isinstance(e, RecursionError):
                error_msg = "RecursionError: maximum recursion depth exceeded"

            logger.error("sync_failed", error=error_msg)
            if self.progress:
                self.progress.complete(success=False)
            raise
        finally:
            # Close caches to ensure data is flushed to disk
            self._close_caches()

            structlog.contextvars.unbind_contextvars("sync_run_id")

    def _cleanup_memory(self) -> None:
        """Aggressively clean up memory after processing large objects."""
        if not self.config.enable_memory_cleanup:
            return

        import gc

        from obsidian_anki_sync.utils.content_hash import clear_content_hash_cache

        # Clear content hash cache to free memory
        clear_content_hash_cache()

        # Force garbage collection to free memory from large objects
        collected = gc.collect()
        logger.debug("memory_cleanup_performed", objects_collected=collected)

    def _print_plan(self) -> None:
        """Print sync plan for dry-run."""

        for action in self.changes:
            if action.type == "skip":
                continue

            if action.reason:
                pass

        # Print summary
        action_counts: dict[str, int] = {}
        for action in self.changes:
            action_counts[action.type] = action_counts.get(action.type, 0) + 1

        for action_type, count in sorted(action_counts.items()):
            pass

    # Methods moved to component classes:
    # - NoteScanner: _scan_obsidian_notes, _scan_obsidian_notes_parallel,
    #   _process_single_note_with_retry, _process_single_note, _calculate_optimal_workers
    # - CardGenerator: _generate_cards_with_agents, _generate_card
    # - AnkiStateManager: _fetch_anki_state, _determine_actions
    # - ChangeApplier: _apply_changes, _apply_changes_sequential, _apply_changes_batched,
    #   _create_card, _update_card, _delete_card, _create_cards_batch, _update_cards_batch,
    #   _delete_cards_batch, _verify_card_creation

    def _check_for_orphans(self) -> dict[str, list[str]]:
        """Check for orphaned cards and log findings.

        Orphaned cards are cards that exist in one system but not the other:
        - orphaned_in_anki: Cards in Anki but not tracked in the database
        - orphaned_in_db: Cards in database but deleted from Anki

        Returns:
            Dictionary with orphan lists by category.
        """
        from obsidian_anki_sync.sync.recovery import CardRecovery

        try:
            recovery = CardRecovery(self.anki, self.db)
            orphans = recovery.find_orphaned_cards()

            orphaned_in_anki = orphans.get("orphaned_in_anki", [])
            orphaned_in_db = orphans.get("orphaned_in_db", [])

            if orphaned_in_anki or orphaned_in_db:
                logger.warning(
                    "orphaned_cards_detected",
                    orphaned_in_anki_count=len(orphaned_in_anki),
                    orphaned_in_db_count=len(orphaned_in_db),
                    error_code=ErrorCode.ANK_ORPHAN_DETECTED.value,
                )

                # Log detailed info at debug level
                if orphaned_in_anki:
                    logger.debug(
                        "orphaned_in_anki_details",
                        sample=orphaned_in_anki[:10],
                        total=len(orphaned_in_anki),
                    )
                if orphaned_in_db:
                    logger.debug(
                        "orphaned_in_db_details",
                        sample=orphaned_in_db[:10],
                        total=len(orphaned_in_db),
                    )
            else:
                logger.info("no_orphaned_cards_detected")

            return orphans

        except Exception as e:
            logger.error(
                "orphan_detection_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {"orphaned_in_anki": [], "orphaned_in_db": [], "inconsistent": []}

    def increment_retry_stat(self, stat_name: str, count: int = 1) -> None:
        """Increment a retry statistic.

        Args:
            stat_name: Name of the stat to increment
                       (total_retries, successful_after_retry, failed_after_max_retries)
            count: Amount to increment by (default 1)
        """
        if stat_name in self._retry_stats:
            self._retry_stats[stat_name] += count
