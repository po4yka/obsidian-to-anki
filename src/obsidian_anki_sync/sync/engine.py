"""Synchronization engine for Obsidian to Anki sync."""

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import diskcache
from pydantic import ValidationError

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

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

from ..anki.client import AnkiClient
from ..apf.generator import APFGenerator
from ..config import Config
from ..models import ManifestData, SyncAction
from ..obsidian.parser import (
    create_qa_extractor,
)
from ..sync.anki_state_manager import AnkiStateManager
from ..sync.card_generator import CardGenerator
from ..sync.change_applier import ChangeApplier
from ..sync.indexer import build_full_index
from ..sync.note_scanner import NoteScanner
from ..sync.state_db import StateDB
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
            use_langgraph=config.use_langgraph,
            use_pydantic_ai=config.use_pydantic_ai,
            use_agent_system=config.use_agent_system,  # Legacy
            vault_path=str(config.vault_path),
            anki_deck=config.anki_deck_name,
            run_mode=config.run_mode,
            delete_mode=config.delete_mode,
        )

        # Initialize card generator (APFGenerator or LangGraphOrchestrator)
        if config.use_langgraph or config.use_pydantic_ai:
            if not AGENTS_AVAILABLE:
                raise RuntimeError(
                    "LangGraph agent system requested but not available. "
                    "Please ensure agent dependencies are installed."
                )
            logger.info("initializing_langgraph_orchestrator")
            from ..agents.langgraph import LangGraphOrchestrator
            self.agent_orchestrator = LangGraphOrchestrator(
                config
            )  # type: ignore
            # Still keep for backward compat
            self.apf_gen = APFGenerator(config)
            self.use_agents = True
        elif config.use_agent_system:
            # Legacy fallback for backward compatibility
            if not AGENTS_AVAILABLE:
                raise RuntimeError(
                    "Agent system requested but not available. "
                    "Please ensure agent dependencies are installed."
                )
            logger.warning("using_legacy_agent_system_deprecated")
            self.agent_orchestrator = AgentOrchestrator(
                config
            )  # type: ignore
            # Still keep for backward compat
            self.apf_gen = APFGenerator(config)
            self.use_agents = True

            # Configure LLM-based Q&A extraction when using agents
            # Use the same provider as the orchestrator
            # Resolve model from config (handles empty strings and presets)
            qa_extractor_model = config.get_model_for_agent("qa_extractor")
            qa_extractor_temp = getattr(
                config, "qa_extractor_temperature", None)
            if qa_extractor_temp is None:
                # Get temperature from model config if not explicitly set
                model_config = config.get_model_config_for_task(
                    "qa_extraction")
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

        # Initialize component classes
        self.card_generator = CardGenerator(
            config=config,
            apf_gen=self.apf_gen,
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
            slug_counters=self._slug_counters,
            slug_counter_lock=self._slug_counter_lock,
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
        ):
            if hasattr(self.agent_orchestrator, "set_progress_display"):
                self.agent_orchestrator.set_progress_display(
                    self.progress_display)

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

            # Step 1: Fetch existing cards for duplicate detection (if needed)
            existing_cards = []
            if self.use_agents and getattr(self.config, "enable_duplicate_detection", False):
                logger.info("fetching_existing_cards_for_duplicate_detection")
                existing_cards = self.anki_state_manager.fetch_existing_cards_for_duplicate_detection()
                logger.info("fetched_existing_cards",
                            count=len(existing_cards))
                # Pass existing cards to card generator
                self.card_generator.set_existing_cards_for_duplicate_detection(
                    existing_cards)

            # Step 2: Scan Obsidian notes and generate cards
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.SCANNING)

            obsidian_cards = self.note_scanner.scan_notes(
                sample_size=sample_size,
                incremental=incremental,
                qa_extractor=self.qa_extractor,
                existing_cards_for_duplicate_detection=existing_cards,
            )

            # Check for interruption
            if self.progress and self.progress.is_interrupted():
                return cast(dict[Any, Any], self.progress.get_stats())

            # Step 2: Fetch Anki state
            if self.progress:
                from .progress import SyncPhase

                self.progress.set_phase(SyncPhase.DETERMINING_ACTIONS)

            anki_cards = self.anki_state_manager.fetch_state()

            # Step 3: Determine sync actions
            self.changes = []
            self.anki_state_manager.determine_actions(
                obsidian_cards, anki_cards, self.changes
            )

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

                self.change_applier.apply_changes(self.changes)

            # Mark as completed
            if self.progress:
                self.progress.complete(success=True)

            logger.info("sync_completed", stats=self.stats)

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
            from ..utils.llm_logging import log_session_summary

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

            # Build result dict
            result = self.progress.get_stats() if self.progress else self.stats
            if index_stats:
                result["index"] = index_stats["overall"]

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

    def _cleanup_memory(self) -> None:
        """Aggressively clean up memory after processing large objects."""
        if not self.config.enable_memory_cleanup:
            return

        import gc

        from ..utils.content_hash import clear_content_hash_cache

        # Clear content hash cache to free memory
        clear_content_hash_cache()

        # Force garbage collection to free memory from large objects
        collected = gc.collect()
        logger.debug("memory_cleanup_performed", objects_collected=collected)

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

    # Legacy methods removed - functionality moved to component classes:
    # - NoteScanner: _scan_obsidian_notes, _scan_obsidian_notes_parallel,
    #   _process_single_note_with_retry, _process_single_note, _calculate_optimal_workers
    # - CardGenerator: _generate_cards_with_agents, _generate_card
    # - AnkiStateManager: _fetch_anki_state, _determine_actions
    # - ChangeApplier: _apply_changes, _apply_changes_sequential, _apply_changes_batched,
    #   _create_card, _update_card, _delete_card, _create_cards_batch, _update_cards_batch,
    #   _delete_cards_batch, _verify_card_creation
