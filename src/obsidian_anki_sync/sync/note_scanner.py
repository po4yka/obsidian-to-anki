"""Note scanning component for SyncEngine.

Handles discovery, parsing, and processing of Obsidian notes.
"""

import random
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from structlog import contextvars as structlog_contextvars

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.domain.interfaces.note_scanner import (
    INoteScanner,
    IParallelProcessor,
    IQueueProcessor,
)
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.obsidian.parser import discover_notes
from obsidian_anki_sync.sync.note_archiver import NoteArchiver
from obsidian_anki_sync.sync.note_processor import SingleNoteProcessor
from obsidian_anki_sync.sync.parallel_processor import ParallelNoteProcessor
from obsidian_anki_sync.sync.queue_processor import QueueNoteProcessor
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class NoteScanner(INoteScanner):
    """Handles scanning and processing of Obsidian notes using dependency injection."""

    def __init__(
        self,
        config: Config,
        state_db: StateDB,
        card_generator: Any,  # CardGenerator - avoid circular import
        archiver: ProblematicNotesArchiver,
        progress_tracker: "ProgressTracker | None" = None,
        progress_display: Any = None,
        stats: dict[str, Any] | None = None,
        parallel_processor: IParallelProcessor | None = None,
        queue_processor: IQueueProcessor | None = None,
    ):
        """Initialize note scanner with dependency injection.

        Args:
            config: Service configuration
            state_db: State database
            card_generator: CardGenerator instance for generating cards
            archiver: ProblematicNotesArchiver for archiving failed notes
            progress_tracker: Optional progress tracker
            progress_display: Optional progress display
            stats: Statistics dictionary to update
            parallel_processor: Optional parallel processor (will be created if not provided)
            queue_processor: Optional queue processor (will be created if not provided)
        """
        self.config = config
        self.db = state_db
        self.card_generator = card_generator
        self.archiver = archiver
        self.progress = progress_tracker
        self.progress_display = progress_display
        self.stats = stats or {}

        # Initialize services with dependency injection
        self.note_archiver = NoteArchiver(
            archiver=archiver,
            batch_size=getattr(self.config, "archiver_batch_size", 64),
            fd_headroom=getattr(self.config, "archiver_min_fd_headroom", 32),
            fd_poll_interval=getattr(self.config, "archiver_fd_poll_interval", 0.05),
        )

        self.note_processor = SingleNoteProcessor(
            config=config,
            card_generator=card_generator,
            archiver=self.note_archiver,
            progress_tracker=progress_tracker,
        )

        self.parallel_processor = parallel_processor or ParallelNoteProcessor(
            config=config,
            note_processor=self.note_processor,
            progress_tracker=progress_tracker,
            stats=self.stats,
        )

        self.queue_processor = queue_processor or QueueNoteProcessor(
            config=config,
            progress_tracker=progress_tracker,
            stats=self.stats,
        )

        self.sync_run_id: str | None = None

    def scan_notes(
        self,
        sample_size: int | None = None,
        incremental: bool = False,
        qa_extractor: Any = None,
        existing_cards_for_duplicate_detection: list | None = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan Obsidian vault and generate cards using appropriate processing strategy.

        Args:
            sample_size: Optional number of notes to randomly process
            incremental: If True, only process new notes not yet in database
            qa_extractor: Optional QA extractor for LLM-based extraction
            existing_cards_for_duplicate_detection: Existing cards from Anki for duplicate detection
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """
        self._bind_sync_context()

        logger.info(
            "scanning_obsidian",
            path=str(self.config.vault_path),
            sample_size=sample_size,
            incremental=incremental,
        )

        # Discover notes
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

        # Apply sampling
        if sample_size and sample_size > 0 and len(note_files) > sample_size:
            note_files = random.sample(note_files, sample_size)
            logger.info("sampling_notes", count=sample_size)

        # Set total notes in progress tracker
        if self.progress:
            self.progress.set_total_notes(len(note_files))

        # Initialize tracking structures
        obsidian_cards: dict[str, Card] = {}
        existing_slugs: set[str] = set()
        error_by_type: defaultdict[str, int] = defaultdict(int)
        error_samples: defaultdict[str, list[str]] = defaultdict(list)

        # Choose processing strategy
        use_parallel = (
            self.config.max_concurrent_generations > 1 and len(note_files) > 1
        )
        use_queue = getattr(self.config, "enable_queue", False)

        # Enable deferred archival for parallel processing
        if use_parallel:
            self.note_archiver.set_defer_archival(True)

        try:
            if use_parallel and use_queue:
                # Use queue-based distributed processing
                result = self.queue_processor.scan_notes_with_queue(
                    note_files=note_files,
                    obsidian_cards=obsidian_cards,
                    existing_slugs=existing_slugs,
                    error_by_type=error_by_type,
                    error_samples=error_samples,
                    qa_extractor=qa_extractor,
                    on_batch_complete=on_batch_complete,
                )
            elif use_parallel:
                # Use parallel processing
                result = self.parallel_processor.scan_notes_parallel(
                    note_files=note_files,
                    obsidian_cards=obsidian_cards,
                    existing_slugs=existing_slugs,
                    error_by_type=error_by_type,
                    error_samples=error_samples,
                    qa_extractor=qa_extractor,
                    on_batch_complete=on_batch_complete,
                )
            else:
                # Use sequential processing
                result = self._scan_notes_sequential(
                    note_files=note_files,
                    obsidian_cards=obsidian_cards,
                    existing_slugs=existing_slugs,
                    error_by_type=error_by_type,
                    error_samples=error_samples,
                    qa_extractor=qa_extractor,
                    on_batch_complete=on_batch_complete,
                )

            return result

        finally:
            # Process deferred archives and disable deferral
            if use_parallel:
                self.note_archiver.set_defer_archival(False)
                self.note_archiver.process_deferred_archives()

    def set_sync_run_id(self, sync_run_id: str | None) -> None:
        """Set the sync run identifier for propagation to worker threads."""

        self.sync_run_id = sync_run_id

    def _scan_notes_sequential(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan notes using sequential processing.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs (will be updated)
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """
        logger.info("sequential_scan_started", total_notes=len(note_files))

        for file_path, relative_path in note_files:
            try:
                cards_dict, new_slugs, result_info = self.note_processor.process_note(
                    file_path=file_path,
                    relative_path=relative_path,
                    existing_slugs=existing_slugs,
                    qa_extractor=qa_extractor,
                )

                # Update tracking structures
                obsidian_cards.update(cards_dict)
                existing_slugs.update(new_slugs)

                # Handle errors from processing
                if result_info.get("errors"):
                    for error_type, count in result_info["errors"].items():
                        error_by_type[error_type] += count
                        if error_type not in error_samples:
                            error_samples[error_type] = []
                        if result_info.get("error_samples", {}).get(error_type):
                            error_samples[error_type].extend(
                                result_info["error_samples"][error_type][:3]  # Limit samples
                            )
                            error_samples[error_type] = error_samples[error_type][:10]  # Limit total samples

                # Call batch completion callback if provided
                if on_batch_complete and cards_dict:
                    on_batch_complete(list(cards_dict.values()))

                # Update progress
                if self.progress:
                    self.progress.increment_processed()

            except Exception as e:
                logger.exception("note_processing_error", path=relative_path, error=str(e))
                error_type = type(e).__name__
                error_by_type[error_type] += 1
                if error_type not in error_samples:
                    error_samples[error_type] = []
                if len(error_samples[error_type]) < 10:
                    error_samples[error_type].append(f"{relative_path}: {str(e)}")

                # Update progress even on error
                if self.progress:
                    self.progress.increment_processed()

        logger.info(
            "sequential_scan_completed",
            total_cards=len(obsidian_cards),
            total_errors=sum(error_by_type.values()),
        )

        return obsidian_cards

    def _bind_sync_context(self) -> None:
        """Bind the current sync_run_id to the logging context if available."""

        if getattr(self, "sync_run_id", None):
            structlog_contextvars.bind_contextvars(sync_run_id=self.sync_run_id)
