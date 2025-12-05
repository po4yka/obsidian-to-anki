"""Parallel processing service for note scanning."""

import contextvars
import time
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    from betterconcurrent import ThreadPoolExecutor, as_completed
except ImportError:
    from concurrent.futures import ThreadPoolExecutor, as_completed

from obsidian_anki_sync.domain.interfaces.note_scanner import IParallelProcessor
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.sync.scanner_utils import calculate_optimal_workers, ThreadSafeSlugView
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class ParallelNoteProcessor(IParallelProcessor):
    """Service for parallel note processing using thread pools."""

    def __init__(
        self,
        config: Any,
        note_processor: Any,  # INoteProcessor - avoid circular import
        progress_tracker: "ProgressTracker | None" = None,
        stats: dict[str, Any] | None = None,
    ):
        """Initialize parallel processor.

        Args:
            config: Service configuration
            note_processor: Note processor instance
            progress_tracker: Optional progress tracker
            stats: Statistics dictionary to update
        """
        self.config = config
        self.note_processor = note_processor
        self.progress = progress_tracker
        self.stats = stats or {}

    def scan_notes_parallel(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan notes using parallel processing.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs (thread-safe updates)
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """
        # Calculate max workers
        if self.config.auto_adjust_workers:
            optimal_workers = calculate_optimal_workers()
            max_workers = min(
                self.config.max_concurrent_generations,
                optimal_workers,
                len(note_files),
            )
        else:
            max_workers = min(self.config.max_concurrent_generations, len(note_files))

        logger.info(
            "parallel_scan_started",
            total_notes=len(note_files),
            max_workers=max_workers,
            auto_adjust=self.config.auto_adjust_workers,
        )

        # Initialize slug counters from existing slugs
        slug_counters = {}
        slug_counter_lock = threading.Lock()
        if slug_counter_lock:
            with slug_counter_lock:
                for slug in existing_slugs:
                    slug_counters[slug] = 0

        # Thread-safe tracking
        slugs_lock = threading.Lock()
        shared_slugs = set(existing_slugs)
        stats_lock = threading.Lock()

        notes_processed = 0
        batch_start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            base_context = contextvars.copy_context()
            future_to_note = {
                executor.submit(
                    base_context.run,
                    self._process_note_with_retry,
                    file_path,
                    relative_path,
                    shared_slugs,
                    qa_extractor,
                    slugs_lock,
                ): (file_path, relative_path)
                for file_path, relative_path in note_files
            }

            for future in as_completed(future_to_note):
                if self.progress and self.progress.is_interrupted():
                    for f in future_to_note:
                        f.cancel()
                    break

                _file_path, relative_path = future_to_note[future]

                try:
                    cards, new_slugs, result_info = future.result()

                    with slugs_lock:
                        obsidian_cards.update(cards)
                        shared_slugs.update(new_slugs)
                        existing_slugs.update(new_slugs)

                    # Atomic processing callback
                    if on_batch_complete and cards:
                        on_batch_complete(list(cards.values()))

                    with stats_lock:
                        notes_processed += 1
                        if result_info["success"]:
                            self.stats["processed"] = self.stats.get("processed", 0) + 1
                        else:
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            if result_info["error_type"]:
                                error_by_type[result_info["error_type"]] += 1
                                if len(error_samples[result_info["error_type"]]) < 3:
                                    error_samples[result_info["error_type"]].append(
                                        f"{relative_path}: {result_info['error']}"
                                    )
                            # Fail-fast: abort processing in strict mode
                            if getattr(self.config, "strict_mode", True):
                                for f in future_to_note:
                                    f.cancel()
                                msg = f"Note processing failed: {result_info['error']}"
                                raise RuntimeError(msg)

                except Exception as e:
                    logger.exception(
                        "parallel_note_processing_failed",
                        file=relative_path,
                        error=str(e),
                    )
                    with stats_lock:
                        notes_processed += 1
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        error_type_name = type(e).__name__
                        error_by_type[error_type_name] += 1
                        if len(error_samples[error_type_name]) < 3:
                            error_samples[error_type_name].append(
                                f"{relative_path}: {e!s}"
                            )
                    # Fail-fast: re-raise in strict mode (cancels remaining futures)
                    if getattr(self.config, "strict_mode", True):
                        for f in future_to_note:
                            f.cancel()
                        raise

                # Progress indicator
                if notes_processed % 10 == 0 or notes_processed == len(note_files):
                    elapsed_time = time.time() - batch_start_time
                    avg_time_per_note = (
                        elapsed_time / notes_processed if notes_processed > 0 else 0
                    )
                    remaining_notes = len(note_files) - notes_processed
                    estimated_remaining = avg_time_per_note * remaining_notes

                    total_notes = len(note_files)
                    percent = (
                        f"{(notes_processed / total_notes * 100):.1f}%"
                        if total_notes > 0
                        else "0.0%"
                    )
                    logger.info(
                        "parallel_batch_progress",
                        processed=notes_processed,
                        total=total_notes,
                        percent=percent,
                        elapsed_seconds=round(elapsed_time, 1),
                        avg_seconds_per_note=round(avg_time_per_note, 2),
                        estimated_remaining_seconds=round(estimated_remaining, 1),
                        cards_generated=len(obsidian_cards),
                        active_workers=max_workers,
                    )

        logger.info(
            "parallel_scan_completed",
            notes=len(note_files),
            cards=len(obsidian_cards),
            workers_used=max_workers,
        )

        return obsidian_cards

    def _process_note_with_retry(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file with retry logic for transient errors.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            slug_lock: Lock for thread-safe slug operations

        Returns:
            Tuple of (cards_dict, new_slugs_set, result_info)
        """
        # Get retry configuration from config
        retry_config = getattr(self.config, "retry_config_parallel", {})
        max_retries = retry_config.get("max_retries", 2)
        retry_delay = retry_config.get("retry_delay", 1.0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.note_processor.process_note_with_retry(
                    file_path, relative_path, existing_slugs, qa_extractor, slug_lock
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "note_processing_retry",
                        file=relative_path,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    # Exponential backoff
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "note_processing_failed_after_retries",
                        file=relative_path,
                        attempts=max_retries + 1,
                        error=str(e),
                    )

        # All retries exhausted
        return (
            {},
            set(),
            {
                "success": False,
                "error": str(last_error) if last_error else "Unknown error",
                "error_type": type(last_error).__name__ if last_error else None,
                "cards_count": 0,
            },
        )
