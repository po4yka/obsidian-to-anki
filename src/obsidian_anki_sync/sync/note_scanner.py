"""Note scanning component for SyncEngine.

Handles discovery, parsing, and processing of Obsidian notes.
"""

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from betterconcurrent import ThreadPoolExecutor, as_completed
except ImportError:
    from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

import yaml  # type: ignore

from ..config import Config
from ..exceptions import ParserError
from ..models import Card
from ..obsidian.parser import discover_notes, parse_note
from ..sync.state_db import StateDB
from ..utils.logging import get_logger
from ..utils.problematic_notes import ProblematicNotesArchiver

if TYPE_CHECKING:
    from ..sync.progress import ProgressTracker

logger = get_logger(__name__)


class NoteScanner:
    """Handles scanning and processing of Obsidian notes."""

    def __init__(
        self,
        config: Config,
        state_db: StateDB,
        card_generator: Any,  # CardGenerator - avoid circular import
        archiver: ProblematicNotesArchiver,
        progress_tracker: "ProgressTracker | None" = None,
        progress_display: Any = None,
        stats: dict[str, Any] | None = None,
        slug_counters: dict[str, int] | None = None,
        slug_counter_lock: Any = None,
    ):
        """Initialize note scanner.

        Args:
            config: Service configuration
            state_db: State database
            card_generator: CardGenerator instance for generating cards
            archiver: ProblematicNotesArchiver for archiving failed notes
            progress_tracker: Optional progress tracker
            progress_display: Optional progress display
            stats: Statistics dictionary to update
            slug_counters: Thread-safe slug counters dict
            slug_counter_lock: Lock for slug counters
        """
        self.config = config
        self.db = state_db
        self.card_generator = card_generator
        self.archiver = archiver
        self.progress = progress_tracker
        self.progress_display = progress_display
        self.stats = stats or {}
        self._slug_counters = slug_counters or {}
        self._slug_counter_lock = slug_counter_lock

    def scan_notes(
        self,
        sample_size: int | None = None,
        incremental: bool = False,
        qa_extractor: Any = None,
    ) -> dict[str, Card]:
        """Scan Obsidian vault and generate cards.

        Args:
            sample_size: Optional number of notes to randomly process
            incremental: If True, only process new notes not yet in database
            qa_extractor: Optional QA extractor for LLM-based extraction

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
            self.progress.set_total_notes(len(note_files))

        obsidian_cards: dict[str, Card] = {}
        existing_slugs: set[str] = set()

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
            return self.scan_notes_parallel(
                note_files,
                obsidian_cards,
                existing_slugs,
                error_by_type,
                error_samples,
                qa_extractor,
            )

        # Sequential processing
        consecutive_errors = 0
        max_consecutive_errors = 3

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
                # Parse note
                metadata, qa_pairs = parse_note(
                    file_path, qa_extractor=qa_extractor)

                # Read full note content if using agent system
                note_content = ""
                use_agents = getattr(self.config, "use_agent_system", False)
                if use_agents:
                    try:
                        note_content = file_path.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError) as e:
                        logger.warning(
                            "failed_to_read_note_content",
                            file=relative_path,
                            error=str(e),
                        )

                logger.debug(
                    "processing_note", file=relative_path, pairs=len(qa_pairs)
                )

                # Generate cards for each Q/A pair and language
                for qa_pair in qa_pairs:
                    for lang in metadata.language_tags:
                        # Check if already processed
                        if self.progress and self.progress.is_note_completed(
                            relative_path, qa_pair.card_index, lang
                        ):
                            continue

                        # Skip previously failed notes on resume
                        if self.progress and self.progress.is_note_failed(
                            relative_path, qa_pair.card_index, lang
                        ):
                            logger.info(
                                "skipping_failed_note",
                                path=relative_path,
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
                            card = self.card_generator.generate_card(
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
                            self._archive_note_safely(
                                file_path=file_path,
                                relative_path=relative_path,
                                error=e,
                                processing_stage="card_generation",
                                note_content=note_content if note_content else None,
                                card_index=qa_pair.card_index,
                                language=lang,
                            )

                            error_by_type[error_type_name] += 1
                            if len(error_samples[error_type_name]) < 3:
                                error_samples[error_type_name].append(
                                    f"{relative_path} (pair {qa_pair.card_index}, {lang}): {error_message}"
                                )

                            self.stats["errors"] = self.stats.get(
                                "errors", 0) + 1
                            consecutive_errors += 1

                            if self.progress:
                                self.progress.fail_note(
                                    relative_path,
                                    qa_pair.card_index,
                                    lang,
                                    error_message,
                                )

                self.stats["processed"] = self.stats.get("processed", 0) + 1

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
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                consecutive_errors += 1
                self._archive_note_safely(
                    file_path=file_path,
                    relative_path=relative_path,
                    error=e,
                    processing_stage="parsing",
                )
            except Exception as e:
                logger.exception(
                    "unexpected_parsing_error",
                    file=relative_path,
                    error_type=type(e).__name__,
                    error_msg=str(e),
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                consecutive_errors += 1
                self._archive_note_safely(
                    file_path=file_path,
                    relative_path=relative_path,
                    error=e,
                    processing_stage="processing",
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

                total_notes = len(note_files)
                percent = (
                    f"{(notes_processed / total_notes * 100):.1f}%"
                    if total_notes > 0
                    else "0.0%"
                )
                logger.info(
                    "batch_progress",
                    processed=notes_processed,
                    total=total_notes,
                    percent=percent,
                    elapsed_seconds=round(elapsed_time, 1),
                    avg_seconds_per_note=round(avg_time_per_note, 2),
                    estimated_remaining_seconds=round(estimated_remaining, 1),
                    cards_generated=len(obsidian_cards),
                )

        # Log if sync was terminated early
        if consecutive_errors >= max_consecutive_errors:
            logger.error(
                "sync_terminated_early",
                reason="too_many_consecutive_errors",
                consecutive_errors=consecutive_errors,
                notes_processed=notes_processed,
                total_notes=len(note_files),
            )

        # Log aggregated error summary
        if error_by_type:
            logger.warning(
                "card_generation_errors_summary",
                total_errors=self.stats.get("errors", 0),
                error_breakdown=error_by_type,
            )
            for err_type, samples in error_samples.items():
                for i, sample in enumerate(samples):
                    logger.warning(
                        "error_sample",
                        error_type=err_type,
                        sample_num=i + 1,
                        error=sample,
                    )

        logger.info(
            "obsidian_scan_completed",
            notes=len(note_files),
            cards=len(obsidian_cards),
        )

        return obsidian_cards

    def scan_notes_parallel(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: defaultdict[str, list[str]],
        qa_extractor: Any = None,
    ) -> dict[str, Card]:
        """Scan Obsidian notes using parallel processing.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs (thread-safe updates)
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor

        Returns:
            Dict of slug -> Card
        """
        # Calculate max workers
        if self.config.auto_adjust_workers:
            optimal_workers = self.calculate_optimal_workers()
            max_workers = min(
                self.config.max_concurrent_generations,
                optimal_workers,
                len(note_files),
            )
        else:
            max_workers = min(
                self.config.max_concurrent_generations, len(note_files)
            )

        logger.info(
            "parallel_scan_started",
            total_notes=len(note_files),
            max_workers=max_workers,
            auto_adjust=self.config.auto_adjust_workers,
        )

        # Initialize slug counters from existing slugs
        if self._slug_counter_lock:
            with self._slug_counter_lock:
                for slug in existing_slugs:
                    self._slug_counters[slug] = 0

        # Thread-safe tracking
        import threading

        slugs_lock = threading.Lock()
        shared_slugs = set(existing_slugs)
        stats_lock = threading.Lock()

        notes_processed = 0
        batch_start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_note = {
                executor.submit(
                    self.process_note_with_retry,
                    file_path,
                    relative_path,
                    shared_slugs,
                    qa_extractor,
                ): (file_path, relative_path)
                for file_path, relative_path in note_files
            }

            for future in as_completed(future_to_note):
                if self.progress and self.progress.is_interrupted():
                    for f in future_to_note:
                        f.cancel()
                    break

                file_path, relative_path = future_to_note[future]

                try:
                    cards, new_slugs, result_info = future.result()

                    with slugs_lock:
                        obsidian_cards.update(cards)
                        shared_slugs.update(new_slugs)
                        existing_slugs.update(new_slugs)

                    with stats_lock:
                        notes_processed += 1
                        if result_info["success"]:
                            self.stats["processed"] = self.stats.get(
                                "processed", 0) + 1
                        else:
                            self.stats["errors"] = self.stats.get(
                                "errors", 0) + 1
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
                    with stats_lock:
                        notes_processed += 1
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        error_type_name = type(e).__name__
                        error_by_type[error_type_name] += 1
                        if len(error_samples[error_type_name]) < 3:
                            error_samples[error_type_name].append(
                                f"{relative_path}: {str(e)}"
                            )

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
                        estimated_remaining_seconds=round(
                            estimated_remaining, 1),
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

    def process_note_with_retry(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
        qa_extractor: Any = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file with retry logic for transient errors.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor

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
                return self.process_note(
                    file_path, relative_path, existing_slugs, qa_extractor
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

    def process_note(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
        qa_extractor: Any = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file and generate cards.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor

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
            # Read full note content if using agent system
            note_content = ""
            use_agents = getattr(self.config, "use_agent_system", False)
            if use_agents:
                try:
                    file_size = file_path.stat().st_size
                    max_content_size = int(
                        self.config.max_note_content_size_mb * 1024 * 1024
                    )

                    if file_size > max_content_size:
                        logger.warning(
                            "note_content_too_large_skipping_agents",
                            file=relative_path,
                            size_mb=round(file_size / (1024 * 1024), 2),
                            max_size_mb=self.config.max_note_content_size_mb,
                        )
                        use_agents = False
                    else:
                        note_content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as e:
                    logger.warning(
                        "failed_to_read_note_content",
                        file=relative_path,
                        error=str(e),
                    )

            # Parse note
            metadata, qa_pairs = parse_note(
                file_path, qa_extractor=qa_extractor)

            # Generate cards for each Q/A pair and language
            for qa_pair in qa_pairs:
                for lang in metadata.language_tags:
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

                    try:
                        card = self.card_generator.generate_card(
                            qa_pair=qa_pair,
                            metadata=metadata,
                            relative_path=relative_path,
                            lang=lang,
                            existing_slugs=existing_slugs.copy(),
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

                        # Archive problematic note
                        self._archive_note_safely(
                            file_path=file_path,
                            relative_path=relative_path,
                            error=e,
                            processing_stage="card_generation",
                            note_content=note_content if note_content else None,
                            card_index=qa_pair.card_index,
                            language=lang,
                        )

                        if self.progress:
                            self.progress.fail_note(
                                relative_path, qa_pair.card_index, lang, error_message
                            )

                        result_info["error"] = error_message
                        result_info["error_type"] = error_type_name

            result_info["success"] = True
            result_info["cards_count"] = len(cards)

        except (
            ParserError,
            yaml.YAMLError,
            OSError,
            UnicodeDecodeError,
        ) as e:
            self._archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="parsing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        except Exception as e:
            self._archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="processing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        return cards, new_slugs, result_info

    def calculate_optimal_workers(self) -> int:
        """Calculate optimal worker count based on system resources.

        Returns:
            Optimal number of workers (at least 1, at most CPU count * 2)
        """
        if psutil is None:
            import os

            return max(1, os.cpu_count() or 4)

        try:
            cpu_count = psutil.cpu_count(logical=True) or 4
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=0.1)

            if cpu_percent > 80:
                base_workers = max(1, int(cpu_count * 0.5))
            elif cpu_percent > 50:
                base_workers = max(1, int(cpu_count * 0.75))
            else:
                base_workers = cpu_count

            memory_based_workers = max(1, int(available_memory_mb / 200))
            optimal = min(base_workers, memory_based_workers, cpu_count * 2)

            logger.debug(
                "optimal_workers_calculated",
                cpu_count=cpu_count,
                cpu_percent=cpu_percent,
                available_memory_mb=int(available_memory_mb),
                memory_based_workers=memory_based_workers,
                optimal_workers=optimal,
            )

            return max(1, optimal)

        except Exception as e:
            logger.warning(
                "failed_to_calculate_optimal_workers",
                error=str(e),
                note="Falling back to CPU count",
            )
            import os

            return max(1, os.cpu_count() or 4)

    def _archive_note_safely(
        self,
        file_path: Path,
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Safely archive a problematic note.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred
            note_content: Optional note content
            card_index: Optional card index
            language: Optional language
        """
        try:
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

            self.archiver.archive_note(
                note_path=file_path,
                error=error,
                error_type=type(error).__name__,
                processing_stage=processing_stage,
                card_index=card_index,
                language=language,
                note_content=note_content if note_content else None,
                context={"relative_path": relative_path},
            )
        except Exception as archive_error:
            logger.warning(
                "failed_to_archive_problematic_note",
                note_path=str(file_path),
                archive_error=str(archive_error),
            )
