"""Note scanning component for SyncEngine.

Handles discovery, parsing, and processing of Obsidian notes.
"""

import errno
import random
import time
from collections import defaultdict
from collections.abc import Collection
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

import asyncio

import yaml  # type: ignore
from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import ConfigurationError, ParserError
from obsidian_anki_sync.models import Card, QAPair
from obsidian_anki_sync.obsidian.parser import discover_notes, parse_note
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.fs_monitor import has_fd_headroom
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class _ThreadSafeSlugView(Collection[str]):
    """Lightweight, optionally locked view over a shared slug set."""

    def __init__(self, slugs: set[str], lock: Any | None = None):
        self._slugs = slugs
        self._lock = lock

    def __contains__(self, item: object) -> bool:  # pragma: no cover - trivial
        if self._lock:
            with self._lock:
                return item in self._slugs
        return item in self._slugs

    def __iter__(self):  # pragma: no cover - trivial
        if self._lock:
            with self._lock:
                return iter(self._slugs.copy())
        return iter(self._slugs)

    def __len__(self) -> int:  # pragma: no cover - trivial
        if self._lock:
            with self._lock:
                return len(self._slugs)
        return len(self._slugs)


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

        # Deferred archival system to prevent "too many open files" during parallel scans
        import threading

        self._archival_lock = threading.Lock()
        self._deferred_archives: list[dict] = []
        self._defer_archival = False  # When True, archival is deferred to end of scan
        self._archiver_batch_size = max(
            1, getattr(self.config, "archiver_batch_size", 64)
        )
        self._archiver_fd_headroom = max(
            1, getattr(self.config, "archiver_min_fd_headroom", 32)
        )
        self._archiver_fd_poll_interval = max(
            0.01, getattr(self.config, "archiver_fd_poll_interval", 0.05)
        )

    def scan_notes(
        self,
        sample_size: int | None = None,
        incremental: bool = False,
        qa_extractor: Any = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> dict[str, Card]:
        """Scan Obsidian vault and generate cards.

        Args:
            sample_size: Optional number of notes to randomly process
            incremental: If True, only process new notes not yet in database
            qa_extractor: Optional QA extractor for LLM-based extraction
            existing_cards_for_duplicate_detection: Existing cards from Anki for duplicate detection

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
            if getattr(self.config, "enable_queue", False):
                return self.scan_notes_with_queue(
                    note_files,
                    obsidian_cards,
                    existing_slugs,
                    error_by_type,
                    error_samples,
                    qa_extractor,
                )
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
                use_agents = (
                    getattr(self.config, "use_langgraph", False)
                    or getattr(self.config, "use_pydantic_ai", False)
                )
                note_content: str | None = None
                if use_agents:
                    try:
                        note_content = file_path.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError) as e:
                        logger.warning(
                            "failed_to_read_note_content",
                            file=relative_path,
                            error=str(e),
                        )
                        note_content = None

                # Parse note using preloaded content when available to avoid extra I/O
                metadata, qa_pairs = parse_note(
                    file_path, qa_extractor=qa_extractor, content=note_content
                )
                note_content = note_content or ""

                logger.debug("processing_note",
                             file=relative_path, pairs=len(qa_pairs))

                tasks = [
                    (qa_pair, lang)
                    for qa_pair in qa_pairs
                    for lang in metadata.language_tags
                    if not (
                        self.progress
                        and self.progress.is_note_completed(
                            relative_path, qa_pair.card_index, lang
                        )
                    )
                ]

                def _generate_single(
                    qa_pair: QAPair,
                    lang: str,
                    relative_path: str = relative_path,
                    metadata: Any = metadata,
                    note_content: str = note_content,
                    qa_pairs: list[QAPair] = qa_pairs,
                    existing_slugs: set[str] = existing_slugs,
                    file_path: Any = file_path,
                ):
                    if self.progress:
                        if self.progress.is_note_failed(
                            relative_path, qa_pair.card_index, lang
                        ):
                            logger.info(
                                "skipping_failed_note",
                                path=relative_path,
                                card_index=qa_pair.card_index,
                                lang=lang,
                            )
                            return None, None, None
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
                        if self.progress:
                            self.progress.complete_note(
                                relative_path, qa_pair.card_index, lang, 1
                            )
                        return card, None, None
                    except Exception as e:  # pragma: no cover - network/LLM
                        error_type_name = type(e).__name__
                        error_message = str(e)

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
                                relative_path,
                                qa_pair.card_index,
                                lang,
                                error_message,
                            )

                        return None, error_message, error_type_name

                # Execute card generation sequentially to avoid nested thread pools
                # The outer loop (scan_notes_parallel) already handles concurrency at the note level
                results = [
                    _generate_single(qa_pair, lang) for qa_pair, lang in tasks
                ]

                failures_this_note = 0
                for (qa_pair, lang), (card, error_message, error_type) in zip(
                    tasks, results
                ):
                    if card:
                        obsidian_cards[card.slug] = card
                        existing_slugs.add(card.slug)
                        consecutive_errors = 0
                    elif error_message and error_type:
                        error_by_type[error_type] += 1
                        if len(error_samples[error_type]) < 3:
                            error_samples[error_type].append(
                                f"{relative_path} (pair {qa_pair.card_index}, {lang}): {error_message}"
                            )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        failures_this_note += 1

                consecutive_errors += failures_this_note

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
                self.config.max_concurrent_generations, len(note_files))

        logger.info(
            "parallel_scan_started",
            total_notes=len(note_files),
            max_workers=max_workers,
            auto_adjust=self.config.auto_adjust_workers,
        )

        # Enable deferred archival to prevent "too many open files" during parallel scan
        self._defer_archival = True

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
                                f"{relative_path}: {e!s}"
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

        # Disable deferred archival and process queued archives sequentially
        self._defer_archival = False
        self._process_deferred_archives()

        return obsidian_cards

    def scan_notes_with_queue(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: defaultdict[str, list[str]],
        qa_extractor: Any = None,
    ) -> dict[str, Card]:
        """Scan Obsidian notes using Redis queue for distribution.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor

        Returns:
            Dict of slug -> Card
        """
        import asyncio

        # We need to run the async queue logic from a sync context
        # This helper function handles the async loop
        def _run_queue_scan():
            return asyncio.run(self._scan_notes_with_queue_async(
                note_files,
                obsidian_cards,
                existing_slugs,
                error_by_type,
                error_samples,
                qa_extractor
            ))

        return _run_queue_scan()

    async def _validate_redis_connection(self, pool) -> bool:
        """Validate Redis connection is healthy before submitting jobs.

        Args:
            pool: ArqRedis pool instance

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            await pool.ping()
            logger.info("redis_connection_healthy")
            return True
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            return False

    async def _enqueue_with_retry(
        self,
        pool,
        func_name: str,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        **kwargs,
    ):
        """Enqueue a job with retry logic and exponential backoff.

        Args:
            pool: ArqRedis pool instance
            func_name: Name of the worker function to call
            *args: Arguments for the worker function
            max_retries: Maximum retry attempts
            initial_delay: Initial delay between retries
            **kwargs: Keyword arguments including _job_id

        Returns:
            Job instance or None if job already exists
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await pool.enqueue_job(func_name, *args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    # Add jitter to prevent thundering herd
                    jitter = delay * random.uniform(-0.1, 0.1)
                    wait_time = delay + jitter
                    logger.warning(
                        "enqueue_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=round(wait_time, 2),
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    delay = min(delay * 2, 30.0)  # Exponential backoff, max 30s
                else:
                    logger.error(
                        "enqueue_retry_exhausted",
                        attempts=max_retries + 1,
                        error=str(e),
                    )
                    raise

        if last_exception:
            raise last_exception

        # Should never reach here, but satisfy linter
        return None

    async def _scan_notes_with_queue_async(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: defaultdict[str, list[str]],
        qa_extractor: Any = None,
    ) -> dict[str, Card]:
        """Async implementation of queue scanning with stability improvements.

        Features:
        - Redis health check before job submission
        - Retry logic with exponential backoff for job submission
        - Circuit breaker pattern for Redis failures
        - Overall and per-job timeouts
        - Adaptive polling with backoff
        """
        logger.info(
            "queue_scan_started",
            total_notes=len(note_files),
            redis_url=self.config.redis_url
        )

        redis_settings = RedisSettings.from_dsn(self.config.redis_url)
        pool = await create_pool(redis_settings)

        # Validate Redis connection before proceeding
        if not await self._validate_redis_connection(pool):
            await pool.close()
            msg = (
                f"Redis unavailable at {self.config.redis_url}. "
                "Check that Redis is running and accessible."
            )
            raise ConfigurationError(msg)

        job_map: dict[str, tuple[Any, str]] = {}  # job_id -> (file_path, relative_path)
        job_submit_times: dict[str, float] = {}  # job_id -> submission timestamp

        # Circuit breaker state for Redis operations
        consecutive_failures = 0
        circuit_breaker_threshold = getattr(
            self.config, "queue_circuit_breaker_threshold", 3
        )

        logger.info("submitting_jobs", total_files=len(note_files))

        # Submit all jobs with retry logic
        for file_path, relative_path in note_files:
            # Check circuit breaker
            if consecutive_failures >= circuit_breaker_threshold:
                logger.error(
                    "circuit_breaker_open",
                    consecutive_failures=consecutive_failures,
                    threshold=circuit_breaker_threshold,
                )
                break

            try:
                # Sanitize job ID to avoid issues with slashes
                job_id = f"note-{relative_path.replace('/', '_').replace('\\', '_')}"

                # Enqueue with retry logic
                job = await self._enqueue_with_retry(
                    pool,
                    "process_note_job",
                    str(file_path),
                    relative_path,
                    max_retries=getattr(self.config, "queue_max_retries", 3),
                    _job_id=job_id,
                )

                submit_time = time.time()
                if job:
                    job_map[job.job_id] = (file_path, relative_path)
                    job_submit_times[job.job_id] = submit_time
                    logger.debug("job_enqueued", job_id=job.job_id, file=relative_path)
                else:
                    # Job with this ID already exists in Redis
                    job_map[job_id] = (file_path, relative_path)
                    job_submit_times[job_id] = submit_time
                    logger.debug(
                        "job_already_exists_tracking", file=relative_path, job_id=job_id
                    )

                # Reset circuit breaker on success
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    "queue_submission_failed",
                    file=relative_path,
                    error=str(e),
                    consecutive_failures=consecutive_failures,
                )
                self.stats["errors"] = self.stats.get("errors", 0) + 1

        logger.info("jobs_submitted", count=len(job_map))

        if not job_map:
            logger.warning("no_jobs_submitted")
            await pool.close()
            return obsidian_cards

        # Poll for results with timeouts and adaptive polling
        pending_jobs = set(job_map.keys())
        completed_jobs = 0

        # Timeout configuration
        max_wait_time = getattr(self.config, "queue_max_wait_time_seconds", 18000)
        job_timeout = getattr(self.config, "queue_job_timeout_seconds", 3600)
        poll_start_time = time.time()

        # Adaptive polling configuration
        poll_interval = getattr(self.config, "queue_poll_interval", 0.5)
        poll_max_interval = getattr(self.config, "queue_poll_max_interval", 5.0)
        last_progress_time = time.time()
        no_progress_threshold = 30  # seconds before increasing poll interval

        while pending_jobs:
            # Check overall timeout
            elapsed_total = time.time() - poll_start_time
            if elapsed_total > max_wait_time:
                logger.error(
                    "queue_polling_timeout",
                    pending_count=len(pending_jobs),
                    elapsed=elapsed_total,
                    max_wait=max_wait_time,
                )
                for job_id in pending_jobs:
                    file_path, relative_path = job_map[job_id]
                    logger.error("job_stuck_at_timeout", job_id=job_id, file=relative_path)
                    self.stats["errors"] = self.stats.get("errors", 0) + 1
                break

            await asyncio.sleep(poll_interval)

            done_this_loop = set()
            progress_made = False

            for job_id in list(pending_jobs):
                # Check per-job timeout
                job_elapsed = time.time() - job_submit_times.get(job_id, poll_start_time)
                if job_elapsed > job_timeout:
                    file_path, relative_path = job_map[job_id]
                    logger.error(
                        "job_timeout",
                        job_id=job_id,
                        file=relative_path,
                        elapsed=job_elapsed,
                        timeout=job_timeout,
                    )
                    self.stats["errors"] = self.stats.get("errors", 0) + 1
                    error_by_type["job_timeout"] += 1
                    if len(error_samples["job_timeout"]) < 3:
                        error_samples["job_timeout"].append(
                            f"{relative_path}: timed out after {job_elapsed:.0f}s"
                        )
                    done_this_loop.add(job_id)
                    continue

                try:
                    job = Job(job_id=job_id, redis=pool)
                    status = await job.status()
                except Exception as e:
                    logger.warning("job_status_check_failed", job_id=job_id, error=str(e))
                    continue

                if status == "complete":
                    progress_made = True
                    try:
                        result = await job.result()
                    except Exception as e:
                        logger.error(
                            "job_result_fetch_failed", job_id=job_id, error=str(e)
                        )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        done_this_loop.add(job_id)
                        completed_jobs += 1
                        continue

                    file_path, relative_path = job_map[job_id]

                    if result.get("success"):
                        for card_dict in result.get("cards", []):
                            try:
                                card = Card(**card_dict)
                                obsidian_cards[card.slug] = card
                                existing_slugs.add(card.slug)
                            except Exception as e:
                                logger.error(
                                    "card_rehydration_failed",
                                    slug=card_dict.get("slug"),
                                    error=str(e),
                                )
                        self.stats["processed"] = self.stats.get("processed", 0) + 1
                    else:
                        error_msg = result.get("error", "Unknown error")
                        logger.warning(
                            "job_failed_result", file=relative_path, error=error_msg
                        )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                        error_by_type["queue_error"] += 1
                        if len(error_samples["queue_error"]) < 3:
                            error_samples["queue_error"].append(
                                f"{relative_path}: {error_msg}"
                            )

                    done_this_loop.add(job_id)
                    completed_jobs += 1

                elif status in ("failed", "not_found"):
                    progress_made = True
                    file_path, relative_path = job_map[job_id]
                    logger.error(
                        "job_execution_failed", file=relative_path, status=status
                    )
                    self.stats["errors"] = self.stats.get("errors", 0) + 1
                    done_this_loop.add(job_id)
                    completed_jobs += 1

            pending_jobs -= done_this_loop

            # Adaptive polling: backoff if no progress
            if progress_made:
                poll_interval = getattr(self.config, "queue_poll_interval", 0.5)
                last_progress_time = time.time()
            elif time.time() - last_progress_time > no_progress_threshold:
                old_interval = poll_interval
                poll_interval = min(poll_interval * 1.5, poll_max_interval)
                if poll_interval != old_interval:
                    logger.debug(
                        "polling_backoff",
                        old_interval=old_interval,
                        new_interval=poll_interval,
                    )

            if done_this_loop:
                logger.info(
                    "queue_progress", completed=completed_jobs, total=len(job_map)
                )

        await pool.close()
        return obsidian_cards

    def process_note_with_retry(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file with retry logic for transient errors.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            existing_cards_for_duplicate_detection: Existing cards from Anki for duplicate detection

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
                # Ensure we have FD headroom before starting a new note processing task
                self._wait_for_fd_headroom()

                return self.process_note(
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

    def process_note(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: set[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file and generate cards.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            existing_cards_for_duplicate_detection: Existing cards from Anki for duplicate detection

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
            use_agents = (
                getattr(self.config, "use_langgraph", False)
                or getattr(self.config, "use_pydantic_ai", False)
            )
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
                    if isinstance(e, OSError) and e.errno in (errno.EMFILE, errno.ENFILE):
                        raise
                    logger.warning(
                        "failed_to_read_note_content",
                        file=relative_path,
                        error=str(e),
                    )

            # Parse note
            metadata, qa_pairs = parse_note(
                file_path, qa_extractor=qa_extractor, content=note_content or None
            )

            slug_view: Collection[str] = _ThreadSafeSlugView(
                existing_slugs, slug_lock)

            tasks = [
                (qa_pair, lang)
                for qa_pair in qa_pairs
                for lang in metadata.language_tags
                if not (
                    self.progress
                    and self.progress.is_note_completed(
                        relative_path, qa_pair.card_index, lang
                    )
                )
            ]

            max_workers = max(
                1, min(self.config.max_concurrent_generations, len(tasks))
            )

            def _generate_single(
                qa_pair: QAPair,
                lang: str,
                relative_path: str = relative_path,
                metadata: Any = metadata,
                note_content: str = note_content,
                qa_pairs: list[QAPair] = qa_pairs,
                file_path: Any = file_path,
                slug_view: Collection[str] = slug_view,
            ):
                if self.progress:
                    self.progress.start_note(
                        relative_path, qa_pair.card_index, lang)
                try:
                    card = self.card_generator.generate_card(
                        qa_pair=qa_pair,
                        metadata=metadata,
                        relative_path=relative_path,
                        lang=lang,
                        existing_slugs=slug_view,
                        note_content=note_content,
                        all_qa_pairs=qa_pairs,
                    )
                    if self.progress:
                        self.progress.complete_note(
                            relative_path, qa_pair.card_index, lang, 1
                        )
                    return card, None, None
                except Exception as e:  # pragma: no cover - network/LLM
                    error_type_name = type(e).__name__
                    error_message = str(e)

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

                    return None, error_message, error_type_name

            if max_workers == 1:
                for qa_pair, lang in tasks:
                    card, error_message, error_type_name = _generate_single(
                        qa_pair, lang
                    )
                    if card:
                        cards[card.slug] = card
                        new_slugs.add(card.slug)
                    elif error_message:
                        result_info["error"] = error_message
                        result_info["error_type"] = error_type_name
            else:
                import threading

                lock = threading.Lock()
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ctx = {
                        executor.submit(_generate_single, qa_pair, lang): (
                            qa_pair,
                            lang,
                        )
                        for qa_pair, lang in tasks
                    }
                    for future in as_completed(future_to_ctx):
                        card, error_message, error_type_name = future.result()
                        with lock:
                            if card:
                                cards[card.slug] = card
                                new_slugs.add(card.slug)
                            elif error_message:
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
            if isinstance(e, OSError) and e.errno in (errno.EMFILE, errno.ENFILE):
                raise
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

        When _defer_archival is True (during parallel scans), archival requests
        are queued and processed sequentially after the scan completes.
        This prevents "too many open files" errors from concurrent file operations.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred
            note_content: Optional note content
            card_index: Optional card index
            language: Optional language
        """
        # During parallel scans, defer archival to prevent file descriptor exhaustion
        if self._defer_archival:
            with self._archival_lock:
                self._deferred_archives.append(
                    {
                        "file_path": file_path,
                        "relative_path": relative_path,
                        "error": error,
                        "processing_stage": processing_stage,
                        "note_content": note_content,
                        "card_index": card_index,
                        "language": language,
                    }
                )
            return

        # Immediate archival (non-parallel mode)
        self._archive_note_immediate(
            file_path=file_path,
            relative_path=relative_path,
            error=error,
            processing_stage=processing_stage,
            note_content=note_content,
            card_index=card_index,
            language=language,
        )

    def _archive_note_immediate(
        self,
        file_path: Path,
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Immediately archive a problematic note.

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
                    note_content = None

            # Retry logic for FD exhaustion
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    self.archiver.archive_note(
                        note_path=file_path,
                        error=error,
                        error_type=type(error).__name__,
                        processing_stage=processing_stage,
                        card_index=card_index,
                        language=language,
                        note_content=note_content if note_content is not None else None,
                        context={"relative_path": relative_path},
                    )
                    return  # Success
                except OSError as e:
                    # Check for "Too many open files" (EMFILE) or "File table overflow" (ENFILE)
                    if e.errno in (errno.EMFILE, errno.ENFILE):
                        if attempt < max_retries:
                            logger.warning(
                                "archival_fd_exhaustion_retry",
                                file=relative_path,
                                attempt=attempt + 1,
                                max_retries=max_retries,
                                error=str(e),
                            )
                            # Wait for headroom before retrying
                            self._wait_for_fd_headroom()
                            continue
                    raise  # Re-raise if not FD error or retries exhausted

        except Exception as archive_error:
            logger.warning(
                "failed_to_archive_problematic_note",
                note_path=str(file_path),
                archive_error=str(archive_error),
            )

    def _process_deferred_archives(self) -> None:
        """Process all deferred archival requests sequentially.

        Called after parallel scan completes to archive failed notes
        without risking file descriptor exhaustion.
        """
        with self._archival_lock:
            deferred_count = len(self._deferred_archives)
            archives_to_process = self._deferred_archives.copy()
            self._deferred_archives.clear()

        if deferred_count == 0:
            return

        logger.info(
            "processing_deferred_archives",
            count=deferred_count,
            batch_size=self._archiver_batch_size,
        )

        archived_count = 0
        for batch_start in range(0, deferred_count, self._archiver_batch_size):
            # Proactively check for headroom before starting a batch
            self._wait_for_fd_headroom()

            batch = archives_to_process[
                batch_start: batch_start + self._archiver_batch_size
            ]

            for archive_request in batch:
                file_path: Path = archive_request["file_path"]
                note_content = archive_request["note_content"]
                if not file_path.exists() and note_content is None:
                    logger.warning(
                        "skipping_deferred_archive_missing_source",
                        file=str(file_path),
                        relative_path=archive_request["relative_path"],
                    )
                    continue

                self._archive_note_immediate(
                    file_path=file_path,
                    relative_path=archive_request["relative_path"],
                    error=archive_request["error"],
                    processing_stage=archive_request["processing_stage"],
                    note_content=note_content,
                    card_index=archive_request["card_index"],
                    language=archive_request["language"],
                )
                archived_count += 1

                if archived_count % 100 == 0:
                    logger.info(
                        "deferred_archive_progress",
                        processed=archived_count,
                        total=deferred_count,
                    )

        logger.info(
            "deferred_archives_completed",
            archived=archived_count,
        )

    def _wait_for_fd_headroom(self) -> None:
        """Pause archival if the process is too close to the FD limit."""
        has_headroom, snapshot = has_fd_headroom(self._archiver_fd_headroom)
        if has_headroom:
            return

        logger.warning(
            "archiver_fd_headroom_low",
            required_headroom=self._archiver_fd_headroom,
            **snapshot,
        )

        fd_wait_start = time.time()
        fd_wait_max = 30  # seconds - prevent infinite hang

        while True:
            time.sleep(self._archiver_fd_poll_interval)
            has_headroom, snapshot = has_fd_headroom(
                self._archiver_fd_headroom)
            if has_headroom:
                logger.debug(
                    "archiver_fd_headroom_restored",
                    **snapshot,
                )
                break

            # Timeout to prevent infinite hang
            if time.time() - fd_wait_start > fd_wait_max:
                logger.error(
                    "fd_headroom_timeout",
                    waited=fd_wait_max,
                    required_headroom=self._archiver_fd_headroom,
                    **snapshot,
                )
                break  # Continue anyway to avoid process hang
