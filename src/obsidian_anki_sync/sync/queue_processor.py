"""Queue-based processing service for distributed note scanning."""

import asyncio
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arq import create_pool
from arq.connections import RedisSettings

from obsidian_anki_sync.domain.interfaces.note_scanner import IQueueProcessor
from obsidian_anki_sync.exceptions import CircuitBreakerOpenError, ConfigurationError
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.sync.scanner_utils import describe_redis_settings
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class QueueNoteProcessor(IQueueProcessor):
    """Service for distributed note processing using Redis queues."""

    def __init__(
        self,
        config: Any,
        progress_tracker: "ProgressTracker | None" = None,
        stats: dict[str, Any] | None = None,
    ):
        """Initialize queue processor.

        Args:
            config: Service configuration
            progress_tracker: Optional progress tracker
            stats: Statistics dictionary to update
        """
        self.config = config
        self.progress = progress_tracker
        self.stats = stats or {}

    def scan_notes_with_queue(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: defaultdict[str, int],
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan notes using Redis queue for distribution.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """
        return asyncio.run(
            self._scan_notes_with_queue_async(
                note_files,
                obsidian_cards,
                existing_slugs,
                error_by_type,
                error_samples,
                qa_extractor,
                on_batch_complete,
            )
        )

    async def _validate_redis_connection(
        self, pool, redis_settings: RedisSettings
    ) -> bool:
        """Validate Redis connection is healthy before submitting jobs.

        Args:
            pool: ArqRedis pool instance

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            await pool.ping()
            logger.info(
                "redis_connection_healthy",
                **describe_redis_settings(redis_settings),
            )
            return True
        except Exception as e:
            logger.error(
                "redis_connection_failed",
                error=str(e),
                **describe_redis_settings(redis_settings),
            )
            return False

    async def _enqueue_with_retry(
        self,
        pool,
        func_name: str,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        redis_settings: RedisSettings | None = None,
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
                        **describe_redis_settings(redis_settings),
                    )
                    await asyncio.sleep(wait_time)
                    # Exponential backoff, max 30s
                    delay = min(delay * 2, 30.0)
                else:
                    logger.error(
                        "enqueue_retry_exhausted",
                        attempts=max_retries + 1,
                        error=str(e),
                        **describe_redis_settings(redis_settings),
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
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Async implementation of queue scanning with stability improvements.

        Features:
        - Redis health check before job submission
        - Retry logic with exponential backoff for job submission
        - Circuit breaker pattern for Redis failures
        - Event-driven result collection using BLPOP (no polling)
        - Overall and per-job timeouts
        """
        logger.info(
            "queue_scan_started",
            total_notes=len(note_files),
            redis_url=self.config.redis_url,
        )

        # Use hardened Redis settings
        redis_settings = RedisSettings.from_dsn(self.config.redis_url)
        redis_settings.conn_timeout = getattr(
            self.config, "redis_socket_connect_timeout", 5.0
        )
        redis_context = describe_redis_settings(redis_settings)

        try:
            pool = await create_pool(redis_settings)
            logger.info("redis_pool_created", **redis_context)
        except Exception as pool_error:
            logger.error(
                "redis_pool_creation_failed",
                error=str(pool_error),
                **redis_context,
            )
            raise

        # Validate Redis connection before proceeding
        if not await self._validate_redis_connection(pool, redis_settings):
            await pool.close()
            msg = (
                f"Redis unavailable at {self.config.redis_url}. "
                "Check that Redis is running and accessible."
            )
            raise ConfigurationError(msg)

        # Generate unique session ID for this batch
        session_id = str(uuid.uuid4())
        result_queue_name = f"obsidian_anki_sync:results:{session_id}"

        # job_id -> (file_path, relative_path)
        job_map: dict[str, tuple[Any, str]] = {}
        # job_id -> submission timestamp
        job_submit_times: dict[str, float] = {}

        # Circuit breaker state for Redis operations
        consecutive_failures = 0
        redis_error_streak = 0
        circuit_breaker_threshold = getattr(
            self.config, "queue_circuit_breaker_threshold", 3
        )

        logger.info(
            "submitting_jobs",
            total_files=len(note_files),
            session_id=session_id,
            result_queue=result_queue_name,
        )
        logger.info(
            "result_queue_initialized",
            queue=result_queue_name,
            session_id=session_id,
            pending=len(note_files),
        )

        # Pre-set TTL on the result queue to avoid orphaned keys if workers never push
        result_ttl = int(getattr(self.config, "result_queue_ttl_seconds", 3600) or 3600)
        try:
            await pool.expire(result_queue_name, result_ttl)
        except Exception as e:
            logger.warning(
                "result_queue_ttl_set_failed",
                queue=result_queue_name,
                error=str(e),
                ttl_seconds=result_ttl,
                **redis_context,
            )

        # Submit all jobs with retry logic
        for file_path, relative_path in note_files:
            # Check circuit breaker - fail fast if too many consecutive failures
            if consecutive_failures >= circuit_breaker_threshold:
                logger.error(
                    "circuit_breaker_open",
                    consecutive_failures=consecutive_failures,
                    threshold=circuit_breaker_threshold,
                )
                await pool.close()
                msg = (
                    f"Circuit breaker opened after {consecutive_failures} "
                    f"consecutive failures (threshold: {circuit_breaker_threshold})"
                )
                raise CircuitBreakerOpenError(
                    msg,
                    consecutive_failures=consecutive_failures,
                    threshold=circuit_breaker_threshold,
                    suggestion=(
                        "Check Redis connectivity and worker health. "
                        "Reduce batch size or increase circuit breaker threshold."
                    ),
                )

            try:
                # Sanitize job ID to avoid issues with slashes and scope by session
                sanitized_path = relative_path.replace("/", "_").replace("\\", "_")
                job_id = f"{session_id}:note-{sanitized_path}:{uuid.uuid4().hex}"

                # Enqueue with retry logic
                job = await self._enqueue_with_retry(
                    pool,
                    "process_note_job",
                    str(file_path),
                    relative_path,
                    job_id=job_id,
                    max_retries=getattr(self.config, "queue_max_retries", 3),
                    redis_settings=redis_settings,
                    _job_id=job_id,
                    result_queue_name=result_queue_name,
                )

                submit_time = time.time()
                if job:
                    job_map[job.job_id] = (file_path, relative_path)
                    job_submit_times[job.job_id] = submit_time
                    logger.debug("job_enqueued", job_id=job.job_id, file=relative_path)

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

        # Wait for results using BLPOP
        pending_jobs = set(job_map.keys())
        completed_jobs = 0

        # Timeout configuration
        max_wait_time = getattr(self.config, "queue_max_wait_time_seconds", 18000)
        start_time = time.time()

        # We use a shorter timeout for BLPOP to allow checking overall timeout
        blpop_timeout = 1.0

        wait_log_interval = 30.0
        last_wait_log = 0.0

        try:
            while pending_jobs:
                # Check overall timeout
                elapsed_total = time.time() - start_time
                if elapsed_total > max_wait_time:
                    logger.error(
                        "queue_polling_timeout",
                        pending_count=len(pending_jobs),
                        elapsed=elapsed_total,
                        max_wait=max_wait_time,
                    )
                    for job_id in pending_jobs:
                        file_path, relative_path = job_map[job_id]
                        logger.error(
                            "job_stuck_at_timeout", job_id=job_id, file=relative_path
                        )
                        self.stats["errors"] = self.stats.get("errors", 0) + 1
                    break

                try:
                    # BLPOP returns (key, value) tuple or None on timeout
                    # We use a small timeout to allow the loop to check overall constraints
                    result_data = await pool.blpop(
                        result_queue_name, timeout=blpop_timeout
                    )
                    redis_error_streak = 0

                    if result_data:
                        _, payload = result_data
                        import json

                        result = json.loads(payload)

                        job_id = result.get("job_id")
                        if not job_id:
                            logger.warning(
                                "result_missing_job_id",
                                queue=result_queue_name,
                                pending=len(pending_jobs),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            error_by_type["queue_error"] += 1
                            if len(error_samples["queue_error"]) < 3:
                                error_samples["queue_error"].append(
                                    "Queue Error: missing job_id in result"
                                )
                            continue

                        if job_id not in job_map:
                            logger.warning(
                                "unexpected_job_id_result",
                                job_id=job_id,
                                queue=result_queue_name,
                                pending=len(pending_jobs),
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            error_by_type["queue_error"] += 1
                            if len(error_samples["queue_error"]) < 3:
                                error_samples["queue_error"].append(
                                    f"Queue Error: unexpected job_id {job_id}"
                                )
                            continue

                        file_path, relative_path = job_map[job_id]

                        if job_id not in pending_jobs:
                            logger.warning(
                                "duplicate_job_result",
                                job_id=job_id,
                                file=relative_path,
                                queue=result_queue_name,
                            )
                            continue

                        pending_jobs.remove(job_id)
                        completed_jobs = len(job_map) - len(pending_jobs)

                        job_submit_time = job_submit_times.get(job_id)
                        job_duration = (
                            round(time.time() - job_submit_time, 2)
                            if job_submit_time
                            else None
                        )

                        logger.debug(
                            "queue_result_received",
                            queue=result_queue_name,
                            job_id=job_id,
                            file=relative_path,
                            success=result.get("success"),
                            cards=len(result.get("cards", [])),
                            completed=completed_jobs,
                            total=len(job_map),
                            duration=job_duration,
                        )

                        if result.get("success"):
                            for card_dict in result.get("cards", []):
                                try:
                                    card = Card(**card_dict)
                                    obsidian_cards[card.slug] = card
                                    existing_slugs.add(card.slug)

                                    # Atomic processing callback
                                    if on_batch_complete:
                                        on_batch_complete([card])
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
                                "job_failed_result",
                                error=error_msg,
                                job_id=job_id,
                                file=relative_path,
                            )
                            self.stats["errors"] = self.stats.get("errors", 0) + 1
                            error_by_type["queue_error"] += 1
                            if len(error_samples["queue_error"]) < 3:
                                error_samples["queue_error"].append(
                                    f"Queue Error: {relative_path}: {error_msg}"
                                )
                            # Fail-fast: abort in strict mode
                            if getattr(self.config, "strict_mode", True):
                                msg = f"Queue job failed: {error_msg}"
                                raise RuntimeError(msg)

                        logger.info(
                            "queue_progress",
                            completed=completed_jobs,
                            total=len(job_map),
                        )

                    elif elapsed_total - last_wait_log >= wait_log_interval:
                        logger.info(
                            "queue_waiting_for_results",
                            queue=result_queue_name,
                            pending=len(pending_jobs),
                            elapsed=int(elapsed_total),
                        )
                        last_wait_log = elapsed_total

                except Exception as e:
                    redis_error_streak += 1
                    logger.error(
                        "error_processing_result_queue",
                        error=str(e),
                        queue=result_queue_name,
                        streak=redis_error_streak,
                        **redis_context,
                    )
                    if redis_error_streak >= circuit_breaker_threshold:
                        message = (
                            "Redis errors exceeded circuit breaker threshold "
                            f"({redis_error_streak}/{circuit_breaker_threshold})."
                        )
                        raise CircuitBreakerOpenError(
                            message,
                            consecutive_failures=redis_error_streak,
                            threshold=circuit_breaker_threshold,
                            suggestion="Inspect Redis availability or network; restart workers.",
                        )
                    # Don't crash the loop, just retry
                    await asyncio.sleep(1.0)

        finally:
            # Cleanup: expire the result queue immediately
            try:
                await pool.delete(result_queue_name)
                logger.info("result_queue_deleted", queue=result_queue_name)
            except Exception as e:
                logger.warning(
                    "failed_to_cleanup_result_queue",
                    queue=result_queue_name,
                    error=str(e),
                )

            await pool.close()
            logger.info("redis_pool_closed", **redis_context)

        return obsidian_cards
