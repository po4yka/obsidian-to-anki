"""Retry logic with exponential backoff."""

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _extract_context(args: tuple, kwargs: dict) -> dict:
            """Extract additional context from args for logging."""
            context = {}
            if args:
                if hasattr(args[0], "__class__"):
                    context["self_class"] = args[0].__class__.__name__
                for key in ["model", "slug", "path", "file"]:
                    if key in kwargs:
                        context[key] = str(kwargs[key])[:50]
            return context

        def _log_success(
            attempt: int,
            retry_start_time: float,
            cumulative_wait_time: float,
            attempt_duration: float,
        ) -> None:
            """Log successful retry if this wasn't the first attempt."""
            if attempt > 1:
                total_retry_time = time.time() - retry_start_time
                logger.info(
                    "retry_succeeded",
                    func=func.__name__,
                    attempt=attempt,
                    total_attempts=attempt,
                    total_retry_time=round(total_retry_time, 2),
                    cumulative_wait_time=round(cumulative_wait_time, 2),
                    attempt_duration=round(attempt_duration, 2),
                )

        def _log_exhausted(
            attempt: int,
            e: Exception,
            retry_start_time: float,
            cumulative_wait_time: float,
            attempt_durations: list[float],
        ) -> None:
            """Log when all retry attempts are exhausted."""
            total_retry_time = time.time() - retry_start_time
            logger.error(
                "retry_exhausted",
                func=func.__name__,
                attempts=attempt,
                error=str(e),
                error_type=type(e).__name__,
                total_retry_time=round(total_retry_time, 2),
                cumulative_wait_time=round(cumulative_wait_time, 2),
                attempt_durations=[round(d, 2) for d in attempt_durations],
            )

        def _log_attempt(
            attempt: int,
            delay: float,
            e: Exception,
            attempt_duration: float,
            cumulative_wait_time: float,
            context: dict,
        ) -> None:
            """Log a retry attempt."""
            logger.warning(
                "retry_attempt",
                func=func.__name__,
                attempt=attempt,
                max_attempts=max_attempts,
                delay=delay,
                error=str(e),
                error_type=type(e).__name__,
                attempt_duration=round(attempt_duration, 2),
                cumulative_wait_time=round(cumulative_wait_time, 2),
                **context,
            )

        # Check if function is async
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                delay = initial_delay
                last_exception = None
                retry_start_time = time.time()
                cumulative_wait_time = 0.0
                attempt_durations: list[float] = []

                for attempt in range(1, max_attempts + 1):
                    attempt_start = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        attempt_duration = time.time() - attempt_start
                        _log_success(
                            attempt,
                            retry_start_time,
                            cumulative_wait_time,
                            attempt_duration,
                        )
                        return result

                    except exceptions as e:
                        attempt_duration = time.time() - attempt_start
                        attempt_durations.append(attempt_duration)
                        last_exception = e

                        if attempt == max_attempts:
                            _log_exhausted(
                                attempt,
                                e,
                                retry_start_time,
                                cumulative_wait_time,
                                attempt_durations,
                            )
                            raise

                        context = _extract_context(args, kwargs)
                        _log_attempt(
                            attempt,
                            delay,
                            e,
                            attempt_duration,
                            cumulative_wait_time,
                            context,
                        )

                        await asyncio.sleep(delay)
                        cumulative_wait_time += delay
                        delay *= backoff_factor

                if last_exception:
                    raise last_exception
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                delay = initial_delay
                last_exception = None
                retry_start_time = time.time()
                cumulative_wait_time = 0.0
                attempt_durations: list[float] = []

                for attempt in range(1, max_attempts + 1):
                    attempt_start = time.time()
                    try:
                        result = func(*args, **kwargs)
                        attempt_duration = time.time() - attempt_start
                        _log_success(
                            attempt,
                            retry_start_time,
                            cumulative_wait_time,
                            attempt_duration,
                        )
                        return result

                    except exceptions as e:
                        attempt_duration = time.time() - attempt_start
                        attempt_durations.append(attempt_duration)
                        last_exception = e

                        if attempt == max_attempts:
                            _log_exhausted(
                                attempt,
                                e,
                                retry_start_time,
                                cumulative_wait_time,
                                attempt_durations,
                            )
                            raise

                        context = _extract_context(args, kwargs)
                        _log_attempt(
                            attempt,
                            delay,
                            e,
                            attempt_duration,
                            cumulative_wait_time,
                            context,
                        )

                        time.sleep(delay)
                        cumulative_wait_time += delay
                        delay *= backoff_factor

                if last_exception:
                    raise last_exception
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
