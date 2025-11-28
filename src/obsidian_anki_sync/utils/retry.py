"""Retry logic with exponential backoff."""

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
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
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

                    # Log successful retry if this wasn't the first attempt
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

                    return result

                except exceptions as e:
                    attempt_duration = time.time() - attempt_start
                    attempt_durations.append(attempt_duration)
                    last_exception = e

                    if attempt == max_attempts:
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
                        raise

                    # Extract additional context from args if possible
                    context = {}
                    if args:
                        # Try to extract useful context from first few args
                        if hasattr(args[0], "__class__"):
                            context["self_class"] = args[0].__class__.__name__
                        # Check for common parameter names in kwargs
                        for key in ["model", "slug", "path", "file"]:
                            if key in kwargs:
                                context[key] = str(kwargs[key])[:50]

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

                    time.sleep(delay)
                    cumulative_wait_time += delay
                    delay *= backoff_factor

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            return func(*args, **kwargs)

        return wrapper

    return decorator
