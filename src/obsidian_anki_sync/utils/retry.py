"""Retry logic with exponential backoff."""

import time
from functools import wraps
from typing import Callable, TypeVar, Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
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

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=attempt,
                            error=str(e)
                        )
                        raise

                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e)
                    )

                    time.sleep(delay)
                    delay *= backoff_factor

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            return func(*args, **kwargs)

        return wrapper
    return decorator

