"""Performance timing utilities for logging."""

import time
from contextlib import contextmanager
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


from collections.abc import Generator

@contextmanager
def log_duration(operation: str, **context: Any) -> Generator[None, None, None]:
    """Context manager to log operation duration.

    Args:
        operation: Operation name (e.g., "note_scanning", "card_generation")
        **context: Additional context to include in log entries

    Yields:
        None
    """
    start_time = time.time()
    logger.debug(f"{operation}_start", **context)

    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(
            f"{operation}_completed",
            operation=operation,
            duration=round(duration, 3),
            **context,
        )


def log_timing(operation: str, duration: float, **context: Any) -> None:
    """Log operation timing information.

    Args:
        operation: Operation name
        duration: Duration in seconds
        **context: Additional context to include
    """
    logger.info(
        f"{operation}_timing",
        operation=operation,
        duration=round(duration, 3),
        **context,
    )
