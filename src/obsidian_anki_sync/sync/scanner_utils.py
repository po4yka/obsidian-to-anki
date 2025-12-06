"""Utility functions and classes for note scanning."""

import time
from collections.abc import Collection
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from arq.connections import RedisSettings

from obsidian_anki_sync.utils.fs_monitor import has_fd_headroom
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


def describe_redis_settings(settings: RedisSettings | None) -> dict[str, Any]:
    """Return non-sensitive Redis connection attributes for logging."""
    if not settings:
        return {}
    return {
        "host": settings.host,
        "port": settings.port,
        "db": getattr(settings, "database", None),
        "ssl": bool(getattr(settings, "ssl", None)),
    }


class ThreadSafeSlugView(Collection[str]):
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


def calculate_optimal_workers() -> int:
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


def wait_for_fd_headroom(required_headroom: int, poll_interval: float = 0.05) -> None:
    """Pause archival if the process is too close to the FD limit.

    Args:
        required_headroom: Minimum number of file descriptors needed
        poll_interval: Time to wait between checks
    """
    has_headroom, snapshot = has_fd_headroom(required_headroom)
    if has_headroom:
        return

    logger.warning(
        "archiver_fd_headroom_low",
        required_headroom=required_headroom,
        **snapshot,
    )

    fd_wait_start = time.time()
    fd_wait_max = 30  # seconds - prevent infinite hang

    while True:
        time.sleep(poll_interval)
        has_headroom, snapshot = has_fd_headroom(required_headroom)
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
                required_headroom=required_headroom,
                **snapshot,
            )
            break  # Continue anyway to avoid process hang

