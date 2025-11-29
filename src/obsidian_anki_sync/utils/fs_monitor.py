"""Lightweight helpers for tracking process file descriptor usage.

The sync pipeline touches thousands of notes per run, so we need visibility
into how many file descriptors (FDs) are currently in use and whether we are
close to the OS-imposed limits.  The helpers in this module avoid hard
dependencies: we try `psutil` first (when installed for observability) and
degrade gracefully to inspecting `/proc/self/fd` (Linux) or `/dev/fd`
(macOS/BSD).  When none of the mechanisms are available we return ``None`` so
callers can decide whether to skip guard-rail logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # pragma: no cover - platform specific
    import resource
except ImportError:  # pragma: no cover - platform specific
    resource = None  # type: ignore

DEFAULT_HEADROOM: Final[int] = 32


def get_open_file_count() -> int | None:
    """Return the current number of open file descriptors for this process."""
    if psutil is not None:
        try:  # pragma: no cover - depends on psutil availability
            return psutil.Process().num_fds()
        except (AttributeError, psutil.Error):
            pass

    for path in (Path("/proc/self/fd"), Path("/dev/fd")):
        count = _count_directory_entries(path)
        if count is not None:
            return count

    return None


def get_fd_limits() -> tuple[int | None, int | None]:
    """Return the (soft, hard) RLIMIT_NOFILE values when available."""
    if resource is None:  # pragma: no cover - unavailable on Windows
        return None, None

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, ValueError, OSError):  # pragma: no cover - unlikely
        return None, None
    return int(soft), int(hard)


def has_fd_headroom(min_headroom: int = DEFAULT_HEADROOM) -> tuple[bool, dict[str, int | None]]:
    """Return whether we have the requested FD headroom."""
    open_count = get_open_file_count()
    soft_limit, hard_limit = get_fd_limits()

    if open_count is None or soft_limit is None:
        # Without reliable data we optimistically assume enough headroom but still
        # report the snapshot so callers can log it for diagnostics.
        snapshot = {
            "open_fd_count": open_count,
            "soft_fd_limit": soft_limit,
            "hard_fd_limit": hard_limit,
        }
        return True, snapshot

    available = soft_limit - open_count
    snapshot = {
        "open_fd_count": open_count,
        "soft_fd_limit": soft_limit,
        "hard_fd_limit": hard_limit,
        "fd_available": available,
    }
    return available >= min_headroom, snapshot


def _count_directory_entries(path: Path) -> int | None:
    """Best-effort helper to count entries in /proc/self/fd or /dev/fd."""
    try:
        if not path.exists() or not path.is_dir():
            return None
    except OSError:
        return None

    try:
        return sum(1 for entry in path.iterdir() if entry.name not in {".", ".."})
    except OSError:
        return None
