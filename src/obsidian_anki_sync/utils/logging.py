"""Logging configuration using loguru for better console and file output."""

import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


def safe_rotation(message, file):
    """Rotation function that rotates at midnight, handles missing files gracefully."""
    try:
        # Check if file exists and has content
        if file is None:
            return False
        # Get current time from the message
        record_time = message.record.get("time")
        if record_time is None:
            return False
        # Rotate at midnight (when hour is 0 and this is the first message of the day)
        if record_time.hour == 0 and record_time.minute == 0:
            # Only rotate if file has content
            try:
                if file.tell() > 0:
                    return True
            except (OSError, ValueError):
                pass
        return False
    except Exception:
        # Fail safe - don't rotate on errors
        return False


def _add_formatted_extra(record: dict) -> bool:
    """Add formatted extra fields to the record for display.

    This filter adds a '_formatted' field to the record's extra dict
    containing a formatted string of all extra fields, prioritizing
    key fields like file, title, note_id.

    Returns:
        True to allow the log record to be processed
    """
    extra = record.get("extra", {})
    if not extra:
        record["extra"]["_formatted"] = ""
        return True

    # Define priority fields that should always be shown first
    priority_fields = ["file", "title", "note_id", "source_path"]
    important_parts = []
    other_parts = []

    for key, value in extra.items():
        # Skip internal fields
        if key in ("name", "_formatted"):
            continue

        if key in priority_fields:
            if value:  # Only show non-empty values
                important_parts.append(f"{key}={value}")
        else:
            if value is not None and value != "":
                other_parts.append(f"{key}={value}")

    # Combine priority fields first, then others
    all_parts = important_parts + other_parts

    if all_parts:
        record["extra"]["_formatted"] = " | " + " ".join(all_parts)
    else:
        record["extra"]["_formatted"] = ""

    return True


@dataclass(slots=True)
class HighVolumeEventPolicy:
    """
    Rate-limiting policy for high-frequency log events.

    Attributes:
        max_occurrences: Maximum number of events allowed within the window.
        window_seconds: Sliding window size in seconds for counting events.
    """

    max_occurrences: int
    window_seconds: float


class ConsoleNoiseFilter:
    """
    Filter that reduces console noise while preserving critical diagnostics.

    This filter wraps a base filter (for formatting extras), enforces module-level
    minimum log levels, and rate-limits specific high-volume events within a
    sliding time window.
    """

    def __init__(
        self,
        base_filter: Callable[[dict], bool],
        level_overrides: Mapping[str, str] | None = None,
        high_volume_policies: Mapping[str, HighVolumeEventPolicy] | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize the console noise filter.

        Args:
            base_filter: Callable used to enrich/validate records before filtering.
            level_overrides: Mapping of module prefixes to minimum log levels.
            high_volume_policies: Mapping of event names to rate-limit policies.
            time_func: Optional time provider for testing (defaults to time.monotonic).
        """
        self.base_filter = base_filter
        self.level_overrides = dict(level_overrides or {})
        self.high_volume_policies = dict(high_volume_policies or {})
        self._resolved_level_overrides = {
            prefix: logger.level(level_name).no
            for prefix, level_name in self.level_overrides.items()
            if level_name
        }
        self._event_windows: dict[str, deque[float]] = {
            event: deque() for event in self.high_volume_policies
        }
        self._lock = threading.Lock()
        self._time_func = time_func or time.monotonic

    def __call__(self, record: dict) -> bool:
        """Apply formatting, level overrides, and rate limits to a log record."""
        if not self.base_filter(record):
            return False

        level_obj = record.get("level")
        if hasattr(level_obj, "no"):
            level_no = getattr(level_obj, "no")
        else:
            try:
                level_no = logger.level(str(level_obj)).no
            except (TypeError, ValueError):
                level_no = logger.level("INFO").no

        module_name = record.get("name", "") or ""
        for prefix, min_level in self._resolved_level_overrides.items():
            if module_name.startswith(prefix) and level_no < min_level:
                return False

        message = record.get("message")
        policy = (
            self.high_volume_policies.get(message) if isinstance(message, str) else None
        )
        if policy:
            now = self._time_func()
            with self._lock:
                window = self._event_windows.setdefault(message, deque())
                while window and now - window[0] > policy.window_seconds:
                    window.popleft()
                if len(window) >= policy.max_occurrences:
                    return False
                window.append(now)

        return True


DEFAULT_CONSOLE_LEVEL_OVERRIDES: dict[str, str] = {
    # Provider factory emits high-frequency INFO logs; raise bar to WARNING.
    "obsidian_anki_sync.providers": "WARNING",
}

DEFAULT_HIGH_VOLUME_EVENTS: dict[str, HighVolumeEventPolicy] = {
    # Limit repetitive provider lifecycle logs that flood the console.
    "creating_provider_from_config": HighVolumeEventPolicy(2, 60.0),
    "creating_provider": HighVolumeEventPolicy(3, 60.0),
    "provider_created_successfully": HighVolumeEventPolicy(3, 60.0),
    # Directory discovery can log hundreds of entries; allow a handful per burst.
    "discover_notes_in_dir": HighVolumeEventPolicy(5, 10.0),
}


def configure_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    log_file: Path | None = None,
    very_verbose: bool = False,
    project_log_dir: Path | None = None,
    error_log_retention_days: int = 90,
    enable_console_noise_filter: bool = True,
) -> None:
    """Configure loguru logging with dual output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_file: Specific log file path (overrides log_dir)
        very_verbose: If True, log full LLM requests/responses
        project_log_dir: Directory for project-level logs (default: ./logs)
        error_log_retention_days: Days to retain error logs (default: 90)
        enable_console_noise_filter: Toggle console-side noise suppression
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    level = log_level.upper()

    if enable_console_noise_filter:
        console_filter: Callable[[dict], bool] = ConsoleNoiseFilter(
            base_filter=_add_formatted_extra,
            level_overrides=DEFAULT_CONSOLE_LEVEL_OVERRIDES,
            high_volume_policies=DEFAULT_HIGH_VOLUME_EVENTS,
        )
    else:
        console_filter = _add_formatted_extra

    # Add console handler - concise format with structured fields
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level><blue>{extra[_formatted]}</blue>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        filter=console_filter,
    )

    # Add file handler - detailed format with rotation (vault-level or custom)
    if log_file:
        # Use specific log file
        log_path = log_file
        # Ensure parent directory exists
        log_file.parent.mkdir(exist_ok=True, parents=True)
    else:
        # Use log directory
        if log_dir is None:
            log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_path = log_dir / "obsidian-anki-sync_{time:YYYY-MM-DD}.log"

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}{extra[_formatted]}",
        level="DEBUG",  # File gets all logs
        # Rotate at midnight only for auto-named files
        rotation=safe_rotation if not log_file else None,
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        backtrace=True,  # Include traceback
        diagnose=True,  # Include variable values in tracebacks
        enqueue=True,  # Thread-safe
        filter=_add_formatted_extra,
    )

    # Add project-level log file handler (in project root)
    if project_log_dir is None:
        project_log_dir = Path("./logs")
    project_log_dir.mkdir(exist_ok=True, parents=True)
    project_log_path = project_log_dir / "obsidian-anki-sync_{time:YYYY-MM-DD}.log"

    logger.add(
        project_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}{extra[_formatted]}",
        level="DEBUG",  # Project logs get all levels
        rotation=safe_rotation,
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        filter=_add_formatted_extra,
    )

    # Add error-specific log file handler (ERROR and above only)
    error_log_path = project_log_dir / "errors_{time:YYYY-MM-DD}.log"

    def error_filter(record: dict) -> bool:
        """Filter to only include ERROR and CRITICAL level logs."""
        return record["level"].no >= logger.level("ERROR").no

    logger.add(
        error_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}{extra[_formatted]}\n{exception}",
        level="ERROR",  # Only ERROR and above
        rotation=safe_rotation,
        retention=f"{error_log_retention_days} days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        filter=lambda record: error_filter(record) and _add_formatted_extra(record),
    )

    if very_verbose:
        # Add a separate handler for very verbose LLM logging
        verbose_log_path = (
            log_path.parent / f"{log_path.stem}_verbose{log_path.suffix}"
            if log_file
            else log_dir / "obsidian-anki-sync_verbose_{time:YYYY-MM-DD}.log"
        )
        logger.add(
            verbose_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="DEBUG",
            filter=lambda record: "llm" in record["name"].lower()
            or "prompt" in record["message"].lower()
            or "response" in record["message"].lower(),
            rotation="00:00" if not log_file else None,
            retention="7 days",  # Keep verbose logs for shorter time
            compression="zip",
        )

    logger.info(
        "logging_configured",
        console_level=level,
        file_level="DEBUG",
        log_dir=str(log_dir) if not log_file else str(log_file.parent),
        log_file=str(log_file) if log_file else None,
        project_log_dir=str(project_log_dir),
        error_log_dir=str(project_log_dir),
        very_verbose=very_verbose,
        console_noise_filter=enable_console_noise_filter,
    )


def get_logger(name: str) -> Any:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Loguru logger bound to the given name
    """
    # Loguru uses a single global logger, but we can bind context
    return logger.bind(name=name)
