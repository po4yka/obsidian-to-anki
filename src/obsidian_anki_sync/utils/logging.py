"""Logging configuration using structlog for structured JSON logging."""

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer
from structlog.stdlib import LoggerFactory, add_log_level, add_logger_name

# Standard library logging levels mapping
_LOG_LEVELS = {
    "TRACE": logging.DEBUG - 5,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "SUCCESS": logging.INFO + 5,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# User-facing event names that should appear on terminal (without --verbose)
# These are the only events shown to users by default, plus all ERROR/CRITICAL
USER_FACING_EVENTS: set[str] = {
    # Sync lifecycle
    "sync_started",
    "sync_completed",
    "sync_interrupted",
    "sync_failed",
    # Summary statistics
    "sync_summary",
    "cards_created_summary",
    "cards_updated_summary",
    "cards_deleted_summary",
    # User warnings
    "missing_files_warning",
    "config_warning",
    "anki_connection_warning",
    "validation_warning",
    "resume_validation_warning",
    # Pre-flight checks
    "preflight_check_failed",
    "preflight_check_passed",
}


def _get_level_no(level_name: str) -> int:
    """Get numeric log level from name."""
    level_name = level_name.upper()
    return _LOG_LEVELS.get(level_name, logging.INFO)


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


class ConsoleNoiseFilterProcessor:
    """
    Structlog processor that reduces console noise while preserving critical diagnostics.

    This processor enforces module-level minimum log levels and rate-limits
    specific high-volume events within a sliding time window.
    """

    def __init__(
        self,
        level_overrides: Mapping[str, str] | None = None,
        high_volume_policies: Mapping[str, HighVolumeEventPolicy] | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize the console noise filter processor.

        Args:
            level_overrides: Mapping of module prefixes to minimum log levels.
            high_volume_policies: Mapping of event names to rate-limit policies.
            time_func: Optional time provider for testing (defaults to time.monotonic).
        """
        self.level_overrides = dict(level_overrides or {})
        self.high_volume_policies = dict(high_volume_policies or {})
        self._resolved_level_overrides = {
            prefix: _get_level_no(level_name)
            for prefix, level_name in self.level_overrides.items()
            if level_name
        }
        self._event_windows: dict[str, deque[float]] = {
            event: deque() for event in self.high_volume_policies
        }
        self._lock = threading.Lock()
        self._time_func = time_func or time.monotonic

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Apply level overrides and rate limits to a log event."""
        # Check module-level overrides
        logger_name = event_dict.get("logger", "") or ""
        level_no = event_dict.get("level", logging.INFO)

        if isinstance(level_no, str):
            level_no = _get_level_no(level_no)
        elif hasattr(level_no, "no"):
            level_no = getattr(level_no, "no")
        elif not isinstance(level_no, int):
            level_no = logging.INFO

        for prefix, min_level in self._resolved_level_overrides.items():
            if logger_name.startswith(prefix) and level_no < min_level:
                # Drop this event
                raise structlog.DropEvent

        # Check rate limiting
        message = event_dict.get("event", "")
        policy = (
            self.high_volume_policies.get(message) if isinstance(message, str) else None
        )
        if policy:
            now = self._time_func()
            with self._lock:
                window = self._event_windows.setdefault(
                    str(message) if message else "", deque()
                )
                while window and now - window[0] > policy.window_seconds:
                    window.popleft()
                if len(window) >= policy.max_occurrences:
                    # Drop this event
                    raise structlog.DropEvent
                window.append(now)

        return event_dict


class UserFacingConsoleFilter(logging.Filter):
    """Logging filter that only passes user-facing events to console.

    This filter ensures that the terminal only shows important progress messages
    while detailed debug information goes to log files.

    Allows:
    - Events in USER_FACING_EVENTS set
    - All ERROR and CRITICAL level messages
    - All messages when verbose mode is enabled
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the user-facing console filter.

        Args:
            verbose: If True, pass all events through (disable filtering)
        """
        super().__init__()
        self.verbose = verbose

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records for console output.

        Args:
            record: The log record to filter

        Returns:
            True if the record should be logged, False otherwise
        """
        # In verbose mode, pass everything through
        if self.verbose:
            return True

        # Always show ERROR and CRITICAL messages
        if record.levelno >= logging.ERROR:
            return True

        # Try to extract event name from the message
        # For structlog logs, the event name is typically the message
        event = record.getMessage()

        # For structured logs, the event might be in a dict that was formatted
        # Check if any user-facing event name appears in the message
        if isinstance(event, str):
            # Check exact match first (common for structlog events)
            if event in USER_FACING_EVENTS:
                return True

            # Check if any user-facing event is in the message (for structured logs)
            for user_event in USER_FACING_EVENTS:
                if user_event in event:
                    return True

        # Filter out everything else from console
        return False


class UserFriendlyConsoleRenderer:
    """Renders user-facing logs in a clean, readable format for terminal output.

    This renderer provides human-friendly messages for user-facing events
    while falling back to the standard console renderer for other messages.
    """

    def __init__(self) -> None:
        """Initialize with a fallback console renderer."""
        self._fallback = ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> str:
        """Render log event as user-friendly string."""
        event = event_dict.get("event", "")
        level = str(event_dict.get("level", "info")).upper()

        # Format based on event type for cleaner user output
        if event == "sync_started":
            vault = event_dict.get("vault", "")
            dry_run = event_dict.get("dry_run", False)
            incremental = event_dict.get("incremental", False)
            sample_size = event_dict.get("sample_size")

            # Build mode description
            mode_parts = []
            if dry_run:
                mode_parts.append("dry-run")
            if incremental:
                mode_parts.append("incremental")
            if sample_size:
                mode_parts.append(f"sample={sample_size}")

            mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
            vault_str = f" for vault: {vault}" if vault else ""
            return f"Starting sync{vault_str}{mode_str}"

        elif event == "sync_completed":
            # Handle both direct fields and stats dict format
            stats = event_dict.get("stats", {})
            created = event_dict.get("created", stats.get("created", 0))
            updated = event_dict.get("updated", stats.get("updated", 0))
            deleted = event_dict.get("deleted", stats.get("deleted", 0))
            # Handle both duration_seconds and total_duration field names
            duration = event_dict.get(
                "duration_seconds", event_dict.get("total_duration", 0)
            )
            return (
                f"Sync completed in {duration:.1f}s: "
                f"{created} created, {updated} updated, {deleted} deleted"
            )

        elif event == "sync_summary":
            total = event_dict.get("total_notes", 0)
            created = event_dict.get("cards_created", 0)
            updated = event_dict.get("cards_updated", 0)
            deleted = event_dict.get("cards_deleted", 0)
            errors = event_dict.get("errors", 0)
            summary = f"Summary: {total} notes processed"
            if created or updated or deleted:
                summary += f" | {created} created, {updated} updated, {deleted} deleted"
            if errors:
                summary += f" | {errors} errors"
            return summary

        elif event == "sync_interrupted":
            return "Sync interrupted by user"

        elif event == "sync_failed":
            error = event_dict.get("error", "Unknown error")
            return f"Sync failed: {error}"

        elif level == "ERROR":
            error = event_dict.get("error", event)
            return f"ERROR: {error}"

        elif level == "WARNING" and event in USER_FACING_EVENTS:
            return f"WARNING: {event}"

        # For other events or verbose mode, use fallback renderer
        return str(self._fallback(logger, method_name, event_dict))


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

# Global state for handlers
_configured = False
_handlers: list[logging.Handler] = []


def _add_formatted_extra_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Add formatted extra fields to the event dict for display.

    This processor adds a '_formatted' field containing a formatted string
    of all extra fields, prioritizing key fields like file, title, note_id.
    """
    # Extract priority fields
    priority_fields = ["file", "title", "note_id", "source_path"]
    important_parts = []
    other_parts = []

    for key, value in event_dict.items():
        # Skip internal fields and standard structlog fields
        if key in ("logger", "level", "event", "timestamp", "exception", "_formatted"):
            continue

        if key in priority_fields:
            if value:  # Only show non-empty values
                important_parts.append(f"{key}={value}")
        elif value is not None and value != "":
            other_parts.append(f"{key}={value}")

    # Combine priority fields first, then others
    all_parts = important_parts + other_parts

    if all_parts:
        event_dict["_formatted"] = " | " + " ".join(all_parts)
    else:
        event_dict["_formatted"] = ""

    return event_dict


def _create_console_renderer() -> ConsoleRenderer:
    """Create console renderer with custom formatting."""
    return ConsoleRenderer(
        colors=True,
        exception_formatter=structlog.dev.plain_traceback,
    )


def _create_json_renderer() -> JSONRenderer:
    """Create JSON renderer for file logs."""
    return JSONRenderer()


def _setup_stdlib_logging(
    log_level: str,
    enable_console_noise_filter: bool,
) -> None:
    """Configure standard library logging to work with structlog."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Base processors for all logs (noise filter is added per-handler, not globally)
    base_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_formatted_extra_processor,
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *base_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    log_file: Path | None = None,
    very_verbose: bool = False,
    verbose: bool = False,
    project_log_dir: Path | None = None,
    error_log_retention_days: int = 90,
    enable_console_noise_filter: bool = True,
) -> None:
    """Configure structlog logging with dual output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_file: Specific log file path (overrides log_dir)
        very_verbose: If True, log full LLM requests/responses to separate file
        verbose: If True, show all log messages on terminal (for debugging)
        project_log_dir: Directory for project-level logs (default: ./logs)
        error_log_retention_days: Days to retain error logs (default: 90)
        enable_console_noise_filter: Toggle console-side noise suppression
    """
    global _configured, _handlers  # noqa: PLW0602

    # Setup standard library logging
    _setup_stdlib_logging(log_level, enable_console_noise_filter)

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in _handlers:
        root_logger.removeHandler(handler)
        handler.close()
    _handlers.clear()

    # Determine log level
    level = _get_level_no(log_level.upper())

    # Determine log paths
    if log_file:
        log_path = log_file
        log_file.parent.mkdir(exist_ok=True, parents=True)
    else:
        if log_dir is None:
            log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        log_path = log_dir / "obsidian-anki-sync.log"

    if project_log_dir is None:
        project_log_dir = Path("./logs")
    project_log_dir.mkdir(exist_ok=True, parents=True)

    # Console handler - human-readable with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    # Console processors (without noise filter if disabled, with it if enabled)
    console_pre_chain: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if enable_console_noise_filter:
        console_pre_chain.append(
            ConsoleNoiseFilterProcessor(
                level_overrides=DEFAULT_CONSOLE_LEVEL_OVERRIDES,
                high_volume_policies=DEFAULT_HIGH_VOLUME_EVENTS,
            )
        )

    console_pre_chain.append(_add_formatted_extra_processor)

    # Add user-facing filter at handler level (filters to user-facing events unless verbose)
    console_handler.addFilter(UserFacingConsoleFilter(verbose=verbose))

    # Use user-friendly renderer when not in verbose mode for cleaner output
    if verbose:
        renderer: Any = _create_console_renderer()
    else:
        renderer = UserFriendlyConsoleRenderer()

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=console_pre_chain,
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    _handlers.append(console_handler)

    # File handler - JSON format with size-based rotation
    from logging.handlers import RotatingFileHandler

    # Use size-based rotation: 50MB max per file, keep 5 backups (250MB total max)
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # File processors (no console noise filter, but include formatted extra)
    file_pre_chain: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_formatted_extra_processor,
    ]

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=_create_json_renderer(),
        foreign_pre_chain=file_pre_chain,
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    _handlers.append(file_handler)

    # Project-level log file handler (skip if same as log_path to avoid duplicate logging)
    project_log_path = project_log_dir / "obsidian-anki-sync.log"
    if project_log_path != log_path:
        project_handler = RotatingFileHandler(
            filename=str(project_log_path),
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding="utf-8",
        )
        project_handler.setLevel(logging.DEBUG)
        project_handler.setFormatter(file_formatter)
        root_logger.addHandler(project_handler)
        _handlers.append(project_handler)

    # Error-specific log file handler (ERROR and above only)
    error_log_path = project_log_dir / "errors.log"

    class ErrorFilter(logging.Filter):
        """Filter to only include ERROR and CRITICAL level logs."""

        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno >= logging.ERROR

    error_handler = RotatingFileHandler(
        filename=str(error_log_path),
        maxBytes=10 * 1024 * 1024,  # 10MB for errors
        backupCount=10,  # Keep more error history
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(ErrorFilter())
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    _handlers.append(error_handler)

    # Verbose LLM logging handler (if enabled)
    if very_verbose:
        if log_file:
            verbose_log_path = (
                log_path.parent / f"{log_path.stem}_verbose{log_path.suffix}"
            )
        else:
            # log_dir is guaranteed to be set if log_file is not set
            assert log_dir is not None
            verbose_log_path = log_dir / "obsidian-anki-sync_verbose.log"

        class VerboseFilter(logging.Filter):
            """Filter for LLM-related verbose logs."""

            def filter(self, record: logging.LogRecord) -> bool:
                logger_name = record.name.lower()
                message = record.getMessage().lower()
                return (
                    "llm" in logger_name or "prompt" in message or "response" in message
                )

        verbose_handler = RotatingFileHandler(
            filename=str(verbose_log_path),
            maxBytes=100 * 1024 * 1024,  # 100MB for verbose LLM logs
            backupCount=3,
            encoding="utf-8",
        )
        verbose_handler.setLevel(logging.DEBUG)
        verbose_handler.addFilter(VerboseFilter())
        verbose_handler.setFormatter(file_formatter)
        root_logger.addHandler(verbose_handler)
        _handlers.append(verbose_handler)

    _configured = True

    # Log configuration
    logger = get_logger("obsidian_anki_sync.utils.logging")
    logger.info(
        "logging_configured",
        console_level=log_level,
        file_level="DEBUG",
        log_dir=str(log_dir) if not log_file else str(log_file.parent),
        log_file=str(log_file) if log_file else None,
        project_log_dir=str(project_log_dir),
        error_log_dir=str(project_log_dir),
        verbose=verbose,
        very_verbose=very_verbose,
        console_noise_filter=enable_console_noise_filter,
    )


def get_logger(name: str) -> Any:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structlog logger bound to the given name
    """
    if not _configured:
        # Auto-configure with defaults if not configured
        configure_logging()

    # Get structlog logger and bind the name
    logger = structlog.get_logger(name)
    # The logger name is automatically set by add_logger_name processor
    return logger
