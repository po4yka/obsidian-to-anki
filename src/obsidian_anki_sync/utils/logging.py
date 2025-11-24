"""Logging configuration using loguru for better console and file output."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger


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


def configure_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    log_file: Path | None = None,
    very_verbose: bool = False,
    project_log_dir: Path | None = None,
    error_log_retention_days: int = 90,
) -> None:
    """Configure loguru logging with dual output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_file: Specific log file path (overrides log_dir)
        very_verbose: If True, log full LLM requests/responses
        project_log_dir: Directory for project-level logs (default: ./logs)
        error_log_retention_days: Days to retain error logs (default: 90)
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    level = log_level.upper()

    # Add console handler - concise format with structured fields
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level><blue>{extra[_formatted]}</blue>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        filter=_add_formatted_extra,
    )

    # Add file handler - detailed format with rotation (vault-level or custom)
    if log_file:
        # Use specific log file
        log_path = log_file
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
        rotation="00:00" if not log_file else None,
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
    project_log_path = project_log_dir / \
        "obsidian-anki-sync_{time:YYYY-MM-DD}.log"

    logger.add(
        project_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}{extra[_formatted]}",
        level="DEBUG",  # Project logs get all levels
        rotation="00:00",
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
        rotation="00:00",
        retention=f"{error_log_retention_days} days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        filter=lambda record: error_filter(
            record) and _add_formatted_extra(record),
    )

    if very_verbose:
        # Add a separate handler for very verbose LLM logging
        verbose_log_path = log_path.parent / \
            f"{log_path.stem}_verbose{log_path.suffix}" if log_file else log_dir / \
            "obsidian-anki-sync_verbose_{time:YYYY-MM-DD}.log"
        logger.add(
            verbose_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="DEBUG",
            filter=lambda record: "llm" in record["name"].lower(
            ) or "prompt" in record["message"].lower() or "response" in record["message"].lower(),
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
