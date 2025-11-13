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


def configure_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure loguru logging with dual output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
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

    # Add file handler - detailed format with rotation
    if log_dir is None:
        log_dir = Path("./logs")

    log_dir.mkdir(exist_ok=True, parents=True)

    logger.add(
        log_dir / "obsidian-anki-sync_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}{extra[_formatted]}",
        level="DEBUG",  # File gets all logs
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        backtrace=True,  # Include traceback
        diagnose=True,  # Include variable values in tracebacks
        enqueue=True,  # Thread-safe
        filter=_add_formatted_extra,
    )

    logger.info(
        "logging_configured",
        console_level=level,
        file_level="DEBUG",
        log_dir=str(log_dir),
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
