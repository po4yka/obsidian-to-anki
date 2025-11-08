"""Logging configuration using loguru for better console and file output."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger


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

    # Add console handler - concise format for readability
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # Add file handler - detailed format with rotation
    if log_dir is None:
        log_dir = Path("./logs")

    log_dir.mkdir(exist_ok=True, parents=True)

    logger.add(
        log_dir / "obsidian-anki-sync_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # File gets all logs
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        backtrace=True,  # Include traceback
        diagnose=True,  # Include variable values in tracebacks
        enqueue=True,  # Thread-safe
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
