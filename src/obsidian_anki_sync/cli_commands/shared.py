"""Shared utilities for CLI commands."""

from pathlib import Path
from typing import Any

from rich.console import Console

from obsidian_anki_sync.config import Config, load_config, set_config
from obsidian_anki_sync.utils.logging import configure_logging, get_logger

# Shared console for all commands
console = Console()

# Global state for config and logger (cached for performance across CLI commands)
# For multi-threaded/async usage, consider using a proper dependency injection framework or context manager.
_config: Config | None = None
_logger: Any | None = None


def get_config_and_logger(
    config_path: Path | None = None,
    log_level: str = "INFO",
    log_file: Path | None = None,
    very_verbose: bool = False,
    verbose: bool = False,
) -> tuple[Config, Any]:
    """Load configuration and logger (dependency injection helper).

    This function uses module-level caching to avoid reloading config
    for each CLI command invocation. The cache is cleared when the
    Python process exits.

    Args:
        config_path: Optional path to config file
        log_level: Logging level
        log_file: Optional specific log file path
        very_verbose: Enable very verbose logging (full LLM requests/responses)
        verbose: Show all log messages on terminal (for debugging)

    Returns:
        Tuple of (Config, Logger)

    Note:
        This caching mechanism is not thread-safe. For concurrent usage,
        consider using a proper dependency injection framework.
    """
    global _config, _logger

    if _config is None:
        _config = load_config(config_path)
        set_config(_config)

        # Configure logging with data_dir log directory (not in vault)
        log_dir = _config.get_log_dir() if _config.vault_path else None
        configure_logging(
            log_level or _config.log_level,
            log_dir=log_dir,
            log_file=log_file,
            very_verbose=very_verbose,
            verbose=verbose,
            project_log_dir=_config.get_log_dir(),
            error_log_retention_days=_config.error_log_retention_days,
            compress_old_error_logs=_config.compress_error_logs,
        )
        _logger = get_logger("cli")

    return _config, _logger
