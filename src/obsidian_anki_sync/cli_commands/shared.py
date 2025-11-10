"""Shared utilities for CLI commands."""

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ..config import Config, load_config, set_config
from ..utils.logging import configure_logging, get_logger

# Shared console for all commands
console = Console()

# Global state for config and logger (cached for performance across CLI commands)
# Note: This is a simple caching mechanism. For multi-threaded/async usage,
# consider using a proper dependency injection framework or context manager.
_config: Config | None = None
_logger: Any | None = None


def get_config_and_logger(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> tuple[Config, Any]:
    """Load configuration and logger (dependency injection helper).

    This function uses module-level caching to avoid reloading config
    for each CLI command invocation. The cache is cleared when the
    Python process exits.

    Args:
        config_path: Optional path to config file
        log_level: Logging level

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

        # Configure logging with vault-specific log directory
        log_dir = Path(_config.vault_path) / ".logs" if _config.vault_path else None
        configure_logging(log_level or _config.log_level, log_dir=log_dir)
        _logger = get_logger("cli")

    return _config, _logger
