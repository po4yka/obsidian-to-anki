"""CLI command modules for obsidian-anki-sync.

This package contains modular command handlers for the CLI.
Currently implemented:
- shared.py: Common utilities (config/logger loading, console)
- sync_handler.py: Sync command implementation

Future modules (see REFACTORING_GUIDE.md):
- validate_handler.py: Note validation command
- init_handler.py: Initialization command
- anki_handler.py: Anki-related commands (decks, models, fields)
- export_handler.py: APKG export command
- index_handler.py: Index and progress commands
- format_handler.py: Code formatting command
"""

from .shared import console, get_config_and_logger
from .sync_handler import run_sync

__all__ = [
    "console",
    "get_config_and_logger",
    "run_sync",
]
