"""CLI command modules for obsidian-anki-sync.

This package contains modular command handlers for the CLI.
Implemented modules:
- shared.py: Common utilities (config/logger loading, console)
- sync_handler.py: Sync command implementation
- test_run_handler.py: Test-run command implementation
- init_handler.py: Initialization command implementation
- anki_handler.py: Anki-related commands (decks, models, fields, export-deck, import-deck, query)
- export_handler.py: APKG export command implementation
- generate_handler.py: Generate command implementation
- index_handler.py: Index and progress commands implementation
- check_handler.py: Check command implementation
- log_handler.py: Log analysis commands implementation
- format_handler.py: Code formatting command implementation
- validate_commands.py: Note validation commands (includes lint-note and validate)
"""

from .anki_handler import (
    run_export_deck,
    run_import_deck,
    run_list_decks,
    run_list_models,
    run_query_anki,
    run_show_model_fields,
)
from .check_handler import run_check_setup
from .export_handler import run_export
from .format_handler import run_format
from .generate_handler import run_generate_cards
from .index_handler import (
    run_clean_progress,
    run_show_index,
    run_show_progress,
)
from .init_handler import run_init
from .log_handler import (
    run_analyze_logs,
    run_list_problematic_notes,
)
from .shared import console, get_config_and_logger
from .sync_handler import run_sync
from .test_run_handler import run_test_run
from .validate_commands import (
    run_lint_note,
    run_validate,
)

__all__ = [
    "console",
    "get_config_and_logger",
    "run_analyze_logs",
    "run_check_setup",
    "run_clean_progress",
    "run_export",
    "run_export_deck",
    "run_format",
    "run_generate_cards",
    "run_import_deck",
    "run_init",
    "run_lint_note",
    "run_list_decks",
    "run_list_models",
    "run_list_problematic_notes",
    "run_query_anki",
    "run_show_index",
    "run_show_model_fields",
    "run_show_progress",
    "run_sync",
    "run_test_run",
    "run_validate",
]
