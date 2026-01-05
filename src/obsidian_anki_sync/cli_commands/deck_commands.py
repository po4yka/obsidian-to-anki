"""Deck-related CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .anki_handler import (
    run_export_deck,
    run_import_deck,
    run_list_decks,
    run_list_models,
    run_show_model_fields,
)
from .export_handler import run_export
from .shared import get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register deck-related commands on the given Typer app."""

    @app.command(name="decks")
    def list_decks(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """List deck names available via AnkiConnect."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_list_decks(config=config, logger=logger)

    @app.command(name="models")
    def list_models(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """List note models (types) available in Anki."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_list_models(config=config, logger=logger)

    @app.command(name="model-fields")
    def show_model_fields(
        model_name: Annotated[
            str, typer.Option("--model", help="Model name to inspect")
        ],
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Show field names for a specific Anki model."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_show_model_fields(config=config, logger=logger, model_name=model_name)

    @app.command()
    def export(
        output: Annotated[
            Path | None,
            typer.Option("--output", "-o", help="Output .apkg file path"),
        ] = None,
        deck_name: Annotated[
            str | None,
            typer.Option("--deck-name", help="Name for the exported deck"),
        ] = None,
        deck_description: Annotated[
            str,
            typer.Option("--deck-description", help="Description for the deck"),
        ] = "",
        sample_size: Annotated[
            int | None,
            typer.Option("--sample", help="Export only N random notes (for testing)"),
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Export Obsidian notes to Anki deck file (.apkg)."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_export(
            config=config,
            logger=logger,
            output=output,
            deck_name=deck_name,
            deck_description=deck_description,
            sample_size=sample_size,
        )

    @app.command(name="export-deck")
    def export_deck(
        deck_name: Annotated[
            str, typer.Argument(help="Name of the Anki deck to export")
        ],
        output: Annotated[
            Path | None,
            typer.Option(
                "--output", "-o", help="Output file path (auto-generated if omitted)"
            ),
        ] = None,
        file_format: Annotated[
            str,
            typer.Option(
                "--format",
                "-f",
                help="Output format: yaml or csv (default: yaml)",
            ),
        ] = "yaml",
        include_note_id: Annotated[
            bool,
            typer.Option(
                "--include-note-id/--no-note-id",
                help="Include noteId field for updates (default: true)",
            ),
        ] = True,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Export an Anki deck to YAML or CSV file."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_export_deck(
            config=config,
            logger=logger,
            deck_name=deck_name,
            output=output,
            file_format=file_format,
            include_note_id=include_note_id,
        )

    @app.command(name="import-deck")
    def import_deck(
        input: Annotated[
            Path,
            typer.Argument(help="Path to input YAML or CSV file"),
        ],
        deck_name: Annotated[
            str | None,
            typer.Option(
                "--deck-name",
                "-d",
                help="Target deck name (uses file deck if omitted)",
            ),
        ] = None,
        note_type: Annotated[
            str | None,
            typer.Option(
                "--note-type",
                "-n",
                help="Note type to use (auto-detected if not specified)",
            ),
        ] = None,
        key_field: Annotated[
            str | None,
            typer.Option(
                "--key-field",
                "-k",
                help="Field to use for identifying existing notes (auto-detected if not specified)",
            ),
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Import cards from YAML or CSV file into Anki deck."""
        config, logger = get_config_and_logger(config_path, log_level)
        run_import_deck(
            config=config,
            logger=logger,
            input_path=input,
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
        )
