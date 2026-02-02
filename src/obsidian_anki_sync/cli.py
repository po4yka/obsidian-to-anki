"""Command-line interface for the sync service."""

from __future__ import annotations

import typer

from .cli_commands import (
    core_commands,
    deck_commands,
    generation_commands,
    log_commands,
    maintenance_commands,
    processing_commands,
    progress_commands,
    rag_commands,
    validate_commands,
)

app = typer.Typer(
    name="obsidian-anki-sync",
    help="Obsidian to Anki APF sync service.",
    no_args_is_help=True,
)

app.add_typer(
    rag_commands.rag_app,
    name="rag",
    help="RAG (Retrieval-Augmented Generation) commands",
)
app.add_typer(
    validate_commands.validate_app,
    name="validate",
    help="Vault validation commands for Q&A notes",
)

core_commands.register(app)
deck_commands.register(app)
processing_commands.register(app)
progress_commands.register(app)
log_commands.register(app)
generation_commands.register(app)
maintenance_commands.register(app)


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
