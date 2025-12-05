"""Anki-related command implementations."""

import json
from pathlib import Path
from typing import Any

import typer

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.anki.exporter import export_deck_from_anki
from obsidian_anki_sync.anki.importer import (
    import_cards_from_csv,
    import_cards_from_yaml,
)
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.preflight import run_preflight_checks

from .shared import console


def run_list_decks(config: Config, logger: Any) -> None:
    """Execute the list-decks operation.

    Args:
        config: Configuration object
        logger: Logger instance

    Raises:
        typer.Exit: On list-decks failure
    """
    logger.info("list_decks_started")

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            decks = sorted(anki.get_deck_names())

        if not decks:
            console.print("[yellow]No decks available.[/yellow]")
        else:
            console.print("\n[bold]Decks:[/bold]")
            for deck in decks:
                console.print(f"  [cyan]• {deck}[/cyan]")

        logger.info("list_decks_completed", count=len(decks))

    except Exception as e:
        logger.error("list_decks_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_list_models(config: Config, logger: Any) -> None:
    """Execute the list-models operation.

    Args:
        config: Configuration object
        logger: Logger instance

    Raises:
        typer.Exit: On list-models failure
    """
    logger.info("list_models_started")

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            models = sorted(anki.get_model_names())

        if not models:
            console.print("[yellow]No models available.[/yellow]")
        else:
            console.print("\n[bold]Models:[/bold]")
            for model in models:
                console.print(f"  [cyan]• {model}[/cyan]")

        logger.info("list_models_completed", count=len(models))

    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_show_model_fields(config: Config, logger: Any, model_name: str) -> None:
    """Execute the show-model-fields operation.

    Args:
        config: Configuration object
        logger: Logger instance
        model_name: Name of the model to inspect

    Raises:
        typer.Exit: On show-model-fields failure
    """
    logger.info("model_fields_started", model=model_name)

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            fields = anki.get_model_field_names(model_name)

        if not fields:
            console.print(
                f"[yellow]Model '{model_name}' has no fields or does not exist.[/yellow]"
            )
        else:
            console.print(f"\n[bold]Fields for model '{model_name}':[/bold]")
            for field in fields:
                console.print(f"  [cyan]• {field}[/cyan]")

        logger.info("model_fields_completed",
                    model=model_name, count=len(fields))

    except Exception as e:
        logger.error("model_fields_failed", model=model_name, error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_export_deck(
    config: Config,
    logger: Any,
    deck_name: str,
    output: Path | None,
    file_format: str,
    include_note_id: bool,
) -> None:
    """Execute the export-deck operation.

    Args:
        config: Configuration object
        logger: Logger instance
        deck_name: Name of the deck to export
        output: Output file path
        file_format: Output format (yaml or csv)
        include_note_id: Whether to include noteId field

    Raises:
        typer.Exit: On export-deck failure
    """
    # Auto-generate output filename if not provided
    if output is None:
        safe_deck_name = deck_name.lower().replace(" ", "-").replace("::", "-")
        extension = "csv" if file_format.lower() == "csv" else "yaml"
        output = Path(f"{safe_deck_name}.{extension}")

    logger.info(
        "export_deck_started",
        deck_name=deck_name,
        output_path=str(output),
        format=file_format,
    )

    try:
        # Run pre-flight checks
        console.print(
            "\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")
        _passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False
        )

        for result in results:
            icon = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"{icon} {result.name}: {result.message}")

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print("\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Export deck
        client = AnkiClient(config.anki_connect_url)
        export_deck_from_anki(
            client=client,
            deck_name=deck_name,
            output_path=output,
            format=file_format,
            include_note_id=include_note_id,
        )

        console.print(
            f"\n[bold green]Successfully exported deck to {output}[/bold green]"
        )
        logger.info("export_deck_completed", output_path=str(output))

    except Exception as e:
        logger.error("export_deck_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_import_deck(
    config: Config,
    logger: Any,
    input_path: Path,
    deck_name: str,
    note_type: str | None,
    key_field: str | None,
) -> None:
    """Execute the import-deck operation.

    Args:
        config: Configuration object
        logger: Logger instance
        input_path: Path to input file
        deck_name: Target deck name
        note_type: Note type to use
        key_field: Field for identifying existing notes

    Raises:
        typer.Exit: On import-deck failure
    """
    if not input_path.exists():
        console.print(f"\n[bold red]Error:[/bold red] File not found: {input_path}")
        raise typer.Exit(code=1)

    # Detect format from extension
    file_format = input_path.suffix.lower()
    if file_format not in (".yaml", ".yml", ".csv"):
        console.print(
            f"\n[bold red]Error:[/bold red] Unsupported file format: {file_format}"
        )
        console.print("[yellow]Supported formats: .yaml, .yml, .csv[/yellow]")
        raise typer.Exit(code=1)

    logger.info(
        "import_deck_started",
        input_path=str(input_path),
        deck_name=deck_name,
        file_format=file_format,
    )

    try:
        # Run pre-flight checks
        console.print(
            "\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")
        _passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False
        )

        for result in results:
            icon = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"{icon} {result.name}: {result.message}")

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print("\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Import cards
        client = AnkiClient(config.anki_connect_url)
        if file_format == ".csv":
            result = import_cards_from_csv(
                client=client,
                input_path=input_path,
                deck_name=deck_name,
                note_type=note_type,
                key_field=key_field,
            )
        else:
            result = import_cards_from_yaml(
                client=client,
                input_path=input_path,
                deck_name=deck_name,
                note_type=note_type,
                key_field=key_field,
            )

        console.print("\n[bold green]Import complete![/bold green]")
        console.print(f"  Created: {result['created']}")
        console.print(f"  Updated: {result['updated']}")
        if result["errors"] > 0:
            console.print(f"  [yellow]Errors: {result['errors']}[/yellow]")

        logger.info("import_deck_completed", **result)

    except Exception as e:
        logger.error("import_deck_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_query_anki(
    config: Config,
    logger: Any,
    action: str,
    params: str | None,
) -> None:
    """Execute the query-anki operation.

    Args:
        config: Configuration object
        logger: Logger instance
        action: AnkiConnect API action name
        params: JSON parameters string

    Raises:
        typer.Exit: On query-anki failure
    """
    try:
        # Special actions
        if action in ("docs", "help"):
            console.print("\n[cyan]AnkiConnect API Documentation:[/cyan]\n")
            console.print("Available actions include:")
            console.print("  - deckNames: Get all deck names")
            console.print("  - modelNames: Get all note type names")
            console.print("  - findNotes: Find notes matching query")
            console.print("  - notesInfo: Get detailed note information")
            console.print("  - cardsInfo: Get card information")
            console.print("  - getDeckStats: Get deck statistics")
            console.print(
                "\nSee https://foosoft.net/projects/anki-connect/ for full documentation"
            )
            return

        # Parse params if provided
        params_dict = {}
        if params:
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError as e:
                console.print(
                    f"\n[bold red]Error:[/bold red] Invalid JSON parameters: {e}"
                )
                raise typer.Exit(code=1)

        # Run pre-flight checks
        console.print(
            "\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")
        _passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False
        )

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print("\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Execute query
        client = AnkiClient(config.anki_connect_url)
        result = client.invoke(action, params_dict if params_dict else None)

        # Output result as JSON
        console.print("\n[bold green]Result:[/bold green]")
        console.print(json.dumps(result, indent=2, ensure_ascii=False))

        logger.info("query_completed", action=action,
                    has_result=result is not None)

    except Exception as e:
        logger.error("query_failed", error=str(e), action=action)
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
