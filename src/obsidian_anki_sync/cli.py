"""Command-line interface for the sync service."""

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .cli_commands.shared import get_config_and_logger
from .cli_commands.sync_handler import run_sync
from .utils.preflight import run_preflight_checks

app = typer.Typer(
    name="obsidian-anki-sync",
    help="Obsidian to Anki APF sync service.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def sync(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without applying")
    ] = False,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
            help="Only process new notes not yet synced (skip existing notes)",
        ),
    ] = False,
    no_index: Annotated[
        bool,
        typer.Option(
            "--no-index",
            help="Skip indexing phase (not recommended)",
        ),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume", help="Resume a previous interrupted sync by session ID"
        ),
    ] = None,
    no_resume: Annotated[
        bool,
        typer.Option(
            "--no-resume", help="Disable automatic resume of incomplete syncs"
        ),
    ] = False,
    use_agents: Annotated[
        bool | None,
        typer.Option(
            "--use-agents/--no-agents",
            help="Use multi-agent system for card generation (requires Ollama)",
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
    """Synchronize Obsidian notes to Anki cards."""
    config, logger = get_config_and_logger(config_path, log_level)

    # Override agent system setting if CLI flag is provided
    if use_agents is not None:
        config.use_agent_system = use_agents
        logger.info("agent_system_override", use_agents=use_agents)

    # Delegate to sync handler
    run_sync(
        config=config,
        logger=logger,
        dry_run=dry_run,
        incremental=incremental,
        no_index=no_index,
        resume=resume,
        no_resume=no_resume,
    )


@app.command(name="test-run")
def test_run(
    count: Annotated[
        int,
        typer.Option(
            "--count",
            min=1,
            help="Number of random notes to process (dry-run)",
        ),
    ] = 10,
    use_agents: Annotated[
        bool | None,
        typer.Option(
            "--use-agents/--no-agents",
            help="Use multi-agent system for card generation (requires Ollama)",
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
    """Run a sample dry-run by processing N random notes."""
    config, logger = get_config_and_logger(config_path, log_level)

    # Override agent system setting if CLI flag is provided
    if use_agents is not None:
        config.use_agent_system = use_agents
        logger.info("agent_system_override", use_agents=use_agents)

    logger.info("test_run_started", sample_count=count)

    # Run pre-flight checks (skip Anki since it's a dry-run)
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    passed, results = run_preflight_checks(config, check_anki=False, check_llm=True)

    # Display results
    for result in results:
        if result.passed:
            icon = "[green]✓[/green]"
        elif result.severity == "warning":
            icon = "[yellow]⚠[/yellow]"
        else:
            icon = "[red]✗[/red]"

        console.print(f"{icon} {result.name}: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]{result.fix_suggestion}[/dim]")

    console.print()

    # Check for errors
    errors = [r for r in results if not r.passed and r.severity == "error"]

    if errors:
        console.print(f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]")
        console.print("[yellow]Fix the errors above and try again.[/yellow]\n")
        raise typer.Exit(code=1)

    console.print("[bold green]All pre-flight checks passed![/bold green]\n")

    from .anki.client import AnkiClient
    from .sync.engine import SyncEngine
    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            engine = SyncEngine(config, db, anki)
            stats = engine.sync(dry_run=True, sample_size=count)

            console.print(
                f"\n[cyan]Processed sample of {count} notes (dry-run).[/cyan]"
            )

            # Create a Rich table for results
            table = Table(
                title="Sample Results", show_header=True, header_style="bold magenta"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                table.add_row(key, str(value))

            console.print(table)

            logger.info("test_run_completed", stats=stats)

    except Exception as e:
        logger.error("test_run_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def validate(
    note_path: Annotated[Path, typer.Argument(help="Path to note file", exists=True)],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Validate note structure and APF compliance."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("validate_started", note_path=str(note_path))

    from .obsidian.parser import parse_note
    from .obsidian.validator import validate_note

    try:
        metadata, qa_pairs = parse_note(note_path)

        # Display parsed information
        console.print()
        console.print(f"[bold]Parsed:[/bold] {note_path}")
        console.print(f"  [cyan]Title:[/cyan] {metadata.title}")
        console.print(f"  [cyan]Topic:[/cyan] {metadata.topic}")
        console.print(f"  [cyan]Languages:[/cyan] {', '.join(metadata.language_tags)}")
        console.print(f"  [cyan]Q/A pairs:[/cyan] {len(qa_pairs)}")

        # Validate
        errors = validate_note(metadata, qa_pairs, note_path)

        if errors:
            console.print("\n[bold red]Validation errors:[/bold red]")
            for error in errors:
                console.print(f"  [red]- {error}[/red]")
            logger.error("validation_failed", errors=errors)
        else:
            console.print("\n[bold green] Validation passed![/bold green]")
            logger.info("validation_passed")

    except Exception as e:
        logger.error("validation_error", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    logger.info("validate_completed")


@app.command()
def init(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Initialize configuration and database."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("init_started")

    # Create .env template if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        env_template = """# Obsidian configuration
VAULT_PATH=/path/to/your/vault
SOURCE_DIR=interview_questions/InterviewQuestions

# Anki configuration
ANKI_CONNECT_URL=http://127.0.0.1:8765
ANKI_DECK_NAME=Interview Questions
ANKI_NOTE_TYPE=APF::Simple

# Deck export configuration (for .apkg file generation)
# EXPORT_DECK_NAME=Interview Questions  # Defaults to ANKI_DECK_NAME
# EXPORT_DECK_DESCRIPTION=Generated from Obsidian notes
# EXPORT_OUTPUT_PATH=output.apkg

# OpenRouter LLM configuration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.3

# Runtime configuration
RUN_MODE=apply
DELETE_MODE=delete
DB_PATH=.sync_state.db
LOG_LEVEL=INFO
"""
        env_path.write_text(env_template)
        console.print(f"[green] Created .env template at {env_path}[/green]")
        logger.info("env_template_created", path=str(env_path))
    else:
        console.print(f"[yellow].env already exists at {env_path}[/yellow]")

    # Initialize database
    from .sync.state_db import StateDB

    with StateDB(config.db_path):
        pass  # Database schema is initialized in __init__

    console.print(f"[green] Initialized database at {config.db_path}[/green]")
    logger.info("database_initialized", path=str(config.db_path))

    logger.info("init_completed")


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

    logger.info("list_decks_started")

    from .anki.client import AnkiClient

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

    logger.info("list_models_started")

    from .anki.client import AnkiClient

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


@app.command(name="model-fields")
def show_model_fields(
    model_name: Annotated[str, typer.Option("--model", help="Model name to inspect")],
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

    logger.info("model_fields_started", model=model_name)

    from .anki.client import AnkiClient

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

        logger.info("model_fields_completed", model=model_name, count=len(fields))

    except Exception as e:
        logger.error("model_fields_failed", model=model_name, error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


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
    use_agents: Annotated[
        bool | None,
        typer.Option(
            "--use-agents/--no-agents",
            help="Use multi-agent system for card generation (requires Ollama)",
        ),
    ] = None,
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

    # Override agent system setting if CLI flag is provided
    if use_agents is not None:
        config.use_agent_system = use_agents
        logger.info("agent_system_override", use_agents=use_agents)

    # Determine output path
    output_path = output or config.export_output_path or Path("output.apkg")

    # Determine deck name
    final_deck_name = deck_name or config.export_deck_name or config.anki_deck_name

    # Determine deck description
    final_description = deck_description or config.export_deck_description or ""

    logger.info(
        "export_started",
        output_path=str(output_path),
        deck_name=final_deck_name,
        sample_size=sample_size,
    )

    # Run pre-flight checks (skip Anki since we're exporting to file)
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    passed, results = run_preflight_checks(config, check_anki=False, check_llm=True)

    # Display results
    for result in results:
        if result.passed:
            icon = "[green]✓[/green]"
        elif result.severity == "warning":
            icon = "[yellow]⚠[/yellow]"
        else:
            icon = "[red]✗[/red]"

        console.print(f"{icon} {result.name}: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]{result.fix_suggestion}[/dim]")

    console.print()

    # Check for errors
    errors = [r for r in results if not r.passed and r.severity == "error"]

    if errors:
        console.print(f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]")
        console.print("[yellow]Fix the errors above and try again.[/yellow]\n")
        raise typer.Exit(code=1)

    console.print("[bold green]All pre-flight checks passed![/bold green]\n")

    from .anki.exporter import export_cards_to_apkg
    from .obsidian.parser import discover_notes, parse_note
    from .sync.state_db import StateDB

    try:
        # Generate cards by processing notes
        console.print("\n[cyan]Generating cards from Obsidian notes...[/cyan]")

        with StateDB(config.db_path) as _:
            # Use a dummy Anki client (won't connect)

            # We don't need AnkiConnect for export, but SyncEngine expects it
            # We'll use the sync engine's card generation logic
            note_paths = discover_notes(config.vault_path, config.source_dir)

            if sample_size:
                import random

                note_paths = random.sample(
                    note_paths, min(sample_size, len(note_paths))
                )

            console.print(f"[cyan]Processing {len(note_paths)} notes...[/cyan]")

            # Generate cards using the sync engine's generation logic
            from typing import Any

            from .apf.generator import APFGenerator

            cards: list[Any] = (
                []
            )  # Can be list[Card] or list[GeneratedCard] depending on path

            if config.use_agent_system:
                console.print("[cyan]Using multi-agent system for generation...[/cyan]")
                from .agents.orchestrator import AgentOrchestrator

                orchestrator = AgentOrchestrator(config)

                for note_path_tuple in note_paths:
                    try:
                        note_path, note_content = note_path_tuple
                        metadata, qa_pairs = parse_note(note_path)
                        result = orchestrator.process_note(
                            note_content, metadata, qa_pairs, note_path
                        )

                        if result.success and result.generation:
                            generated_cards = result.generation.cards
                            cards.extend(generated_cards)
                            console.print(
                                f"  [green][/green] {metadata.title} "
                                f"({len(generated_cards)} cards)"
                            )
                        else:
                            error_msg = "Pipeline failed"
                            if result.post_validation:
                                error_msg = (
                                    result.post_validation.error_details or error_msg
                                )
                            console.print(
                                f"  [red][/red] {metadata.title}: {error_msg}"
                            )
                    except Exception as e:
                        console.print(f"  [red][/red] {note_path.name}: {e}")

            else:
                console.print("[cyan]Using OpenRouter for generation...[/cyan]")
                generator = APFGenerator(config)

                for note_path_tuple in note_paths:
                    try:
                        note_path, _ = note_path_tuple
                        metadata, qa_pairs = parse_note(note_path)

                        for qa in qa_pairs:
                            for lang in metadata.language_tags:
                                card = generator.generate_card(
                                    metadata, qa, lang, note_path
                                )
                                cards.append(card)

                        console.print(
                            f"  [green][/green] {metadata.title} "
                            f"({len(qa_pairs) * len(metadata.language_tags)} cards)"
                        )
                    except Exception as e:
                        console.print(f"  [red][/red] {note_path.name}: {e}")

            if not cards:
                console.print("\n[yellow]No cards generated. Exiting.[/yellow]")
                return

            # Export to .apkg
            console.print(
                f"\n[cyan]Exporting {len(cards)} cards to {output_path}...[/cyan]"
            )

            export_cards_to_apkg(
                cards=cards,  # type: ignore[arg-type]
                output_path=output_path,
                deck_name=final_deck_name,
                deck_description=final_description,
            )

            console.print(
                f"\n[bold green] Successfully exported {len(cards)} cards "
                f"to {output_path}[/bold green]"
            )

            console.print("\n[cyan]Import this file into Anki to add the cards.[/cyan]")

            logger.info(
                "export_completed",
                output_path=str(output_path),
                card_count=len(cards),
            )

    except Exception as e:
        logger.error("export_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="index")
def show_index(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Show vault and Anki card index statistics."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("show_index_started")

    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db:
            stats = db.get_index_statistics()

            if stats["total_notes"] == 0 and stats["total_cards"] == 0:
                console.print(
                    "\n[yellow]Index is empty. Run sync to build the index.[/yellow]"
                )
                return

            # Display index statistics
            console.print("\n[bold cyan]Vault & Anki Index:[/bold cyan]\n")

            # Notes table
            notes_table = Table(
                title="Notes Index", show_header=True, header_style="bold magenta"
            )
            notes_table.add_column("Metric", style="cyan")
            notes_table.add_column("Count", style="green")

            notes_table.add_row("Total Notes", str(stats["total_notes"]))

            note_status = stats.get("note_status", {})
            for status, count in sorted(note_status.items()):
                notes_table.add_row(f"  {status.capitalize()}", str(count))

            console.print(notes_table)
            console.print()

            # Cards table
            cards_table = Table(
                title="Cards Index", show_header=True, header_style="bold magenta"
            )
            cards_table.add_column("Metric", style="cyan")
            cards_table.add_column("Count", style="green")

            cards_table.add_row("Total Cards", str(stats["total_cards"]))
            cards_table.add_row("In Obsidian", str(stats["cards_in_obsidian"]))
            cards_table.add_row("In Anki", str(stats["cards_in_anki"]))
            cards_table.add_row("In Database", str(stats["cards_in_database"]))

            console.print(cards_table)
            console.print()

            # Card status breakdown
            card_status = stats.get("card_status", {})
            if card_status:
                console.print("[bold]Card Status Breakdown:[/bold]")
                for status, count in sorted(card_status.items()):
                    console.print(f"  [cyan]{status}:[/cyan] {count}")
                console.print()

        logger.info("show_index_completed")

    except Exception as e:
        logger.error("show_index_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="progress")
def show_progress(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Show recent sync progress and incomplete sessions."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("show_progress_started")

    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db:
            # Get incomplete syncs
            incomplete = db.get_incomplete_progress()

            if incomplete:
                console.print("\n[bold yellow]Incomplete Syncs:[/bold yellow]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Session ID", style="cyan")
                table.add_column("Phase", style="yellow")
                table.add_column("Progress", style="green")
                table.add_column("Updated At", style="blue")

                for session in incomplete:
                    progress_str = (
                        f"{session['notes_processed']}/{session['total_notes']} notes"
                    )
                    table.add_row(
                        session["session_id"][:8] + "...",
                        session["phase"],
                        progress_str,
                        session["updated_at"],
                    )

                console.print(table)
                console.print(
                    "\n[cyan]Resume with: obsidian-anki-sync sync --resume <session-id>[/cyan]"
                )
            else:
                console.print("\n[green]No incomplete syncs found.[/green]")

            # Get recent syncs
            recent = db.get_all_progress(limit=10)

            if recent:
                console.print("\n[bold]Recent Syncs:[/bold]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Session ID", style="cyan")
                table.add_column("Phase", style="yellow")
                table.add_column("Progress", style="green")
                table.add_column("Errors", style="red")
                table.add_column("Started At", style="blue")

                for session in recent:
                    progress_str = (
                        f"{session['notes_processed']}/{session['total_notes']}"
                    )
                    table.add_row(
                        session["session_id"][:8] + "...",
                        session["phase"],
                        progress_str,
                        str(session["errors"]),
                        session["started_at"],
                    )

                console.print(table)

        logger.info("show_progress_completed")

    except Exception as e:
        logger.error("show_progress_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="clean-progress")
def clean_progress(
    session_id: Annotated[
        str | None,
        typer.Option("--session", help="Specific session ID to delete"),
    ] = None,
    all_completed: Annotated[
        bool,
        typer.Option("--all-completed", help="Delete all completed sessions"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Clean up sync progress records."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("clean_progress_started")

    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db:
            if session_id:
                db.delete_progress(session_id)
                console.print(
                    f"[green] Deleted progress for session: {session_id}[/green]"
                )
                logger.info("progress_deleted", session_id=session_id)

            elif all_completed:
                # Get all completed sessions and delete them
                all_progress = db.get_all_progress(limit=1000)
                deleted_count = 0

                for session in all_progress:
                    if session["phase"] in ("completed", "failed"):
                        db.delete_progress(session["session_id"])
                        deleted_count += 1

                console.print(
                    f"[green] Deleted {deleted_count} completed sessions[/green]"
                )
                logger.info("completed_progress_deleted", count=deleted_count)

            else:
                console.print(
                    "[yellow]Please specify --session <id> or --all-completed[/yellow]"
                )

        logger.info("clean_progress_completed")

    except Exception as e:
        logger.error("clean_progress_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="check")
def check_setup(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
    skip_anki: Annotated[
        bool,
        typer.Option("--skip-anki", help="Skip Anki connectivity check"),
    ] = False,
    skip_llm: Annotated[
        bool,
        typer.Option("--skip-llm", help="Skip LLM provider connectivity check"),
    ] = False,
) -> None:
    """Run pre-flight checks to validate your setup."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("check_setup_started")

    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    passed, results = run_preflight_checks(
        config, check_anki=not skip_anki, check_llm=not skip_llm
    )

    # Display results with detailed formatting
    for result in results:
        if result.passed:
            icon = "[green]✓[/green]"
            color = "green"
        elif result.severity == "warning":
            icon = "[yellow]⚠[/yellow]"
            color = "yellow"
        else:
            icon = "[red]✗[/red]"
            color = "red"

        console.print(f"{icon} [bold]{result.name}[/bold]: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]💡 {result.fix_suggestion}[/dim]")

    console.print()

    # Count errors and warnings
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    passed_checks = [r for r in results if r.passed]

    # Display summary table
    summary_table = Table(title="Check Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Status", style="cyan")
    summary_table.add_column("Count", style="green")

    summary_table.add_row("✓ Passed", str(len(passed_checks)))
    summary_table.add_row("⚠ Warnings", str(len(warnings)))
    summary_table.add_row("✗ Errors", str(len(errors)))

    console.print(summary_table)
    console.print()

    if errors:
        console.print("[bold red]❌ Setup validation failed![/bold red]")
        console.print("[yellow]Fix the errors above before running sync.[/yellow]\n")
        logger.error("check_setup_failed", errors=len(errors), warnings=len(warnings))
        raise typer.Exit(code=1)
    elif warnings:
        console.print("[bold yellow]⚠️  Setup validation passed with warnings.[/bold yellow]")
        console.print("[dim]You may proceed, but some features may not work as expected.[/dim]\n")
        logger.info("check_setup_passed_with_warnings", warnings=len(warnings))
    else:
        console.print("[bold green]✅ All checks passed! Your setup is ready.[/bold green]\n")
        logger.info("check_setup_passed")


@app.command()
def format(
    check: Annotated[
        bool,
        typer.Option("--check", help="Run formatters in check mode (no modifications)"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Run code formatters (ruff + black)."""
    config, logger = get_config_and_logger(config_path, log_level)
    logger.info("format_started", check=check)

    paths = ["src", "tests"]
    commands = [
        (["ruff", "check"], ["--fix"] if not check else [], paths),
        (["black"], ["--check"] if check else [], paths),
    ]

    try:
        for base, extra, target_paths in commands:
            cmd = base + extra + target_paths
            logger.debug("format_command", command=cmd)

            # Display what we're running
            mode = "Checking" if check else "Formatting"
            tool = base[0]
            console.print(f"[cyan]{mode} with {tool}...[/cyan]")

            subprocess.run(cmd, check=True)

        console.print("[green] Format completed successfully[/green]")
        logger.info("format_completed", check=check)
    except subprocess.CalledProcessError as exc:
        logger.error("format_failed", returncode=exc.returncode, cmd=exc.cmd)
        console.print(f"[bold red]Error:[/bold red] Formatter failed: {exc.cmd}")
        raise typer.Exit(code=1)


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
