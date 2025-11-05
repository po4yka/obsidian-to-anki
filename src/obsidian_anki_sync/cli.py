"""Command-line interface for the sync service."""

import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .config import Config, load_config, set_config
from .utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="obsidian-anki-sync",
    help="Obsidian to Anki APF sync service.",
    no_args_is_help=True,
)

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
        configure_logging(log_level or _config.log_level)
        _logger = get_logger("cli")

    return _config, _logger


@app.command()
def sync(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Preview changes without applying")
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume a previous interrupted sync by session ID"),
    ] = None,
    no_resume: Annotated[
        bool,
        typer.Option("--no-resume", help="Disable automatic resume of incomplete syncs"),
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

    logger.info("sync_started", dry_run=dry_run, vault=str(config.vault_path))

    from .anki.client import AnkiClient
    from .sync.engine import SyncEngine
    from .sync.progress import ProgressTracker
    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            # Check for incomplete syncs or resume request
            progress_tracker = None
            session_id = resume

            if not no_resume:
                if not session_id:
                    # Check for incomplete syncs
                    incomplete = db.get_incomplete_progress()
                    if incomplete:
                        latest = incomplete[0]
                        console.print(
                            f"\n[yellow]Found incomplete sync from {latest['updated_at']}[/yellow]"
                        )
                        console.print(
                            f"  Session: {latest['session_id']}"
                        )
                        console.print(
                            f"  Progress: {latest['notes_processed']}/{latest['total_notes']} notes"
                        )

                        # Ask user if they want to resume
                        resume_choice = typer.confirm(
                            "Resume this sync?", default=True
                        )
                        if resume_choice:
                            session_id = latest['session_id']

                # Create or resume progress tracker
                if session_id:
                    try:
                        progress_tracker = ProgressTracker(db, session_id=session_id)
                        console.print(
                            f"\n[green]Resuming sync session: {session_id}[/green]\n"
                        )
                    except ValueError as e:
                        console.print(f"\n[red]Cannot resume: {e}[/red]\n")
                        return
                else:
                    # Start new sync with progress tracking
                    progress_tracker = ProgressTracker(db)
                    console.print(
                        f"\n[cyan]Starting new sync session: {progress_tracker.progress.session_id}[/cyan]\n"
                    )

            engine = SyncEngine(config, db, anki, progress_tracker=progress_tracker)
            stats = engine.sync(dry_run=dry_run)

            # Create a Rich table for results
            table = Table(
                title="Sync Results", show_header=True, header_style="bold magenta"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                table.add_row(key, str(value))

            console.print()
            console.print(table)

            logger.info("sync_completed", stats=stats)

    except Exception as e:
        logger.error("sync_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


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
            console.print("\n[bold green]✓ Validation passed![/bold green]")
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
        console.print(f"[green]✓ Created .env template at {env_path}[/green]")
        logger.info("env_template_created", path=str(env_path))
    else:
        console.print(f"[yellow].env already exists at {env_path}[/yellow]")

    # Initialize database
    from .sync.state_db import StateDB

    db = StateDB(config.db_path)
    db.close()

    console.print(f"[green]✓ Initialized database at {config.db_path}[/green]")
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
            source_path = config.vault_path / config.source_dir
            note_paths = discover_notes(source_path)

            if sample_size:
                import random

                note_paths = random.sample(
                    note_paths, min(sample_size, len(note_paths))
                )

            console.print(f"[cyan]Processing {len(note_paths)} notes...[/cyan]")

            # Generate cards using the sync engine's generation logic
            from .apf.generator import APFGenerator

            cards = []

            if config.use_agent_system:
                console.print("[cyan]Using multi-agent system for generation...[/cyan]")
                from .agents.orchestrator import AgentOrchestrator

                orchestrator = AgentOrchestrator(config)

                for note_path in note_paths:
                    try:
                        metadata, qa_pairs = parse_note(note_path)
                        result = orchestrator.process_note(
                            metadata, qa_pairs, note_path
                        )

                        if result.success:
                            cards.extend(result.cards)
                            console.print(
                                f"  [green]✓[/green] {metadata.title} "
                                f"({len(result.cards)} cards)"
                            )
                        else:
                            console.print(
                                f"  [red]✗[/red] {metadata.title}: "
                                f"{result.error_message}"
                            )
                    except Exception as e:
                        console.print(f"  [red]✗[/red] {note_path.name}: {e}")

            else:
                console.print("[cyan]Using OpenRouter for generation...[/cyan]")
                generator = APFGenerator(
                    api_key=config.openrouter_api_key,
                    model=config.openrouter_model,
                    temperature=config.llm_temperature,
                    top_p=config.llm_top_p,
                )

                for note_path in note_paths:
                    try:
                        metadata, qa_pairs = parse_note(note_path)

                        for qa in qa_pairs:
                            for lang in metadata.language_tags:
                                card = generator.generate_card(
                                    metadata, qa, lang, note_path
                                )
                                cards.append(card)

                        console.print(
                            f"  [green]✓[/green] {metadata.title} "
                            f"({len(qa_pairs) * len(metadata.language_tags)} cards)"
                        )
                    except Exception as e:
                        console.print(f"  [red]✗[/red] {note_path.name}: {e}")

            if not cards:
                console.print("\n[yellow]No cards generated. Exiting.[/yellow]")
                return

            # Export to .apkg
            console.print(
                f"\n[cyan]Exporting {len(cards)} cards to {output_path}...[/cyan]"
            )

            export_cards_to_apkg(
                cards=cards,
                output_path=output_path,
                deck_name=final_deck_name,
                deck_description=final_description,
            )

            console.print(
                f"\n[bold green]✓ Successfully exported {len(cards)} cards "
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
                    progress_str = f"{session['notes_processed']}/{session['total_notes']} notes"
                    table.add_row(
                        session['session_id'][:8] + "...",
                        session['phase'],
                        progress_str,
                        session['updated_at'],
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
                    progress_str = f"{session['notes_processed']}/{session['total_notes']}"
                    table.add_row(
                        session['session_id'][:8] + "...",
                        session['phase'],
                        progress_str,
                        str(session['errors']),
                        session['started_at'],
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
                    f"[green]✓ Deleted progress for session: {session_id}[/green]"
                )
                logger.info("progress_deleted", session_id=session_id)

            elif all_completed:
                # Get all completed sessions and delete them
                all_progress = db.get_all_progress(limit=1000)
                deleted_count = 0

                for session in all_progress:
                    if session['phase'] in ('completed', 'failed'):
                        db.delete_progress(session['session_id'])
                        deleted_count += 1

                console.print(
                    f"[green]✓ Deleted {deleted_count} completed sessions[/green]"
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

        console.print("[green]✓ Format completed successfully[/green]")
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
