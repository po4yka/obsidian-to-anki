"""Command-line interface for the sync service."""

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from .cli_commands.shared import get_config_and_logger
from .cli_commands.sync_handler import run_sync
from .obsidian.note_validator import validate_note_structure
from .obsidian.parser import parse_frontmatter
from .utils.log_analyzer import LogAnalyzer
from .utils.preflight import run_preflight_checks
from .utils.problematic_notes import ProblematicNotesArchiver

app = typer.Typer(
    name="obsidian-anki-sync",
    help="Obsidian to Anki APF sync service.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def sync(
    dry_run: Annotated[
        bool, typer.Option(
            "--dry-run", help="Preview changes without applying")
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
            help="Number of random notes to process",
        ),
    ] = 10,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Preview changes without applying (default: --dry-run)",
        ),
    ] = True,
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
    index: Annotated[
        bool,
        typer.Option(
            "--index/--no-index",
            help="Build the full vault index before sampling (default: --no-index)",
        ),
    ] = False,
) -> None:
    """Run a sample by processing N random notes."""
    config, logger = get_config_and_logger(config_path, log_level)

    # Override agent system setting if CLI flag is provided
    if use_agents is not None:
        config.use_agent_system = use_agents
        logger.info("agent_system_override", use_agents=use_agents)

    logger.info("test_run_started", sample_count=count, dry_run=dry_run)

    # Run pre-flight checks (skip Anki if dry-run)
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    passed, results = run_preflight_checks(
        config, check_anki=not dry_run, check_llm=True
    )

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
        console.print(
            f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]"
        )
        console.print("[yellow]Fix the errors above and try again.[/yellow]\n")
        raise typer.Exit(code=1)

    console.print("[bold green]All pre-flight checks passed![/bold green]\n")

    from .anki.client import AnkiClient
    from .sync.engine import SyncEngine
    from .sync.state_db import StateDB
    from .utils.progress_display import ProgressDisplay

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            # Create progress display for visual feedback
            progress_display = ProgressDisplay(show_reflections=True)

            engine = SyncEngine(config, db, anki)
            engine.set_progress_display(progress_display)

            stats = engine.sync(
                dry_run=dry_run,
                sample_size=count,
                build_index=index,
            )

            mode_text = "dry-run" if dry_run else "applied"
            console.print(
                f"\n[cyan]Processed sample of {count} notes ({mode_text}).[/cyan]"
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


@app.command(name="lint-note")
def lint_note(
    note_path: Annotated[
        Path, typer.Argument(help="Path to note file (markdown)", exists=True)
    ],
    enforce_bilingual: Annotated[
        bool,
        typer.Option(
            "--enforce-bilingual/--allow-partial",
            help="Require complete Q/A pairs for every language tag.",
        ),
    ] = True,
    check_code_fences: Annotated[
        bool,
        typer.Option(
            "--check-code-fences/--skip-code-fences",
            help="Validate that ``` fences are balanced.",
        ),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
) -> None:
    """Lint a note for bilingual completeness and formatting issues."""
    _, logger = get_config_and_logger(config_path, log_level)
    logger.info("lint_note_started", path=str(note_path))

    # Security: Check file size to prevent DoS attacks
    try:
        file_size = note_path.stat().st_size
        max_file_size = 10 * 1024 * 1024  # 10MB limit
        if file_size > max_file_size:
            console.print(
                f"[bold red]Error:[/bold red] File too large: {note_path} "
                f"({file_size} bytes). Maximum allowed size is {max_file_size} bytes."
            )
            raise typer.Exit(code=1)
    except OSError as exc:
        console.print(f"[bold red]Failed to check file size:[/bold red] {exc}")
        raise typer.Exit(code=1)

    try:
        content = note_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        console.print(f"[bold red]Failed to read note:[/bold red] {exc}")
        raise typer.Exit(code=1)

    metadata = parse_frontmatter(content, note_path)
    errors = validate_note_structure(
        metadata,
        content,
        enforce_languages=enforce_bilingual,
        check_code_fences=check_code_fences,
    )

    if errors:
        console.print(f"[bold red]Found {len(errors)} issue(s):[/bold red]")
        for issue in errors:
            console.print(f"  [red]- {issue}[/red]")
        logger.warning("lint_note_failed", path=str(note_path), errors=errors)
        raise typer.Exit(code=1)

    console.print("[bold green]No lint issues detected.[/bold green]")
    logger.info("lint_note_passed", path=str(note_path))


@app.command()
def validate(
    note_path: Annotated[Path, typer.Argument(help="Path to note file", exists=True)],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        console.print(
            f"  [cyan]Languages:[/cyan] {', '.join(metadata.language_tags)}")
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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

        logger.info("model_fields_completed",
                    model=model_name, count=len(fields))

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
        typer.Option(
            "--sample", help="Export only N random notes (for testing)"),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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

    passed, results = run_preflight_checks(
        config, check_anki=False, check_llm=True)

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
        console.print(
            f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]"
        )
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

            console.print(
                f"[cyan]Processing {len(note_paths)} notes...[/cyan]")

            # Generate cards using the sync engine's generation logic
            from .apf.generator import APFGenerator
            from .models import Card

            cards: list[Card] = []

            if config.use_agent_system:
                console.print(
                    "[cyan]Using multi-agent system for generation...[/cyan]")
                import asyncio
                import inspect

                from .agents.orchestrator import AgentOrchestrator

                orchestrator = AgentOrchestrator(config)

                for note_path_tuple in note_paths:
                    try:
                        note_path, note_content = note_path_tuple
                        metadata, qa_pairs = parse_note(note_path)
                        # Handle both sync and async orchestrators
                        if inspect.iscoroutinefunction(orchestrator.process_note):
                            result = asyncio.run(
                                orchestrator.process_note(
                                    note_content, metadata, qa_pairs, note_path
                                )
                            )
                        else:
                            result = orchestrator.process_note(
                                note_content, metadata, qa_pairs, note_path
                            )

                        if result.success and result.generation:
                            generated_cards = result.generation.cards
                            # Convert GeneratedCard to Card using orchestrator's method
                            converted_cards = orchestrator.convert_to_cards(
                                generated_cards, metadata, qa_pairs
                            )
                            cards.extend(converted_cards)
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
                console.print(
                    "[cyan]Using OpenRouter for generation...[/cyan]")
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
                console.print(
                    "\n[yellow]No cards generated. Exiting.[/yellow]")
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
                f"\n[bold green] Successfully exported {len(cards)} cards "
                f"to {output_path}[/bold green]"
            )

            console.print(
                "\n[cyan]Import this file into Anki to add the cards.[/cyan]")

            logger.info(
                "export_completed",
                output_path=str(output_path),
                card_count=len(cards),
            )

    except Exception as e:
        logger.error("export_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="export-deck")
def export_deck(
    deck_name: Annotated[str, typer.Argument(help="Name of the Anki deck to export")],
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Export an Anki deck to YAML or CSV file."""
    config, logger = get_config_and_logger(config_path, log_level)

    # Auto-generate output filename if not provided
    if output is None:
        safe_deck_name = deck_name.lower().replace(" ", "-").replace("::", "-")
        extension = "csv" if file_format.lower() == "csv" else "yaml"
        output = Path(f"{safe_deck_name}.{extension}")

    logger.info(
        "export_deck_started",
        deck_name=deck_name,
        output_path=str(output),
        format=format,
    )

    from .anki.client import AnkiClient
    from .anki.exporter import export_deck_from_anki

    try:
        # Run pre-flight checks
        console.print(
            "\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")
        passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False)

        for result in results:
            icon = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            console.print(f"{icon} {result.name}: {result.message}")

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print(f"\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Export deck
        client = AnkiClient(config.anki_connect_url)
        export_deck_from_anki(
            client=client,
            deck_name=deck_name,
            output_path=output,
            format=format,
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


@app.command(name="import-deck")
def import_deck(
    input: Annotated[
        Path,
        typer.Argument(help="Path to YAML or CSV file to import"),
    ],
    deck_name: Annotated[
        str,
        typer.Option("--deck", "-d", help="Target Anki deck name"),
    ],
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Import cards from YAML or CSV file into Anki deck."""
    config, logger = get_config_and_logger(config_path, log_level)

    if not input.exists():
        console.print(f"\n[bold red]Error:[/bold red] File not found: {input}")
        raise typer.Exit(code=1)

    # Detect format from extension
    file_format = input.suffix.lower()
    if file_format not in (".yaml", ".yml", ".csv"):
        console.print(
            f"\n[bold red]Error:[/bold red] Unsupported file format: {file_format}"
        )
        console.print("[yellow]Supported formats: .yaml, .yml, .csv[/yellow]")
        raise typer.Exit(code=1)

    logger.info(
        "import_deck_started",
        input_path=str(input),
        deck_name=deck_name,
        file_format=file_format,
    )

    from .anki.client import AnkiClient
    from .anki.importer import import_cards_from_csv, import_cards_from_yaml

    try:
        # Run pre-flight checks
        console.print(
            "\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")
        passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False)

        for result in results:
            icon = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            console.print(f"{icon} {result.name}: {result.message}")

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print(f"\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Import cards
        client = AnkiClient(config.anki_connect_url)
        if file_format == ".csv":
            result = import_cards_from_csv(
                client=client,
                input_path=input,
                deck_name=deck_name,
                note_type=note_type,
                key_field=key_field,
            )
        else:
            result = import_cards_from_yaml(
                client=client,
                input_path=input,
                deck_name=deck_name,
                note_type=note_type,
                key_field=key_field,
            )

        console.print(f"\n[bold green]Import complete![/bold green]")
        console.print(f"  Created: {result['created']}")
        console.print(f"  Updated: {result['updated']}")
        if result["errors"] > 0:
            console.print(f"  [yellow]Errors: {result['errors']}[/yellow]")

        logger.info("import_deck_completed", **result)

    except Exception as e:
        logger.error("import_deck_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="process-file")
def process_file(
    input: Annotated[
        Path,
        typer.Argument(help="Path to input YAML or CSV file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path (required)"),
    ],
    prompt: Annotated[
        Path,
        typer.Option("--prompt", "-p", help="Path to prompt template file"),
    ],
    field: Annotated[
        str | None,
        typer.Option(
            "--field", help="Field name to update (single field mode)"),
    ] = None,
    json_mode: Annotated[
        bool,
        typer.Option(
            "--json", help="Expect JSON response and merge all fields"),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model to use"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b",
                     help="Number of concurrent API requests"),
    ] = 5,
    retries: Annotated[
        int,
        typer.Option("--retries", "-r",
                     help="Number of retries for failed requests"),
    ] = 3,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview without making API calls"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Re-process all rows, ignoring existing output"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Limit number of new rows to process"),
    ] = None,
    require_result_tag: Annotated[
        bool,
        typer.Option(
            "--require-result-tag",
            help="Extract content from <result></result> tags only",
        ),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Process cards from file with LLM and save results."""
    config, logger = get_config_and_logger(config_path, log_level)

    if not input.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Input file not found: {input}")
        raise typer.Exit(code=1)

    if field and json_mode:
        console.print(
            "\n[bold red]Error:[/bold red] --field and --json are mutually exclusive"
        )
        raise typer.Exit(code=1)

    if not field and not json_mode:
        console.print(
            "\n[bold red]Error:[/bold red] Must specify either --field or --json"
        )
        raise typer.Exit(code=1)

    # Load prompt template
    if not prompt.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Prompt file not found: {prompt}")
        raise typer.Exit(code=1)

    with prompt.open("r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Determine model
    model_name = model or config.openrouter_model or "gpt-4o-mini"
    if not model_name:
        console.print(
            "\n[bold red]Error:[/bold red] No model specified and no default configured"
        )
        raise typer.Exit(code=1)

    logger.info(
        "process_file_started",
        input_path=str(input),
        output_path=str(output),
        model=model_name,
        field=field,
        json_mode=json_mode,
    )

    from .cli_commands.process_file import (
        get_processed_slugs,
        load_cards_from_file,
        process_card_with_llm,
        save_cards_to_file,
    )

    try:
        # Load input cards
        console.print(f"\n[cyan]Loading cards from {input}...[/cyan]")
        all_cards = load_cards_from_file(input)
        console.print(f"[green]Found {len(all_cards)} cards[/green]")

        # Determine format
        input_file_format = input.suffix.lower()
        if input_file_format == ".csv":
            output_file_format = "csv"
        else:
            output_file_format = "yaml"

        # Get already processed cards
        processed_slugs = set()
        processed_cards = []
        if output.exists() and not force:
            processed_slugs = get_processed_slugs(output, output_format)
            processed_cards = load_cards_from_file(output)
            console.print(
                f"[cyan]Found {len(processed_cards)} already processed cards[/cyan]"
            )

        # Filter cards to process
        cards_to_process = []
        for card in all_cards:
            slug = card.get("slug", "")
            if force or slug not in processed_slugs:
                cards_to_process.append(card)

        if limit:
            cards_to_process = cards_to_process[:limit]

        if not cards_to_process:
            console.print("\n[yellow]No new cards to process.[/yellow]")
            return

        console.print(
            f"\n[cyan]Processing {len(cards_to_process)} cards...[/cyan]")

        if dry_run:
            console.print(
                "[yellow]Dry run mode: No API calls will be made[/yellow]")
            for card in cards_to_process[:5]:  # Show first 5 as examples
                console.print(
                    f"  Would process: {card.get('slug', 'unknown')}")
            return

        # Initialize LLM client
        from .providers.factory import ProviderFactory

        provider = ProviderFactory.create_from_config(config)
        # Get the underlying client (OpenAI, Anthropic, etc.)
        if hasattr(provider, "client"):
            llm_client = provider.client
        elif hasattr(provider, "get_client"):
            llm_client = provider.get_client()
        else:
            # Fallback: try to use provider directly
            llm_client = provider

        # Process cards
        updated_cards = []
        success_count = 0
        error_count = 0

        for i, card in enumerate(cards_to_process, 1):
            try:
                console.print(
                    f"  [{i}/{len(cards_to_process)}] Processing {card.get('slug', 'unknown')}..."
                )
                updated_card = process_card_with_llm(
                    card=card.copy(),
                    prompt_template=prompt_template,
                    llm_client=llm_client,
                    model=model_name,
                    field_name=field,
                    json_mode=json_mode,
                    require_result_tag=require_result_tag,
                )
                updated_cards.append(updated_card)

                if "_error" not in updated_card:
                    success_count += 1
                else:
                    error_count += 1

                # Save incrementally every 10 cards
                if i % 10 == 0:
                    all_processed = processed_cards + updated_cards
                    save_cards_to_file(all_processed, output, output_format)
                    console.print(
                        f"  [dim]Saved progress ({i}/{len(cards_to_process)})[/dim]"
                    )

            except Exception as e:
                logger.error(
                    "card_processing_error", error=str(e), slug=card.get("slug")
                )
                error_count += 1
                card["_error"] = str(e)
                updated_cards.append(card)

        # Final save
        all_processed = processed_cards + updated_cards
        save_cards_to_file(all_processed, output, output_file_format)

        console.print(f"\n[bold green]Processing complete![/bold green]")
        console.print(f"  Success: {success_count}")
        console.print(f"  Errors: {error_count}")
        console.print(f"  Output: {output}")

        logger.info(
            "process_file_completed",
            total=len(cards_to_process),
            success=success_count,
            errors=error_count,
        )

    except Exception as e:
        logger.error("process_file_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="query")
def query_anki(
    action: Annotated[
        str,
        typer.Argument(
            help="AnkiConnect API action name (e.g., 'deckNames', 'findNotes')"
        ),
    ],
    params: Annotated[
        str | None,
        typer.Argument(help="JSON string of parameters (optional)"),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Query AnkiConnect API directly."""
    config, logger = get_config_and_logger(config_path, log_level)

    from .anki.client import AnkiClient

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
            import json

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
        passed, results = run_preflight_checks(
            config, check_anki=True, check_llm=False)

        errors = [r for r in results if not r.passed and r.severity == "error"]
        if errors:
            console.print(f"\n[bold red]Pre-flight checks failed.[/bold red]")
            raise typer.Exit(code=1)

        # Execute query
        client = AnkiClient(config.anki_connect_url)
        result = client.invoke(action, params_dict if params_dict else None)

        # Output result as JSON
        import json

        console.print("\n[bold green]Result:[/bold green]")
        console.print(json.dumps(result, indent=2, ensure_ascii=False))

        logger.info("query_completed", action=action,
                    has_result=result is not None)

    except Exception as e:
        logger.error("query_failed", error=str(e), action=action)
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="generate")
def generate_cards(
    term: Annotated[
        str,
        typer.Argument(help="Term or phrase to generate cards for"),
    ],
    prompt: Annotated[
        Path,
        typer.Option("--prompt", "-p", help="Path to prompt template file"),
    ],
    count: Annotated[
        int,
        typer.Option("--count", "-c",
                     help="Number of card examples to generate"),
    ] = 3,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model to use"),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option("--temperature", "-t", help="LLM temperature (0.0-2.0)"),
    ] = 1.0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Display cards without importing"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Export cards to file instead of importing"
        ),
    ] = None,
    copy_mode: Annotated[
        bool,
        typer.Option(
            "--copy", help="Copy prompt to clipboard for manual LLM interaction"
        ),
    ] = False,
    log: Annotated[
        Path | None,
        typer.Option(
            "--log", help="Generate log file with detailed debug information"),
    ] = None,
    very_verbose: Annotated[
        bool,
        typer.Option(
            "--very-verbose",
            help="Log full LLM responses to log file (automatically enables --log)",
        ),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Generate multiple card examples for a term and interactively select which to keep."""
    # Enable logging if requested
    log_file = log
    if very_verbose and not log_file:
        # Auto-generate log file name if very-verbose is enabled
        safe_term = term.lower().replace(" ", "-")[:20]
        log_file = Path(f"generate_{safe_term}.log")

    config, logger = get_config_and_logger(
        config_path, log_level, log_file=log_file, very_verbose=very_verbose
    )

    if not prompt.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Prompt file not found: {prompt}")
        raise typer.Exit(code=1)

    # Parse prompt template
    from .prompts.template_parser import parse_template_file

    try:
        template = parse_template_file(prompt)
    except Exception as e:
        console.print(
            f"\n[bold red]Error:[/bold red] Failed to parse template: {e}")
        raise typer.Exit(code=1)

    # Determine model
    model_name = model or config.openrouter_model or "gpt-4o-mini"
    if not model_name:
        console.print(
            "\n[bold red]Error:[/bold red] No model specified and no default configured"
        )
        raise typer.Exit(code=1)

    logger.info(
        "generate_cards_started",
        term=term,
        count=count,
        model=model_name,
        template_path=str(prompt),
    )

    # Build prompt with substitutions
    prompt_text = template.substitute(term=term, count=count)

    # Copy mode: copy to clipboard and wait for manual input
    if copy_mode:
        from .utils.clipboard import copy_to_clipboard

        console.print("\n[cyan]Copy mode: Prompt copied to clipboard[/cyan]")
        if copy_to_clipboard(prompt_text):
            console.print("[green]Prompt copied![/green]")
            console.print("\n[yellow]Instructions:[/yellow]")
            console.print(
                "1. Paste the prompt into your LLM interface (ChatGPT, Claude, etc.)"
            )
            console.print("2. Copy the complete JSON response")
            console.print("3. Paste it here and press Enter")
            console.print("4. Type 'END' on a new line to finish\n")

            # Wait for user input
            lines = []
            console.print(
                "[cyan]Paste JSON response (type 'END' on new line to finish):[/cyan]"
            )
            while True:
                try:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]Input cancelled[/yellow]")
                    return

            response_text = "\n".join(lines)
        else:
            console.print(
                "[yellow]Clipboard not available, displaying prompt:[/yellow]"
            )
            console.print(f"\n[dim]{prompt_text}[/dim]\n")
            response_text = Prompt.ask(
                "[cyan]Paste LLM response[/cyan]", default="")
    else:
        # Normal mode: call LLM
        from .providers.factory import ProviderFactory

        provider = ProviderFactory.create_from_config(config)
        if hasattr(provider, "client"):
            llm_client = provider.client
        else:
            llm_client = provider

        console.print(
            f"\n[cyan]Generating {count} card candidates for '{term}'...[/cyan]"
        )

        try:
            if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
                # OpenAI-style client
                response = llm_client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                response_text = response.choices[0].message.content or ""
            else:
                # Generic interface
                response_text = llm_client.generate(
                    prompt_text, model=model_name, temperature=temperature
                )

        except Exception as e:
            logger.error("card_generation_failed", error=str(e))
            console.print(
                f"\n[bold red]Error generating cards:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Parse JSON response
    import json

    try:
        cards_data = json.loads(response_text)
        if not isinstance(cards_data, list):
            cards_data = [cards_data]
    except json.JSONDecodeError as e:
        console.print(
            f"\n[bold red]Error:[/bold red] Invalid JSON response: {e}")
        console.print(
            f"[yellow]Response was:[/yellow]\n{response_text[:200]}...")
        raise typer.Exit(code=1)

    # Convert to CardCandidate objects
    from .utils.card_selector import CardCandidate

    candidates = []
    for i, card_data in enumerate(cards_data):
        if not isinstance(card_data, dict):
            continue

        # Map fields using template field_map if available
        fields = {}
        if template.field_map:
            for template_key, anki_field in template.field_map.items():
                if template_key in card_data:
                    fields[anki_field] = card_data[template_key]
        else:
            fields = card_data

        candidate = CardCandidate(index=i, fields=fields)
        candidates.append(candidate)

    if not candidates:
        console.print("\n[yellow]No valid cards generated.[/yellow]")
        return

    # Check for duplicates if we have a deck
    if template.deck and not dry_run:
        from .anki.client import AnkiClient

        try:
            client = AnkiClient(config.anki_connect_url)
            note_ids = client.find_notes(f'deck:"{template.deck}"')
            if note_ids:
                notes_info = client.notes_info(note_ids)
                # Simple duplicate check based on field content
                for candidate in candidates:
                    for note_info in notes_info:
                        note_fields = note_info.get("fields", {})
                        # Check if any field matches
                        for field_name, field_value in candidate.fields.items():
                            if field_name in note_fields:
                                if field_value[:50] in note_fields[field_name][:50]:
                                    candidate.is_duplicate = True
                                    candidate.duplicate_reason = (
                                        f"Similar to existing card in {template.deck}"
                                    )
                                    break
        except Exception as e:
            logger.warning("duplicate_check_failed", error=str(e))

    # Interactive selection
    if not dry_run and not output:
        from .utils.card_selector import select_cards_interactive

        selected_indices = select_cards_interactive(
            candidates, title=f"Select cards to add for '{term}'"
        )

        if not selected_indices:
            console.print("\n[yellow]No cards selected. Exiting.[/yellow]")
            return

        selected_candidates = [candidates[i] for i in selected_indices]

        # Quality check if configured
        if template.quality_check:
            console.print("\n[cyan]Running quality checks...[/cyan]")
            from .providers.factory import ProviderFactory
            from .utils.quality_check import run_quality_check

            quality_provider = ProviderFactory.create_from_config(config)
            if hasattr(quality_provider, "client"):
                quality_client = quality_provider.client
            else:
                quality_client = quality_provider

            for candidate in selected_candidates:
                quality_result = run_quality_check(
                    card_fields=candidate.fields,
                    quality_config=template.quality_check,
                    llm_client=quality_client,
                )
                candidate.quality_score = quality_result["score"]
                candidate.quality_reason = quality_result["reason"]

                if not quality_result["is_valid"] or quality_result["score"] < 0.7:
                    console.print(
                        f"  [yellow]Card {candidate.index + 1} flagged:[/yellow] {quality_result['reason']}"
                    )

        # Import selected cards
        if template.deck:
            import tempfile

            # Convert to YAML format for import
            import yaml

            from .anki.client import AnkiClient
            from .anki.importer import import_cards_from_yaml

            cards_for_import = []
            for candidate in selected_candidates:
                card_data = {
                    "noteType": template.note_type or "APF::Simple",
                    **candidate.fields,
                }
                cards_for_import.append(card_data)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(cards_for_import, f)
                temp_path = Path(f.name)

            try:
                client = AnkiClient(config.anki_connect_url)
                result = import_cards_from_yaml(
                    client=client,
                    input_path=temp_path,
                    deck_name=template.deck,
                    note_type=template.note_type,
                )
                console.print(
                    f"\n[bold green]Successfully added {result['created']} card(s)![/bold green]"
                )
            finally:
                temp_path.unlink()

    elif output:
        # Export to file
        import yaml

        cards_for_export = []
        for candidate in candidates:
            card_data = {
                "noteType": template.note_type or "APF::Simple",
                **candidate.fields,
            }
            cards_for_export.append(card_data)

        with output.open("w", encoding="utf-8") as f:
            yaml.dump(cards_for_export, f,
                      default_flow_style=False, allow_unicode=True)

        console.print(
            f"\n[bold green]Exported {len(cards_for_export)} cards to {output}[/bold green]"
        )
    else:
        # Dry run: just display
        console.print(
            f"\n[cyan]Generated {len(candidates)} card candidates:[/cyan]\n")
        for candidate in candidates:
            console.print(f"[bold]Card {candidate.index + 1}:[/bold]")
            for key, value in candidate.fields.items():
                console.print(
                    f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}"
                )
            console.print()

    logger.info("generate_cards_completed", term=term, count=len(candidates))


@app.command(name="index")
def show_index(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Show vault and Anki card index statistics."""
    _, logger = get_config_and_logger(config_path, log_level)

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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
    skip_anki: Annotated[
        bool,
        typer.Option("--skip-anki", help="Skip Anki connectivity check"),
    ] = False,
    skip_llm: Annotated[
        bool,
        typer.Option(
            "--skip-llm", help="Skip LLM provider connectivity check"),
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
        elif result.severity == "warning":
            icon = "[yellow]⚠[/yellow]"
        else:
            icon = "[red]✗[/red]"

        console.print(f"{icon} [bold]{result.name}[/bold]: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]💡 {result.fix_suggestion}[/dim]")

    console.print()

    # Count errors and warnings
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    passed_checks = [r for r in results if r.passed]

    # Display summary table
    summary_table = Table(
        title="Check Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Status", style="cyan")
    summary_table.add_column("Count", style="green")

    summary_table.add_row("✓ Passed", str(len(passed_checks)))
    summary_table.add_row("⚠ Warnings", str(len(warnings)))
    summary_table.add_row("✗ Errors", str(len(errors)))

    console.print(summary_table)
    console.print()

    if errors:
        console.print("[bold red]❌ Setup validation failed![/bold red]")
        console.print(
            "[yellow]Fix the errors above before running sync.[/yellow]\n")
        logger.error("check_setup_failed", errors=len(
            errors), warnings=len(warnings))
        raise typer.Exit(code=1)
    elif warnings:
        console.print(
            "[bold yellow]⚠️  Setup validation passed with warnings.[/bold yellow]"
        )
        console.print(
            "[dim]You may proceed, but some features may not work as expected.[/dim]\n"
        )
        logger.info("check_setup_passed_with_warnings", warnings=len(warnings))
    else:
        console.print(
            "[bold green]✅ All checks passed! Your setup is ready.[/bold green]\n"
        )
        logger.info("check_setup_passed")


@app.command(name="analyze-logs")
def analyze_logs(
    days: Annotated[
        int,
        typer.Option("--days", "-d",
                     help="Number of days to analyze (default: 7)"),
    ] = 7,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Analyze log files and show error summary."""
    config, logger = get_config_and_logger(config_path, log_level)
    logger.info("analyze_logs_started", days=days)

    analyzer = LogAnalyzer(log_dir=config.project_log_dir)
    summary = analyzer.generate_summary_report(days=days)
    error_analysis = analyzer.analyze_errors(days=days)

    console.print(
        f"\n[bold cyan]Log Analysis Summary (Last {days} days)[/bold cyan]\n")

    # Overall statistics
    table = Table(title="Overall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Log Entries", str(summary["total_log_entries"]))
    table.add_row("Total Errors", str(error_analysis["total_errors"]))

    for level, count in summary["levels"].items():
        table.add_row(f"Level: {level}", str(count))

    console.print(table)
    console.print()

    # Error breakdown
    if error_analysis["total_errors"] > 0:
        error_table = Table(title="Error Breakdown")
        error_table.add_column("Category", style="cyan")
        error_table.add_column("Count", style="red")

        error_table.add_row("Total Errors", str(
            error_analysis["total_errors"]))

        if error_analysis["errors_by_type"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]By Error Type[/bold]", "")
            for error_type, count in list(error_analysis["errors_by_type"].items())[
                :10
            ]:
                error_table.add_row(f"  {error_type}", str(count))

        if error_analysis["errors_by_stage"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]By Processing Stage[/bold]", "")
            for stage, count in error_analysis["errors_by_stage"].items():
                error_table.add_row(f"  {stage}", str(count))

        if error_analysis["errors_by_file"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]Most Problematic Files[/bold]", "")
            for file_path, count in list(error_analysis["errors_by_file"].items())[:5]:
                error_table.add_row(f"  {file_path}", str(count))

        console.print(error_table)
        console.print()

        # Recent errors
        if error_analysis["recent_errors"]:
            recent_table = Table(title="Recent Errors (Last 10)")
            recent_table.add_column("Timestamp", style="dim")
            recent_table.add_column("Level", style="red")
            recent_table.add_column("Error Type", style="yellow")
            recent_table.add_column("File", style="cyan")
            recent_table.add_column("Message", style="white", max_width=50)

            for error in error_analysis["recent_errors"][:10]:
                recent_table.add_row(
                    error.get("timestamp", "")[:19],
                    error.get("level", ""),
                    error.get("error_type", "Unknown")[:30],
                    (error.get("file") or "")[:40],
                    error.get("message", "")[:50],
                )

            console.print(recent_table)
    else:
        console.print(
            "[green]No errors found in the specified period![/green]\n")

    logger.info("analyze_logs_completed")


@app.command(name="list-problematic-notes")
def list_problematic_notes(
    error_type: Annotated[
        str | None,
        typer.Option("--error-type", help="Filter by error type"),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            help="Filter by category (parser_errors, validation_errors, llm_errors, generation_errors, other_errors)",
        ),
    ] = None,
    date: Annotated[
        str | None,
        typer.Option("--date", help="Filter by date (YYYY-MM-DD)"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-n", help="Limit number of results"),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """List archived problematic notes."""
    config, logger = get_config_and_logger(config_path, log_level)
    logger.info("list_problematic_notes_started")

    archiver = ProblematicNotesArchiver(
        archive_dir=config.problematic_notes_dir,
        enabled=True,
    )

    notes = archiver.get_archived_notes(
        error_type=error_type,
        category=category,
        date=date,
        limit=limit,
    )

    if not notes:
        console.print(
            "[yellow]No problematic notes found matching the criteria.[/yellow]\n"
        )
        return

    console.print(
        f"\n[bold cyan]Problematic Notes ({len(notes)} found)[/bold cyan]\n")

    table = Table()
    table.add_column("Original Path", style="cyan", max_width=50)
    table.add_column("Error Type", style="red")
    table.add_column("Category", style="yellow")
    table.add_column("Stage", style="dim")
    table.add_column("Timestamp", style="dim")

    for note in notes:
        table.add_row(
            note.get("original_path", "")[:50],
            note.get("error_type", "Unknown"),
            note.get("category", "unknown"),
            note.get("processing_stage", "unknown"),
            note.get("timestamp", "")[:19] if note.get("timestamp") else "",
        )

    console.print(table)
    console.print()

    console.print(
        f"[dim]Archived notes are stored in: {config.problematic_notes_dir}[/dim]\n"
    )

    logger.info("list_problematic_notes_completed", count=len(notes))


@app.command()
def format(
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Run formatters in check mode (no modifications)"),
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
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
        console.print(
            f"[bold red]Error:[/bold red] Formatter failed: {exc.cmd}")
        raise typer.Exit(code=1)


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
