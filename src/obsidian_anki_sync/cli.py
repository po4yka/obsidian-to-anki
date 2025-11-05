"""Command-line interface for the sync service."""

import subprocess
from pathlib import Path
from typing import Annotated, Optional

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

# Global state for config and logger
_config: Optional[Config] = None
_logger = None


def get_config_and_logger(
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> tuple[Config, object]:
    """Load configuration and logger (dependency injection helper)."""
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
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Synchronize Obsidian notes to Anki cards."""
    config, logger = get_config_and_logger(config_path, log_level)

    logger.info("sync_started", dry_run=dry_run, vault=str(config.vault_path))

    from .anki.client import AnkiClient
    from .sync.engine import SyncEngine
    from .sync.state_db import StateDB

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            engine = SyncEngine(config, db, anki)
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
    config_path: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Run a sample dry-run by processing N random notes."""
    config, logger = get_config_and_logger(config_path, log_level)

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
        Optional[Path],
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
        Optional[Path],
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
        Optional[Path],
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
        Optional[Path],
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
        Optional[Path],
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
def format(
    check: Annotated[
        bool,
        typer.Option("--check", help="Run formatters in check mode (no modifications)"),
    ] = False,
    config_path: Annotated[
        Optional[Path],
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
