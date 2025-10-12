"""Command-line interface for the sync service."""

import subprocess
from pathlib import Path
from typing import Optional

import click

from .config import get_config, load_config, set_config
from .utils.logging import configure_logging, get_logger


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config.yaml')
@click.option('--log-level', default='INFO', help='Log level (DEBUG, INFO, WARN, ERROR)')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], log_level: str) -> None:
    """Obsidian to Anki APF sync service."""
    # Load configuration
    config_path = Path(config) if config else None
    cfg = load_config(config_path)
    set_config(cfg)

    # Configure logging
    configure_logging(log_level or cfg.log_level)

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = cfg
    ctx.obj['logger'] = get_logger('cli')


@cli.command()
@click.option('--dry-run', is_flag=True, help='Preview changes without applying')
@click.pass_context
def sync(ctx: click.Context, dry_run: bool) -> None:
    """Synchronize Obsidian notes to Anki cards."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("sync_started", dry_run=dry_run, vault=str(config.vault_path))

    from .anki.client import AnkiClient
    from .sync.state_db import StateDB
    from .sync.engine import SyncEngine

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            engine = SyncEngine(config, db, anki)
            stats = engine.sync(dry_run=dry_run)

            click.echo("\n=== Sync Results ===")
            for key, value in stats.items():
                click.echo(f"{key}: {value}")

            logger.info("sync_completed", stats=stats)

    except Exception as e:
        logger.error("sync_failed", error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@cli.command(name="test-run")
@click.option(
    '--count',
    default=10,
    show_default=True,
    type=click.IntRange(1, None, clamp=True),
    help='Number of random notes to process (dry-run)'
)
@click.pass_context
def test_run(ctx: click.Context, count: int) -> None:
    """Run a sample dry-run by processing N random notes."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("test_run_started", sample_count=count)

    from .anki.client import AnkiClient
    from .sync.state_db import StateDB
    from .sync.engine import SyncEngine

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            engine = SyncEngine(config, db, anki)
            stats = engine.sync(dry_run=True, sample_size=count)

            click.echo(f"\nProcessed sample of {count} notes (dry-run).")
            click.echo("=== Sample Results ===")
            for key, value in stats.items():
                click.echo(f"{key}: {value}")

            logger.info("test_run_completed", stats=stats)

    except Exception as e:
        logger.error("test_run_failed", error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('note_path', type=click.Path(exists=True))
@click.pass_context
def validate(ctx: click.Context, note_path: str) -> None:
    """Validate note structure and APF compliance."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("validate_started", note_path=note_path)

    from .obsidian.parser import parse_note
    from .obsidian.validator import validate_note

    try:
        note_path_obj = Path(note_path)
        metadata, qa_pairs = parse_note(note_path_obj)

        click.echo(f"\nParsed: {note_path}")
        click.echo(f"  Title: {metadata.title}")
        click.echo(f"  Topic: {metadata.topic}")
        click.echo(f"  Languages: {', '.join(metadata.language_tags)}")
        click.echo(f"  Q/A pairs: {len(qa_pairs)}")

        # Validate
        errors = validate_note(metadata, qa_pairs, note_path_obj)

        if errors:
            click.echo("\nValidation errors:")
            for error in errors:
                click.echo(f"  - {error}")
            logger.error("validation_failed", errors=errors)
        else:
            click.echo("\nValidation passed!")
            logger.info("validation_passed")

    except Exception as e:
        logger.error("validation_error", error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()

    logger.info("validate_completed")


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize configuration and database."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("init_started")

    # Create .env template if it doesn't exist
    env_path = Path('.env')
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
        click.echo(f"Created .env template at {env_path}")
        logger.info("env_template_created", path=str(env_path))
    else:
        click.echo(f".env already exists at {env_path}")

    # Initialize database
    from .sync.state_db import StateDB

    db = StateDB(config.db_path)
    db.close()

    click.echo(f"Initialized database at {config.db_path}")
    logger.info("database_initialized", path=str(config.db_path))

    logger.info("init_completed")


@cli.command(name="decks")
@click.pass_context
def list_decks(ctx: click.Context) -> None:
    """List deck names available via AnkiConnect."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("list_decks_started")

    from .anki.client import AnkiClient

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            decks = sorted(anki.get_deck_names())

        if not decks:
            click.echo("No decks available.")
        else:
            click.echo("\nDecks:")
            for deck in decks:
                click.echo(f"  - {deck}")

        logger.info("list_decks_completed", count=len(decks))

    except Exception as e:
        logger.error("list_decks_failed", error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@cli.command(name="models")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """List note models (types) available in Anki."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("list_models_started")

    from .anki.client import AnkiClient

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            models = sorted(anki.get_model_names())

        if not models:
            click.echo("No models available.")
        else:
            click.echo("\nModels:")
            for model in models:
                click.echo(f"  - {model}")

        logger.info("list_models_completed", count=len(models))

    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@cli.command(name="model-fields")
@click.option('--model', 'model_name', required=True, help='Model name to inspect')
@click.pass_context
def show_model_fields(ctx: click.Context, model_name: str) -> None:
    """Show field names for a specific Anki model."""
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info("model_fields_started", model=model_name)

    from .anki.client import AnkiClient

    try:
        with AnkiClient(config.anki_connect_url) as anki:
            fields = anki.get_model_field_names(model_name)

        if not fields:
            click.echo(f"Model '{model_name}' has no fields or does not exist.")
        else:
            click.echo(f"\nFields for model '{model_name}':")
            for field in fields:
                click.echo(f"  - {field}")

        logger.info("model_fields_completed", model=model_name, count=len(fields))

    except Exception as e:
        logger.error("model_fields_failed", model=model_name, error=str(e))
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--check/--fix', default=False, help='Run formatters in check mode (no modifications).')
@click.pass_context
def format(ctx: click.Context, check: bool) -> None:
    """Run code formatters (ruff + black)."""
    logger = ctx.obj['logger']
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
            subprocess.run(cmd, check=True)

        logger.info("format_completed", check=check)
    except subprocess.CalledProcessError as exc:
        logger.error("format_failed", returncode=exc.returncode, cmd=exc.cmd)
        raise click.ClickException(f"Formatter failed: {exc.cmd}")


if __name__ == '__main__':
    cli()
