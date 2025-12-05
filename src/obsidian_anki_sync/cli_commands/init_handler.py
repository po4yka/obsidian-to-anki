"""Init command implementation logic."""

from pathlib import Path
from typing import Any

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.state_db import StateDB

from .shared import console


def run_init(config: Config, logger: Any) -> None:
    """Execute the init operation.

    Args:
        config: Configuration object
        logger: Logger instance

    Raises:
        typer.Exit: On init failure
    """
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
OPENROUTER_MODEL=qwen/qwen-2.5-32b-instruct
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
    try:
        with StateDB(config.db_path):
            pass  # Database schema is initialized in __init__
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
        console.print(f"[bold red]Failed to initialize database:[/bold red] {e}")
        raise

    console.print(f"[green] Initialized database at {config.db_path}[/green]")
    logger.info("database_initialized", path=str(config.db_path))

    logger.info("init_completed")
