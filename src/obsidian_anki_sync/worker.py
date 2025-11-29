"""Worker for processing card generation jobs via Redis queue."""

import os
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings

from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.obsidian.parser import parse_note
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize worker context."""
    logger.info("worker_startup")

    try:
        config = load_config()
        ctx["config"] = config

        # Initialize orchestrator
        orchestrator = LangGraphOrchestrator(config)
        if hasattr(orchestrator, "setup_async"):
            await orchestrator.setup_async()

        ctx["orchestrator"] = orchestrator
        logger.info("worker_orchestrator_initialized")
    except Exception as e:
        logger.warning("worker_startup_config_failed", error=str(e))
        # Worker can still run without full config - jobs will handle config individually
        ctx["config"] = None
        ctx["orchestrator"] = None


async def shutdown(ctx: dict[str, Any]) -> None:
    """Cleanup worker context."""
    logger.info("worker_shutdown")


async def process_note_job(
    ctx: dict[str, Any],
    file_path: str,
    relative_path: str,
    metadata_dict: dict[str, Any] | None = None,
    qa_pairs_dicts: list[dict[str, Any]] | None = None,
    existing_cards_dicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Process a note generation job.

    Args:
        ctx: Worker context
        file_path: Absolute path to note file
        relative_path: Relative path for logging
        metadata_dict: Optional pre-parsed metadata
        qa_pairs_dicts: Optional pre-parsed QA pairs
        existing_cards_dicts: Optional existing cards for duplicate detection

    Returns:
        Dict containing generated cards and status
    """
    # Get or create config and orchestrator
    config = ctx.get("config")
    orchestrator = ctx.get("orchestrator")

    if config is None or orchestrator is None:
        # Initialize config and orchestrator for this job
        config = load_config()
        orchestrator = LangGraphOrchestrator(config)
        if hasattr(orchestrator, "setup_async"):
            await orchestrator.setup_async()

    logger.info("worker_processing_job", file=relative_path)

    try:
        path_obj = Path(file_path)
        if not path_obj.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "cards": [],
            }

        # Read content
        note_content = path_obj.read_text(encoding="utf-8")

        # Parse if not provided
        if metadata_dict and qa_pairs_dicts:
            metadata = NoteMetadata(**metadata_dict)
            qa_pairs = [QAPair(**qa) for qa in qa_pairs_dicts]
        else:
            metadata, qa_pairs = parse_note(path_obj, content=note_content)

        # Reconstruct existing cards if provided
        existing_cards = None
        if existing_cards_dicts:
            from obsidian_anki_sync.agents.models import GeneratedCard

            existing_cards = [GeneratedCard(**c) for c in existing_cards_dicts]

        # Process note
        result = await orchestrator.process_note(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            file_path=path_obj,
            existing_cards=existing_cards,
        )

        if not result.success or not result.generation:
            return {
                "success": False,
                "error": "Pipeline failed to generate cards",
                "cards": [],
            }

        # Convert to cards
        cards = orchestrator.convert_to_cards(
            result.generation.cards, metadata, qa_pairs, path_obj
        )

        # Serialize cards for return
        # We return dicts because Card objects might not be pickleable or we want to be safe
        cards_dicts = [card.model_dump() for card in cards]

        return {"success": True, "cards": cards_dicts, "slugs": [c.slug for c in cards]}

    except Exception as e:
        logger.exception("worker_job_failed", file=relative_path, error=str(e))
        return {"success": False, "error": str(e), "cards": []}


class WorkerSettings:
    """Arq worker settings."""

    functions = [process_note_job]
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = 600  # 10 minutes

    # Use environment variables with defaults
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    max_jobs = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "5"))
