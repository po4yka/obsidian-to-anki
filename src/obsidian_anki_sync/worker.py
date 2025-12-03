"""Worker for processing card generation jobs via Redis queue."""

from __future__ import annotations

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
    result_queue_name: str | None = None,
) -> dict[str, Any]:
    """Process a note generation job.

    Args:
        ctx: Worker context
        file_path: Absolute path to note file
        relative_path: Relative path for logging
        metadata_dict: Optional pre-parsed metadata
        qa_pairs_dicts: Optional pre-parsed QA pairs
        existing_cards_dicts: Optional existing cards for duplicate detection
        result_queue_name: Optional Redis list to push results to

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

    generation_sla = getattr(config, "worker_generation_timeout_seconds", 900.0)
    validation_sla = getattr(config, "worker_validation_timeout_seconds", 900.0)

    try:
        path_obj = Path(file_path)
        if not path_obj.exists():
            result = {
                "success": False,
                "error": f"File not found: {file_path}",
                "cards": [],
            }
            if result_queue_name:
                await _push_result(ctx, result_queue_name, result)
            return result

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
        pipeline_result = await orchestrator.process_note(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            file_path=path_obj,
            existing_cards=existing_cards,
        )

        pipeline_context = {
            "total_time": round(pipeline_result.total_time, 3),
            "retry_count": pipeline_result.retry_count,
            "stage_times": pipeline_result.stage_times,
            "last_error": pipeline_result.last_error,
            "generation_sla": generation_sla,
            "validation_sla": validation_sla,
        }

        sla_violation = _detect_stage_sla(
            pipeline_result.stage_times, generation_sla, validation_sla
        )
        if sla_violation is not None:
            logger.error(
                "pipeline_stage_timeout",
                file=relative_path,
                stage=sla_violation["stage"],
                elapsed=round(sla_violation["elapsed"], 2),
                budget=sla_violation["budget"],
            )
            result = {
                "success": False,
                "error": sla_violation["message"],
                "cards": [],
            }
            if result_queue_name:
                await _push_result(ctx, result_queue_name, result)
            return result

        if not pipeline_result.success or not pipeline_result.generation:
            # Build detailed error message from pipeline result
            error_details = []
            if not pipeline_result.success:
                error_details.append("Pipeline marked as failed")
            if pipeline_result.pre_validation and not pipeline_result.pre_validation.is_valid:
                error_details.append(
                    f"Pre-validation failed ({pipeline_result.pre_validation.error_type}): "
                    f"{pipeline_result.pre_validation.error_details}"
                )
            if pipeline_result.post_validation and not pipeline_result.post_validation.is_valid:
                error_details.append(
                    f"Post-validation failed ({pipeline_result.post_validation.error_type}): "
                    f"{pipeline_result.post_validation.error_details}"
                )
            if not pipeline_result.generation:
                error_details.append("No cards generated")
            elif pipeline_result.generation.total_cards == 0:
                error_details.append("Generator returned zero cards")

            if pipeline_result.post_validation:
                logger.warning(
                    "post_validation_payload",
                    file=relative_path,
                    is_valid=pipeline_result.post_validation.is_valid,
                    error_type=pipeline_result.post_validation.error_type,
                    error_details=pipeline_result.post_validation.error_details[:500],
                )

            error_msg = (
                "; ".join(error_details) if error_details else "Unknown pipeline error"
            )
            logger.warning(
                "pipeline_generation_failed",
                file=relative_path,
                error=error_msg,
                pre_valid=pipeline_result.pre_validation.is_valid
                if pipeline_result.pre_validation
                else None,
                post_valid=pipeline_result.post_validation.is_valid
                if pipeline_result.post_validation
                else None,
                stage_times=pipeline_result.stage_times,
                retry_count=pipeline_result.retry_count,
                last_error=pipeline_result.last_error,
            )
            result = {
                "success": False,
                "error": error_msg,
                "cards": [],
            }
            if result_queue_name:
                await _push_result(ctx, result_queue_name, result)
            return result

        # Convert to cards
        cards = orchestrator.convert_to_cards(
            pipeline_result.generation.cards, metadata, qa_pairs, path_obj
        )

        # Serialize cards for return
        # We return dicts because Card objects might not be pickleable or we want to be safe
        cards_dicts = [card.model_dump() for card in cards]

        logger.info(
            "pipeline_generation_succeeded",
            file=relative_path,
            cards=len(cards_dicts),
            slugs=[card.slug for card in cards],
            **pipeline_context,
        )

        result = {"success": True, "cards": cards_dicts, "slugs": [c.slug for c in cards]}
        if result_queue_name:
            await _push_result(ctx, result_queue_name, result)
        return result

    except Exception as e:
        logger.exception("worker_job_failed", file=relative_path, error=str(e))
        result = {"success": False, "error": str(e), "cards": []}
        if result_queue_name:
            await _push_result(ctx, result_queue_name, result)
        return result


async def _push_result(ctx: dict[str, Any], queue_name: str, result: dict[str, Any]) -> None:
    """Push job result to Redis list."""
    try:
        import json
        # Use arq's redis connection from context if available, otherwise create new
        redis = ctx.get("redis")
        if not redis:
            # Fallback if ctx doesn't have redis (shouldn't happen in arq worker)
            return

        # Serialize result
        # We need to handle potential non-serializable objects if any remain
        # But cards_dicts should be pure python dicts/lists/etc.
        payload = json.dumps(result)
        await redis.rpush(queue_name, payload)
        # Set expiry on result queue to prevent memory leaks (e.g., 1 hour)
        await redis.expire(queue_name, 3600)
    except Exception as e:
        logger.error("failed_to_push_result", queue=queue_name, error=str(e))


class WorkerSettings:
    """Arq worker settings."""

    functions = [process_note_job]
    on_startup = startup
    on_shutdown = shutdown
    # Increased timeouts for complex notes with multiple LLM calls
    # Generation includes: pre-validation, card-splitting, generation, linter
    # Validation includes: post-validation with retries, context enrichment
    _gen_budget = float(os.getenv("WORKER_GENERATION_TIMEOUT_SECONDS", "2700"))
    _val_budget = float(os.getenv("WORKER_VALIDATION_TIMEOUT_SECONDS", "2700"))
    job_timeout = int(_gen_budget + _val_budget + 180)  # 5580s = 93 minutes total

    # Use environment variables with defaults
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    redis_settings.conn_timeout = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))
    max_jobs = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "50"))


def _detect_stage_sla(
    stage_times: dict[str, float] | None,
    generation_sla: float,
    validation_sla: float,
) -> dict[str, Any] | None:
    """Return metadata when a stage exceeds its configured budget."""
    stage_times = stage_times or {}
    generation_time = stage_times.get("generation")
    if generation_time is not None and generation_time > generation_sla:
        return {
            "stage": "generation",
            "elapsed": generation_time,
            "budget": generation_sla,
            "message": f"generation exceeded {generation_sla:.0f}s SLA (took {generation_time:.1f}s)",
        }
    validation_time = stage_times.get("post_validation")
    if validation_time is not None and validation_time > validation_sla:
        return {
            "stage": "post_validation",
            "elapsed": validation_time,
            "budget": validation_sla,
            "message": f"post_validation exceeded {validation_sla:.0f}s SLA (took {validation_time:.1f}s)",
        }
    return None
