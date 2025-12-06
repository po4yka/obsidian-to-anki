"""Correction pipeline node functions for LangGraph workflow.

This module implements nodes responsible for correcting and fixing note content
before processing through the main pipeline:
- Auto-fix: Deterministic fixes for common note issues
- Note Correction: Optional proactive note correction for quality improvement
"""

import time

from obsidian_anki_sync.agents.models import NoteCorrectionResult
from obsidian_anki_sync.agents.parser_repair import ParserRepairAgent
from obsidian_anki_sync.utils.logging import get_logger

from .node_helpers import increment_step_count
from .state import PipelineState, get_config

logger = get_logger(__name__)


async def autofix_node(state: PipelineState) -> PipelineState:
    """Execute auto-fix stage to correct note issues before processing.

    This node runs deterministic fixes for common note issues:
    - Trailing whitespace
    - Empty references sections
    - Title format (bilingual)
    - MOC field mismatches
    - Section ordering
    - Missing Related Questions sections
    - Broken wikilinks
    - Broken related entries

    This is a permanent step that always runs as the first stage of the pipeline.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with fixed content and autofix results
    """
    from obsidian_anki_sync.agents.autofix import AutoFixRegistry

    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "autofix"):
        return state

    logger.info("langgraph_autofix_start")
    start_time = time.time()

    try:
        # Get configuration
        config = get_config(state)
        note_content = state.get("note_content", "")
        file_path = state.get("file_path")

        # Get note index for link validation (if available from config)
        note_index = getattr(config, "note_index", None) or set()

        # Get enabled handlers from state or use all
        enabled_handlers = state.get("autofix_handlers")

        # Initialize registry
        registry = AutoFixRegistry(
            note_index=note_index,
            enabled_handlers=enabled_handlers,
            write_back=state.get("autofix_write_back", False),
        )

        # Run auto-fix
        result = registry.fix_all(content=note_content, file_path=file_path)

        # Update state with fixed content if modifications were made
        if result.file_modified and result.fixed_content:
            state["note_content"] = result.fixed_content
            logger.info(
                "autofix_content_updated",
                issues_fixed=result.issues_fixed,
                file_path=str(file_path) if file_path else None,
            )

        # Store result in state
        state["autofix"] = result.model_dump()
        state["stage_times"]["autofix"] = time.time() - start_time
        state["messages"].append(
            f"Autofix: {result.issues_fixed}/{len(result.issues_found)} issues fixed"
        )

        logger.info(
            "langgraph_autofix_complete",
            issues_found=len(result.issues_found),
            issues_fixed=result.issues_fixed,
            file_modified=result.file_modified,
            duration_ms=round((time.time() - start_time) * 1000, 2),
        )

    except Exception as e:
        logger.exception("autofix_error", error=str(e))
        # Non-blocking: continue to next stage even if autofix fails
        state["autofix"] = {
            "file_modified": False,
            "issues_found": [],
            "issues_fixed": 0,
            "issues_skipped": 0,
            "fix_time": time.time() - start_time,
        }
        state["messages"].append(f"Autofix: failed ({e})")

    # Move to next stage
    state["current_stage"] = "note_correction"
    return state


async def note_correction_node(state: PipelineState) -> PipelineState:
    """Execute optional proactive note correction stage.

    Improves note quality before parsing (grammar, clarity, completeness).
    Only runs if enable_note_correction is True.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with corrected note content
    """

    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "note_correction"):
        return state

    logger.info("langgraph_note_correction_start")
    start_time = time.time()

    # Check if note correction is enabled
    config = get_config(state)
    if not getattr(config, "enable_note_correction", False):
        logger.info("note_correction_skipped", reason="disabled")
        state["current_stage"] = "pre_validation"
        return state

    # Get note correction model
    try:
        model_name = config.get_model_for_agent("note_correction")
        if not model_name or model_name == "":
            # Fallback to parser repair model if note correction model not set
            model_name = config.get_model_for_agent("parser_repair")

        # Create provider for note correction (reuse parser repair infrastructure)
        from obsidian_anki_sync.providers.factory import ProviderFactory

        provider = ProviderFactory.create_from_config(config)
        correction_model = model_name
        correction_config = config.get_model_config_for_task("parser_repair")
        correction_temp = correction_config.get("temperature", 0.0)

        # Create repair agent for proactive correction
        correction_agent = ParserRepairAgent(
            ollama_client=provider,
            model=correction_model,
            temperature=correction_temp,
            enable_content_generation=getattr(
                config, "parser_repair_generate_content", True
            ),
            repair_missing_sections=getattr(config, "repair_missing_sections", True),
        )

        # Perform proactive analysis and correction
        note_content = state.get("note_content", "")
        file_path = state.get("file_path")

        correction_result = (
            await correction_agent.analyze_and_correct_proactively_async(
                content=note_content, file_path=file_path
            )
        )

        # Update state with correction result
        state["note_correction"] = correction_result.model_dump()

        # If correction was applied, update note content
        if correction_result.needs_correction and correction_result.corrected_content:
            state["note_content"] = correction_result.corrected_content
            logger.info(
                "note_correction_applied",
                issues_found=len(correction_result.issues_found),
                corrections_applied=len(correction_result.corrections_applied),
                quality_before=correction_result.quality_score,
                quality_after=(
                    correction_result.quality_after.overall_score
                    if correction_result.quality_after
                    else None
                ),
            )
            state["messages"].append(
                f"Note correction: {len(correction_result.corrections_applied)} corrections applied"
            )
        else:
            logger.info(
                "note_correction_no_action_needed",
                quality_score=correction_result.quality_score,
            )
            state["messages"].append(
                f"Note correction: quality score {correction_result.quality_score:.2f}, no corrections needed"
            )

    except Exception as e:
        logger.warning("note_correction_failed", error=str(e))
        # Create fallback result
        fallback_result = NoteCorrectionResult(
            needs_correction=False,
            quality_score=0.5,
            issues_found=[f"Correction failed: {e!s}"],
            corrections_applied=[],
            confidence=0.0,
            correction_time=time.time() - start_time,
        )
        state["note_correction"] = fallback_result.model_dump()
        state["messages"].append("Note correction: failed, continuing")

    state["stage_times"]["note_correction"] = time.time() - start_time
    state["current_stage"] = "pre_validation"

    return state

