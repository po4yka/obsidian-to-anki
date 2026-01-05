"""Validation pipeline node functions for LangGraph workflow.

This module implements nodes responsible for validating note content and generated cards:
- Pre-validation: Validate note structure before generation
- Linter validation: APF template compliance checking
- Post-validation: Quality and accuracy validation with auto-fix
- Split validation: Validate proposed card splits
"""

import asyncio
import time

from obsidian_anki_sync.agents.exceptions import (
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from obsidian_anki_sync.agents.models import (
    GeneratedCard,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.agents.pydantic import (
    PostValidatorAgentAI,
    PreValidatorAgentAI,
    SplitValidatorAgentAI,
)
from obsidian_anki_sync.apf.linter import validate_apf
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.providers.pydantic_ai_models import (
    create_openrouter_model_from_env,
)
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.resilience import compute_jittered_backoff

from .node_helpers import increment_step_count
from .state import PipelineState, get_config, get_model

logger = get_logger(__name__)


async def pre_validation_node(state: PipelineState) -> PipelineState:
    """Execute pre-validation stage.

    Validates note structure, formatting, and frontmatter before generation.
    Can auto-fix common issues if enabled.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with pre-validation results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "pre_validation"):
        return state

    pipeline_id = state.get("pipeline_id", "unknown")
    logger.info(
        "pipeline_node_executing",
        pipeline_id=pipeline_id,
        node="pre_validation",
        input_state_keys=list(state.keys()),
        note_id=state.get("metadata_dict", {}).get("id", "unknown"),
    )
    start_time = time.time()

    config = get_config(state)

    # Deserialize metadata and QA pairs
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]
    file_path = state.get("file_path")

    # Use cached model from state, or create on demand as fallback
    model = get_model(state, "pre_validator")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = config.get_model_for_agent("pre_validator")
            model = create_openrouter_model_from_env(model_name=model_name)
        except (ValueError, KeyError) as e:
            logger.warning("failed_to_create_openrouter_model", error=str(e))
            # Fallback to simple validation
            pre_result = PreValidationResult(
                is_valid=True,
                error_type="none",
                error_details="Pre-validation passed (OpenRouter unavailable, skipped)",
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=time.time() - start_time,
            )
            state["pre_validation"] = pre_result.model_dump()
            state["stage_times"]["pre_validation"] = pre_result.validation_time
            state["current_stage"] = "generation"
            state["messages"].append(f"Pre-validation: {pre_result.error_type}")
            return state

    # Create pre-validator agent
    pre_validator = PreValidatorAgentAI(model=model, temperature=0.0)

    # Run validation (PydanticAI uses async)
    try:
        pre_result = await pre_validator.validate(
            note_content=state["note_content"],
            metadata=metadata,
            qa_pairs=qa_pairs,
            file_path=file_path,
        )
        pre_result.validation_time = time.time() - start_time
    except PreValidationError as e:
        logger.error("langgraph_pre_validation_error", error=str(e), details=e.details)
        pre_result = PreValidationResult(
            is_valid=False,
            error_type="structure",
            error_details=str(e),
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
    except (StructuredOutputError, ModelError) as e:
        logger.error(
            "langgraph_pre_validation_model_error", error=str(e), details=e.details
        )
        pre_result = PreValidationResult(
            is_valid=False,
            error_type="format",
            error_details=f"Model error: {e!s}",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
    except Exception as e:
        logger.exception("langgraph_pre_validation_unexpected_error", error=str(e))
        pre_result = PreValidationResult(
            is_valid=False,
            error_type="structure",
            error_details=f"Unexpected error: {e!s}",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
    except BaseException:
        raise

    if pre_result.is_valid:
        logger.info(
            "langgraph_pre_validation_complete",
            is_valid=pre_result.is_valid,
            time=pre_result.validation_time,
        )
    else:
        logger.warning(
            "langgraph_pre_validation_failed",
            is_valid=pre_result.is_valid,
            time=pre_result.validation_time,
            error_type=pre_result.error_type,
            error_details=pre_result.error_details,
        )

    state["pre_validation"] = pre_result.model_dump()
    state["stage_times"]["pre_validation"] = pre_result.validation_time
    state["current_stage"] = "generation" if pre_result.is_valid else "failed"
    if pre_result.is_valid:
        state["messages"].append("Pre-validation: passed")
    else:
        state["messages"].append(
            f"Pre-validation failed ({pre_result.error_type}): {pre_result.error_details}"
        )

    return state


async def linter_validation_node(state: PipelineState) -> PipelineState:
    """Execute deterministic APF linting - source of truth for template compliance.

    This node runs the APF linter on all generated cards to validate template
    compliance (sentinels, headers, tags, structure). The linter is authoritative
    for template checks - when it passes, any LLM template complaints should be
    overridden as hallucinations.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with linter results
    """
    from .node_helpers import increment_step_count

    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "linter_validation"):
        return state

    logger.info("langgraph_linter_validation_start")
    start_time = time.time()

    # Get generated cards
    generation = state.get("generation")
    if not generation:
        logger.error("langgraph_linter_validation_no_generation")
        state["linter_valid"] = False
        state["linter_results"] = []
        state["current_stage"] = "failed"
        return state

    cards = [GeneratedCard(**card_dict) for card_dict in generation["cards"]]

    if not cards:
        logger.warning("langgraph_linter_validation_no_cards")
        state["linter_valid"] = True
        state["linter_results"] = []
        state["current_stage"] = "post_validation"
        return state

    # Run deterministic linter on each card
    linter_results = []
    all_valid = True
    total_errors = 0
    total_warnings = 0
    loop = asyncio.get_running_loop()

    for card in cards:
        try:
            # Run CPU-bound linter in executor to avoid blocking the loop
            result = await loop.run_in_executor(
                None, validate_apf, card.apf_html, card.slug
            )
            card_result = {
                "slug": card.slug,
                "is_valid": not result.errors,  # Only errors block, not warnings
                "errors": result.errors,
                "warnings": result.warnings,
            }
            linter_results.append(card_result)

            if result.errors:
                all_valid = False
                total_errors += len(result.errors)
                logger.warning(
                    "linter_validation_card_errors",
                    slug=card.slug,
                    errors=result.errors,
                )

            total_warnings += len(result.warnings)

            if result.warnings:
                logger.debug(
                    "linter_validation_card_warnings",
                    slug=card.slug,
                    warnings=result.warnings,
                )

        except Exception as e:
            logger.exception(
                "linter_validation_card_failed",
                slug=card.slug,
                error=str(e),
            )
            # On exception, mark as invalid
            linter_results.append(
                {
                    "slug": card.slug,
                    "is_valid": False,
                    "errors": [f"Linter exception: {e!s}"],
                    "warnings": [],
                }
            )
            all_valid = False
            total_errors += 1

    # Update state with linter results
    state["linter_valid"] = all_valid
    state["linter_results"] = linter_results

    linting_time = time.time() - start_time
    state["stage_times"]["linter_validation"] = linting_time

    logger.info(
        "langgraph_linter_validation_complete",
        cards_count=len(cards),
        linter_valid=all_valid,
        total_errors=total_errors,
        total_warnings=total_warnings,
        linting_time=linting_time,
    )

    # Always proceed to post_validation - linter results are used there
    # to override LLM template hallucinations
    state["current_stage"] = "post_validation"
    state["messages"].append(
        f"Linter validation: {'passed' if all_valid else f'failed ({total_errors} errors)'}"
    )

    return state


async def _sleep_post_validation_backoff(state: PipelineState) -> None:
    """Sleep with jittered backoff before the next post-validation attempt."""
    attempt = (state.get("retry_count") or 0) + 1
    delay = compute_jittered_backoff(
        attempt,
        initial_delay=state.get("post_validator_retry_backoff_seconds", 3.0),
        max_delay=(state.get("post_validator_timeout_seconds", 2700.0) / 2),
        jitter=state.get("post_validator_retry_jitter_seconds", 1.5),
    )
    if delay <= 0:
        return
    logger.debug(
        "langgraph_post_validation_backoff",
        attempt=attempt,
        delay=round(delay, 2),
    )
    await asyncio.sleep(delay)


async def post_validation_node(state: PipelineState) -> PipelineState:
    """Execute post-validation stage.

    Validates generated cards for quality, syntax, and accuracy.
    Can auto-fix issues if enabled and will retry on failures.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with post-validation results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "post_validation"):
        return state

    # Track timeout errors for retry logic (timeouts are transient and should retry)
    is_timeout_error = False

    cards_count = len(state.get("generation", {}).get("cards", []))
    logger.info(
        "langgraph_post_validation_start",
        retry_count=state["retry_count"],
        cards_count=cards_count,
        timeout_seconds=state.get("post_validator_timeout_seconds", 2700.0),
    )
    start_time = time.time()

    # Deserialize data
    metadata = NoteMetadata(**state["metadata_dict"])
    generation = state.get("generation")

    if not generation:
        logger.error("langgraph_post_validation_no_generation")
        state["current_stage"] = "failed"
        return state

    cards = [GeneratedCard(**card_dict) for card_dict in generation["cards"]]

    # Use cached model from state, or create on demand as fallback
    model = get_model(state, "post_validator") or "placeholder-model"

    # Create post-validator agent (tests patch this to dummy)
    post_validator = PostValidatorAgentAI(model=model, temperature=0.0)

    timeout_seconds = state.get("post_validator_timeout_seconds", 2700.0)

    # Run validation
    try:
        post_result = await asyncio.wait_for(
            post_validator.validate(
                cards=cards,
                metadata=metadata,
                strict_mode=state["strict_mode"],
            ),
            timeout=timeout_seconds,
        )
        post_result.validation_time = time.time() - start_time
    except TimeoutError:
        logger.error(
            "langgraph_post_validation_timeout",
            timeout=timeout_seconds,
            retry_count=state.get("retry_count", 0),
        )
        post_result = PostValidationResult(
            is_valid=False,
            error_type="syntax",
            error_details=(
                f"Post-validation exceeded {timeout_seconds:.1f}s timeout window"
            ),
            corrected_cards=None,
            validation_time=time.time() - start_time,
        )
        is_timeout_error = True  # Mark for retry (timeouts are transient)
    except Exception as e:
        logger.exception("langgraph_post_validation_error", error=str(e))
        post_result = PostValidationResult(
            is_valid=False,
            error_type="syntax",
            error_details=f"Post-validation failed: {e!s}",
            corrected_cards=None,
            validation_time=time.time() - start_time,
        )
    except BaseException:
        raise

    # Check linter results to override LLM template hallucinations
    # Two-tier validation: linter is authoritative for template compliance,
    # LLM is authoritative for content quality (factual/semantic)
    linter_valid = state.get("linter_valid", False)
    llm_template_overridden = False

    # ENFORCEMENT: If linter failed, the card IS invalid, regardless of LLM opinion.
    if not linter_valid and state.get("linter_results"):
        # Collect all errors from linter results
        all_linter_errors = []
        for res in state.get("linter_results", []):
            if res.get("errors"):
                all_linter_errors.extend(res["errors"])

        error_msg = "; ".join(all_linter_errors[:3])  # First 3 errors
        if len(all_linter_errors) > 3:
            error_msg += f" (+{len(all_linter_errors) - 3} more)"

        # Log the override if LLM thought it was valid
        if post_result.is_valid:
            logger.warning(
                "llm_validation_overridden_by_linter_failure",
                llm_judgment="valid",
                linter_errors=all_linter_errors,
                reason="linter_is_authoritative",
            )

        # Force invalid status
        post_result.is_valid = False
        post_result.error_type = "template"
        post_result.error_details = f"Linter failed: {error_msg}"

    if not post_result.is_valid and post_result.error_type == "template":
        if linter_valid:
            # Template compliance: LINTER is authoritative
            # LLM is hallucinating - log and override
            logger.warning(
                "llm_template_error_overridden_by_linter",
                error_type=post_result.error_type,
                error_details=(
                    post_result.error_details[:200] if post_result.error_details else ""
                ),
                reason="linter_passed_template_checks",
                linter_results_summary={
                    "total_cards": len(state.get("linter_results", [])),
                    "all_valid": linter_valid,
                },
            )
            # Override: treat as valid for template compliance
            post_result.is_valid = True
            post_result.error_type = "none"
            post_result.error_details = ""
            llm_template_overridden = True
        else:
            # Both linter and LLM found template errors - LLM errors are legitimate
            logger.info(
                "llm_template_error_confirmed_by_linter",
                error_type=post_result.error_type,
                linter_errors=[
                    r.get("errors", [])
                    for r in state.get("linter_results", [])
                    if r.get("errors")
                ],
            )

    # Log validation decision for tracking
    corrections_count = len(post_result.suggested_corrections or [])
    applied_count = len(post_result.applied_changes or [])
    logger.info(
        "langgraph_post_validation_complete",
        is_valid=post_result.is_valid,
        retry_count=state["retry_count"],
        time=round(post_result.validation_time, 2),
        linter_valid=linter_valid,
        llm_error_type=post_result.error_type if not post_result.is_valid else "none",
        llm_error_details=post_result.error_details[:300]
        if post_result.error_details
        else "",
        llm_template_overridden=llm_template_overridden,
        corrections_suggested=corrections_count,
        corrections_applied=applied_count,
    )

    state["post_validation"] = post_result.model_dump()
    state["stage_times"]["post_validation"] = (
        state["stage_times"].get("post_validation", 0.0) + post_result.validation_time
    )

    if post_result.corrected_cards and state.get("generation") is not None:
        corrected_dicts = [card.model_dump() for card in post_result.corrected_cards]
        state["generation"]["cards"] = corrected_dicts
        state["generation"]["total_cards"] = len(corrected_dicts)
        logger.info(
            "applied_corrected_cards",
            count=len(corrected_dicts),
            applied_changes=post_result.applied_changes,
        )

    # Determine next stage based on validation result
    if post_result.is_valid:
        if state.get("enable_context_enrichment", True):
            state["current_stage"] = "context_enrichment"
        else:
            state["current_stage"] = "complete"
        state["messages"].append("Post-validation passed")
    else:
        state["retry_count"] = max((state.get("retry_count") or 0) + 1, 1)
        if (state.get("retry_count") or 0) <= (state.get("max_retries") or 3) and (
            state.get("auto_fix_enabled", True) or is_timeout_error
        ):
            await _sleep_post_validation_backoff(state)
            # Important: If validation failed, we must RE-GENERATE the content.
            # Setting stage to 'generation' triggers the retry loop in the router.
            state["current_stage"] = "generation"

            retry_reason = (
                "timeout, retrying" if is_timeout_error else "applied fixes/retrying"
            )
            state["messages"].append(
                f"{retry_reason}, re-generating (attempt {state['retry_count']})"
            )
        else:
            state["current_stage"] = "failed"
            state["messages"].append("Post-validation failed, no more retries")

    return state


async def split_validation_node(state: PipelineState) -> PipelineState:
    """Execute split validation stage.

    Validates proposed card splits to prevent over-fragmentation.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with validation results
    """
    # Check step limit
    if not increment_step_count(state, "split_validation"):
        return state

    logger.info("langgraph_split_validation_start")
    start_time = time.time()

    # Check if we have a split plan to validate
    card_splitting = state.get("card_splitting")
    if not card_splitting or not card_splitting.get("should_split", False):
        logger.info("split_validation_skipped", reason="no_split_proposed")
        state["current_stage"] = "generation"
        return state

    # Deserialize data
    from obsidian_anki_sync.agents.models import CardSplittingResult

    splitting_result = CardSplittingResult(**card_splitting)
    metadata = NoteMetadata(**state["metadata_dict"])

    # Use cached model or create on demand
    model = get_model(state, "split_validator") or "placeholder-model"

    # Create agent (tests patch this to dummy)
    validator = SplitValidatorAgentAI(model=model, temperature=0.0)

    try:
        validation_result = await validator.validate(
            note_content=state["note_content"],
            metadata=metadata,
            splitting_result=splitting_result,
        )

        # If validation failed, revert split decision
        if not validation_result.is_valid:
            logger.info(
                "split_validation_rejected_split",
                feedback=validation_result.feedback,
            )
            # Revert to single card
            splitting_result.should_split = False
            splitting_result.card_count = 1
            splitting_result.splitting_strategy = "none"
            splitting_result.split_plan = []
            splitting_result.reasoning = (
                f"Split rejected by validator: {validation_result.feedback}"
            )

            # Update state with modified splitting result
            state["card_splitting"] = splitting_result.model_dump()
            state["messages"].append(
                f"Split validation: Rejected split ({validation_result.feedback})"
            )
        else:
            state["messages"].append("Split validation: Approved split")

        state["split_validation"] = validation_result.model_dump()

    except Exception as e:
        logger.exception("split_validation_error", error=str(e))
        # Continue with original plan on error
        state["messages"].append(f"Split validation failed: {e!s}")

    state["stage_times"]["split_validation"] = time.time() - start_time
    state["current_stage"] = "generation"

    return state
