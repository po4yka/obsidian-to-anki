"""Pipeline node functions for LangGraph workflow.

This module implements all 8 pipeline nodes that process notes through
the card generation workflow:
1. Note Correction (optional)
2. Pre-Validation
3. Card Splitting (optional)
4. Generation
5. Post-Validation
6. Context Enrichment (optional)
7. Memorization Quality (optional)
8. Duplicate Detection (optional)
"""

import asyncio
import time
from pathlib import Path

from ...models import NoteMetadata, QAPair
from ...providers.pydantic_ai_models import create_openrouter_model_from_env
from ...utils.logging import get_logger
from ..exceptions import ModelError, PreValidationError, StructuredOutputError
from ..models import (
    CardSplittingResult,
    ContextEnrichmentResult,
    GeneratedCard,
    GenerationResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from ..pydantic import (
    CardSplittingAgentAI,
    ContextEnrichmentAgentAI,
    DuplicateDetectionAgentAI,
    GeneratorAgentAI,
    MemorizationQualityAgentAI,
    PostValidatorAgentAI,
    PreValidatorAgentAI,
)
from .node_helpers import increment_step_count
from .state import PipelineState

logger = get_logger(__name__)


# ============================================================================
# Node Functions
# ============================================================================


async def note_correction_node(state: PipelineState) -> PipelineState:
    """Execute optional proactive note correction stage.

    Improves note quality before parsing (grammar, clarity, completeness).
    Only runs if enable_note_correction is True.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with corrected note content
    """
    from ..models import NoteCorrectionResult
    from ..parser_repair import ParserRepairAgent

    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "note_correction"):
        return state

    logger.info("langgraph_note_correction_start")
    start_time = time.time()

    # Check if note correction is enabled
    config = state["config"]
    if not getattr(config, "enable_note_correction", False):
        logger.info("note_correction_skipped", reason="disabled")
        state["current_stage"] = "pre_validation"
        return state

    # Get note correction model
    try:
        model_name = config.get_model_for_agent("note_correction")
        if not model_name or model_name == "":
            # Fallback to parser repair model if note correction model not set
            model_name = getattr(config, "parser_repair_model", "qwen3:8b")

        # Create provider for note correction (reuse parser repair infrastructure)
        from ...providers.factory import ProviderFactory

        provider = ProviderFactory.create_from_config(config)
        correction_model = model_name
        correction_temp = getattr(config, "note_correction_temperature", 0.0)

        # Create repair agent for proactive correction
        correction_agent = ParserRepairAgent(
            ollama_client=provider,
            model=correction_model,
            temperature=correction_temp,
            enable_content_generation=getattr(
                config, "parser_repair_generate_content", True
            ),
            repair_missing_sections=getattr(
                config, "repair_missing_sections", True),
        )

        # Perform proactive analysis and correction
        note_content = state.get("note_content", "")
        file_path = Path(state["file_path"]) if state.get(
            "file_path") else None

        correction_result = correction_agent.analyze_and_correct_proactively(
            content=note_content, file_path=file_path
        )

        # Update state with correction result
        state["note_correction"] = correction_result.model_dump()

        # If correction was applied, update note content
        if (
            correction_result.needs_correction
            and correction_result.corrected_content
        ):
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
            issues_found=[f"Correction failed: {str(e)}"],
            corrections_applied=[],
            confidence=0.0,
            correction_time=time.time() - start_time,
        )
        state["note_correction"] = fallback_result.model_dump()
        state["messages"].append("Note correction: failed, continuing")

    state["stage_times"]["note_correction"] = time.time() - start_time
    state["current_stage"] = "pre_validation"

    return state


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

    logger.info("langgraph_pre_validation_start")
    start_time = time.time()

    # Deserialize metadata and QA pairs
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]
    file_path = Path(state["file_path"]) if state["file_path"] else None

    # Use cached model from state, or create on demand as fallback
    model = state.get("pre_validator_model")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent("pre_validator")
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
            state["messages"].append(
                f"Pre-validation: {pre_result.error_type}")
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
        logger.error("langgraph_pre_validation_error",
                     error=str(e), details=e.details)
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
            error_details=f"Model error: {str(e)}",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
    except Exception as e:
        logger.exception(
            "langgraph_pre_validation_unexpected_error", error=str(e))
        pre_result = PreValidationResult(
            is_valid=False,
            error_type="structure",
            error_details=f"Unexpected error: {str(e)}",
            auto_fix_applied=False,
            fixed_content=None,
            validation_time=time.time() - start_time,
        )
    except BaseException:
        raise

    logger.info(
        "langgraph_pre_validation_complete",
        is_valid=pre_result.is_valid,
        time=pre_result.validation_time,
    )

    state["pre_validation"] = pre_result.model_dump()
    state["stage_times"]["pre_validation"] = pre_result.validation_time
    state["current_stage"] = "generation" if pre_result.is_valid else "failed"
    state["messages"].append(f"Pre-validation: {pre_result.error_type}")

    return state


async def card_splitting_node(state: PipelineState) -> PipelineState:
    """Execute card splitting analysis stage.

    Determines if note should generate 1 or N cards based on content complexity.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with card splitting results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "card_splitting"):
        return state

    logger.info("langgraph_card_splitting_start")
    start_time = time.time()

    # Check if card splitting is enabled
    if not state.get("enable_card_splitting", True):
        logger.info("card_splitting_skipped", reason="disabled")
        state["card_splitting"] = None
        state["current_stage"] = "generation"
        return state

    # Deserialize metadata and QA pairs
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

    # Use cached model from state, or create on demand as fallback
    model = state.get("card_splitting_model")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent("card_splitting")
            model = create_openrouter_model_from_env(model_name=model_name)
        except (ValueError, KeyError) as e:
            logger.warning(
                "failed_to_create_card_splitting_model", error=str(e))
            # Fallback: skip card splitting analysis
            state["card_splitting"] = None
            state["current_stage"] = "generation"
            state["stage_times"]["card_splitting"] = time.time() - start_time
            state["messages"].append(
                "Card splitting skipped (model unavailable)")
            return state

    # Create card splitting agent
    splitting_agent = CardSplittingAgentAI(model=model, temperature=0.0)

    # Get splitting preferences from config
    config = state["config"]
    preferred_size = getattr(config, "card_splitting_preferred_size", "medium")
    _prefer_splitting = getattr(
        config, "card_splitting_prefer_splitting", True
    )  # Reserved for future use
    min_confidence = getattr(config, "card_splitting_min_confidence", 0.7)
    max_cards = getattr(config, "card_splitting_max_cards_per_note", 10)

    # Run splitting analysis
    try:
        splitting_result = await splitting_agent.analyze(
            note_content=state["note_content"],
            metadata=metadata,
            qa_pairs=qa_pairs,
        )
        splitting_result.decision_time = time.time() - start_time

        # Apply preferences and safety limits
        # Check confidence threshold
        if splitting_result.confidence < min_confidence:
            logger.warning(
                "card_splitting_below_confidence_threshold",
                confidence=splitting_result.confidence,
                threshold=min_confidence,
                decision=splitting_result.should_split,
            )
            # If confidence is too low, use fallback strategy or default to no split
            if splitting_result.fallback_strategy:
                logger.info(
                    "card_splitting_using_fallback",
                    fallback=splitting_result.fallback_strategy,
                )
            elif splitting_result.should_split:
                # Low confidence on split - default to no split
                logger.info(
                    "card_splitting_low_confidence_defaulting_to_no_split",
                    confidence=splitting_result.confidence,
                )
                splitting_result.should_split = False
                splitting_result.card_count = 1
                splitting_result.splitting_strategy = "none"

        # Apply max cards safety limit
        if splitting_result.card_count > max_cards:
            logger.warning(
                "card_splitting_exceeds_max_cards",
                requested=splitting_result.card_count,
                max_allowed=max_cards,
            )
            splitting_result.card_count = max_cards
            # Truncate split plan if needed
            if len(splitting_result.split_plan) > max_cards:
                splitting_result.split_plan = splitting_result.split_plan[:max_cards]

        # Apply preferred size bias
        if preferred_size == "small" and not splitting_result.should_split:
            # Prefer smaller cards - encourage splitting
            logger.debug(
                "card_splitting_preferred_size_small_encouraging_split")
        elif preferred_size == "large" and splitting_result.should_split:
            # Prefer larger cards - might want to discourage splitting
            # But respect the agent's decision unless confidence is low
            if splitting_result.confidence < 0.8:
                logger.debug(
                    "card_splitting_preferred_size_large_low_confidence",
                    confidence=splitting_result.confidence,
                )
    except Exception as e:
        logger.exception("langgraph_card_splitting_error", error=str(e))
        # Fallback: assume no splitting needed
        splitting_result = CardSplittingResult(
            should_split=False,
            card_count=1,
            splitting_strategy="none",
            reasoning="",
            split_plan=[],
            decision_time=time.time() - start_time,
        )
    except BaseException:
        raise

    logger.info(
        "langgraph_card_splitting_complete",
        should_split=splitting_result.should_split,
        recommended_splits=len(splitting_result.split_plan),
        time=splitting_result.decision_time,
    )

    state["card_splitting"] = splitting_result.model_dump()
    state["stage_times"]["card_splitting"] = splitting_result.decision_time
    state["current_stage"] = "generation"
    state["messages"].append(
        f"Card splitting: {'split into ' + str(len(splitting_result.split_plan)) if splitting_result.should_split else 'no split needed'}"
    )

    return state


async def generation_node(state: PipelineState) -> PipelineState:
    """Execute card generation stage.

    Generates APF cards from Q/A pairs using the configured LLM.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with generation results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "generation"):
        return state

    logger.info("langgraph_generation_start")
    start_time = time.time()

    # Deserialize data
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

    # Use cached model from state, or create on demand as fallback
    model = state.get("generator_model")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent("generator")
            model = create_openrouter_model_from_env(model_name=model_name)
        except Exception as e:
            logger.error("failed_to_create_generator_model", error=str(e))
            # Return error result
            gen_result = GenerationResult(
                cards=[],
                total_cards=0,
                generation_time=time.time() - start_time,
                model_used="none",
            )
            state["generation"] = gen_result.model_dump()
            state["stage_times"]["generation"] = gen_result.generation_time
            state["current_stage"] = "failed"
            state["messages"].append("Generation failed: model unavailable")
            return state

    # Check if we have a split plan from card splitting agent
    card_splitting = state.get("card_splitting")
    expected_card_count = None
    split_plan = None
    if card_splitting:
        splitting_result = CardSplittingResult(**card_splitting)
        expected_card_count = splitting_result.card_count
        split_plan = splitting_result.split_plan
        logger.info(
            "langgraph_generation_with_split_plan",
            expected_cards=expected_card_count,
            strategy=splitting_result.splitting_strategy,
            confidence=splitting_result.confidence,
        )

    # Create generator agent
    generator = GeneratorAgentAI(model=model, temperature=0.3)

    # Run generation
    try:
        gen_result = await generator.generate_cards(
            note_content=state["note_content"],
            metadata=metadata,
            qa_pairs=qa_pairs,
            slug_base=state["slug_base"],
        )
        gen_result.generation_time = time.time() - start_time

        # Validate against split plan if available
        if expected_card_count is not None:
            actual_count = gen_result.total_cards
            if actual_count != expected_card_count:
                logger.warning(
                    "langgraph_generation_card_count_mismatch",
                    expected=expected_card_count,
                    actual=actual_count,
                    strategy=(
                        splitting_result.splitting_strategy if card_splitting else None
                    ),
                )
                # This is a warning, not an error - generation may produce different count
                # based on actual Q&A pairs vs split plan expectations

        # Log split plan alignment if available
        if split_plan and gen_result.cards:
            logger.debug(
                "langgraph_generation_split_plan_alignment",
                plan_cards=len(split_plan),
                generated_cards=len(gen_result.cards),
            )

    except Exception as e:
        logger.exception("langgraph_generation_error", error=str(e))

        # If we have a split plan, log it for debugging
        if split_plan:
            logger.error(
                "langgraph_generation_failed_with_split_plan",
                expected_cards=expected_card_count,
                split_plan_count=len(split_plan),
                error=str(e),
            )

        gen_result = GenerationResult(
            cards=[],
            total_cards=0,
            generation_time=time.time() - start_time,
            model_used=str(model),
        )
    except BaseException:
        raise

    logger.info(
        "langgraph_generation_complete",
        cards_count=gen_result.total_cards,
        expected_count=expected_card_count,
        time=gen_result.generation_time,
    )

    state["generation"] = gen_result.model_dump()
    state["stage_times"]["generation"] = gen_result.generation_time
    state["current_stage"] = (
        "post_validation" if gen_result.total_cards > 0 else "failed"
    )
    state["messages"].append(f"Generated {gen_result.total_cards} cards")

    return state


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

    logger.info("langgraph_post_validation_start",
                retry_count=state["retry_count"])
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
    model = state.get("post_validator_model")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent("post_validator")
            model = create_openrouter_model_from_env(model_name=model_name)
        except Exception as e:
            logger.warning(
                "failed_to_create_post_validator_model", error=str(e))
            # Assume valid if validator unavailable
            post_result = PostValidationResult(
                is_valid=True,
                error_type="none",
                error_details="Post-validation passed (OpenRouter unavailable, skipped)",
                corrected_cards=None,
                validation_time=time.time() - start_time,
            )
            state["post_validation"] = post_result.model_dump()
            state["stage_times"]["post_validation"] = (
                state["stage_times"].get("post_validation", 0.0)
                + post_result.validation_time
            )
            state["current_stage"] = "complete"
            state["messages"].append("Post-validation skipped")
            return state

    # Create post-validator agent
    post_validator = PostValidatorAgentAI(model=model, temperature=0.0)

    # Run validation
    try:
        post_result = await post_validator.validate(
            cards=cards,
            metadata=metadata,
            strict_mode=state["strict_mode"],
        )
        post_result.validation_time = time.time() - start_time
    except Exception as e:
        logger.exception("langgraph_post_validation_error", error=str(e))
        post_result = PostValidationResult(
            is_valid=False,
            error_type="syntax",
            error_details=f"Post-validation failed: {str(e)}",
            corrected_cards=None,
            validation_time=time.time() - start_time,
        )
    except BaseException:
        raise

    logger.info(
        "langgraph_post_validation_complete",
        is_valid=post_result.is_valid,
        retry_count=state["retry_count"],
        time=post_result.validation_time,
    )

    state["post_validation"] = post_result.model_dump()
    state["stage_times"]["post_validation"] = (
        state["stage_times"].get("post_validation", 0.0) +
        post_result.validation_time
    )

    # Determine next stage based on validation result
    if post_result.is_valid:
        # Move to enrichment if enabled, otherwise complete
        if state.get("enable_context_enrichment", True):
            state["current_stage"] = "context_enrichment"
        else:
            state["current_stage"] = "complete"
        state["messages"].append("Post-validation passed")
    elif (state.get("retry_count") or 0) < (state.get("max_retries") or 3) and state.get("auto_fix_enabled", False):
        # Apply corrections if available
        if post_result.corrected_cards and state["generation"] is not None:
            # Update generation with corrected cards
            corrected_dicts = [
                card.model_dump() for card in post_result.corrected_cards
            ]
            state["generation"]["cards"] = corrected_dicts
            state["generation"]["total_cards"] = len(corrected_dicts)
            logger.info("applied_corrected_cards", count=len(corrected_dicts))

        state["retry_count"] = (state.get("retry_count") or 0) + 1
        state["current_stage"] = "post_validation"  # Re-validate corrections
        state["messages"].append(
            f"Applied fixes, re-validating (attempt {state['retry_count']})"
        )
    else:
        state["current_stage"] = "failed"
        state["messages"].append("Post-validation failed, no more retries")

    return state


async def context_enrichment_node(state: PipelineState) -> PipelineState:
    """Execute context enrichment stage.

    Enhances generated cards with examples, mnemonics, and helpful context.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with enrichment results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "context_enrichment"):
        return state

    logger.info("langgraph_context_enrichment_start")
    start_time = time.time()

    # Check if enrichment is enabled
    if not state.get("enable_context_enrichment", True):
        logger.info("context_enrichment_skipped", reason="disabled")
        state["context_enrichment"] = None
        state["current_stage"] = (
            "memorization_quality"
            if state.get("enable_memorization_quality", True)
            else "complete"
        )
        return state

    # Check if we have cards to enrich
    if state["generation"] is None or not state["generation"]["cards"]:
        logger.warning("context_enrichment_no_cards")
        state["context_enrichment"] = None
        state["current_stage"] = (
            "memorization_quality"
            if state.get("enable_memorization_quality", True)
            else "complete"
        )
        return state

    try:
        # Use cached model from state, or create on demand as fallback
        model = state.get("context_enrichment_model")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent(
                "context_enrichment")
            model = create_openrouter_model_from_env(model_name=model_name)

        # Create enrichment agent
        enrichment_agent = ContextEnrichmentAgentAI(
            model=model, temperature=0.3)

        # Deserialize metadata and cards
        metadata = NoteMetadata(**state["metadata_dict"])
        cards = [
            GeneratedCard(**card_dict) for card_dict in state["generation"]["cards"]
        ]

        # Enrich each card in parallel
        tasks = [enrichment_agent.enrich(card, metadata) for card in cards]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        enriched_cards = []
        total_enriched = 0

        for i, result in enumerate(results):
            card = cards[i]
            if isinstance(result, Exception):
                logger.exception(
                    "card_enrichment_failed", slug=card.slug, error=str(result)
                )
                enriched_cards.append(card)  # Keep original on error
            elif result.should_enrich and result.enriched_card:
                enriched_cards.append(result.enriched_card)
                total_enriched += 1
                logger.info(
                    "card_enriched",
                    slug=card.slug,
                    additions=result.additions_summary,
                )
            else:
                enriched_cards.append(card)  # Keep original

        # Update generation with enriched cards
        state["generation"]["cards"] = [card.model_dump()
                                        for card in enriched_cards]

        # Create enrichment result summary
        enrichment_result = ContextEnrichmentResult(
            should_enrich=total_enriched > 0,
            enriched_card=None,  # Individual results not stored in summary
            additions=[],
            additions_summary=f"Enriched {total_enriched}/{len(cards)} cards",
            enrichment_rationale="Enhanced cards with examples and context",
            enrichment_time=time.time() - start_time,
        )

        state["context_enrichment"] = enrichment_result.model_dump()
        state["stage_times"]["context_enrichment"] = enrichment_result.enrichment_time

        logger.info(
            "langgraph_context_enrichment_complete",
            enriched_count=total_enriched,
            total_cards=len(cards),
            time=enrichment_result.enrichment_time,
        )

    except (ValueError, KeyError) as e:
        logger.warning("context_enrichment_failed", error=str(e))
        state["context_enrichment"] = None
        state["stage_times"]["context_enrichment"] = time.time() - start_time

    # Move to next stage
    state["current_stage"] = (
        "memorization_quality"
        if state.get("enable_memorization_quality", True)
        else "complete"
    )
    state["messages"].append(
        f"Context enrichment: {state['context_enrichment'] is not None}"
    )

    return state


async def memorization_quality_node(state: PipelineState) -> PipelineState:
    """Execute memorization quality assessment stage.

    Evaluates cards for spaced repetition effectiveness.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with quality assessment results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "memorization_quality"):
        return state

    logger.info("langgraph_memorization_quality_start")
    start_time = time.time()

    # Check if quality check is enabled
    if not state.get("enable_memorization_quality", True):
        logger.info("memorization_quality_skipped", reason="disabled")
        state["memorization_quality"] = None
        state["current_stage"] = "complete"
        return state

    # Check if we have cards to assess
    if state["generation"] is None or not state["generation"]["cards"]:
        logger.warning("memorization_quality_no_cards")
        state["memorization_quality"] = None
        state["current_stage"] = "complete"
        return state

    try:
        # Use cached model from state, or create on demand as fallback
        model = state.get("memorization_quality_model")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent(
                "memorization_quality")
            model = create_openrouter_model_from_env(model_name=model_name)

        # Create memorization quality agent
        quality_agent = MemorizationQualityAgentAI(
            model=model, temperature=0.0)

        # Deserialize metadata and cards
        metadata = NoteMetadata(**state["metadata_dict"])
        cards = [
            GeneratedCard(**card_dict) for card_dict in state["generation"]["cards"]
        ]

        # Assess all cards
        quality_result = await quality_agent.assess(cards, metadata)

        state["memorization_quality"] = quality_result.model_dump()
        state["stage_times"]["memorization_quality"] = quality_result.assessment_time

        logger.info(
            "langgraph_memorization_quality_complete",
            is_memorizable=quality_result.is_memorizable,
            score=quality_result.memorization_score,
            issues_count=len(quality_result.issues),
            time=quality_result.assessment_time,
        )

        # Log issues if any
        if not quality_result.is_memorizable:
            logger.warning(
                "memorization_quality_issues",
                score=quality_result.memorization_score,
                issues=quality_result.issues,
                improvements=quality_result.suggested_improvements,
            )

    except (ValueError, KeyError) as e:
        logger.warning("memorization_quality_failed", error=str(e))
        # Create permissive fallback result
        quality_result = MemorizationQualityResult(
            is_memorizable=True,
            memorization_score=0.7,
            issues=[],
            strengths=[],
            suggested_improvements=[f"Assessment failed: {str(e)}"],
            assessment_time=time.time() - start_time,
        )
        state["memorization_quality"] = quality_result.model_dump()
        state["stage_times"]["memorization_quality"] = quality_result.assessment_time

    # Move to duplicate detection if enabled, otherwise complete
    state["current_stage"] = (
        "duplicate_detection"
        if state.get("enable_duplicate_detection", False)
        else "complete"
    )
    # Safe access to memorization quality score
    mem_quality = state.get("memorization_quality")
    if mem_quality is not None and isinstance(mem_quality, dict):
        mem_score = mem_quality.get("memorization_score")
        if mem_score is not None:
            state["messages"].append(
                f"Memorization quality: score={mem_score:.2f}"
            )

    return state


async def duplicate_detection_node(state: PipelineState) -> PipelineState:
    """Execute duplicate detection stage.

    Checks newly generated cards against existing cards from Anki.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with duplicate detection results
    """
    # Check step limit (best practice: cycle protection)
    if not increment_step_count(state, "duplicate_detection"):
        return state

    logger.info("langgraph_duplicate_detection_start")
    start_time = time.time()

    # Check if duplicate detection is enabled
    if not state.get("enable_duplicate_detection", False):
        logger.info("duplicate_detection_skipped", reason="disabled")
        state["duplicate_detection"] = None
        state["current_stage"] = "complete"
        return state

    # Check if we have cards to check
    if state["generation"] is None or not state["generation"]["cards"]:
        logger.warning("duplicate_detection_no_cards")
        state["duplicate_detection"] = None
        state["current_stage"] = "complete"
        return state

    # Check if we have existing cards to compare against
    if not state.get("existing_cards_dicts"):
        logger.info("duplicate_detection_skipped", reason="no_existing_cards")
        state["duplicate_detection"] = None
        state["current_stage"] = "complete"
        return state

    try:
        # Use cached model from state, or create on demand as fallback
        model = state.get("duplicate_detection_model")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = state["config"].get_model_for_agent(
                "duplicate_detection")
            model = create_openrouter_model_from_env(model_name=model_name)

        # Create duplicate detection agent
        detection_agent = DuplicateDetectionAgentAI(
            model=model, temperature=0.0)

        # Deserialize cards
        new_cards = [
            GeneratedCard(**card_dict) for card_dict in state["generation"]["cards"]
        ]
        existing_cards_dicts = state.get("existing_cards_dicts")
        assert (
            existing_cards_dicts is not None
        ), "existing_cards_dicts should not be None"
        existing_cards = [
            GeneratedCard(**card_dict) for card_dict in existing_cards_dicts
        ]

        # Check each new card against existing cards in parallel
        tasks = [
            detection_agent.find_duplicates(new_card, existing_cards)
            for new_card in new_cards
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        duplicate_results = []
        total_duplicates_found = 0

        for i, result in enumerate(results):
            new_card = new_cards[i]
            if isinstance(result, Exception):
                logger.exception(
                    "card_duplicate_check_failed", slug=new_card.slug, error=str(result)
                )
                # Add empty result for this card
                duplicate_results.append(
                    {
                        "card_slug": new_card.slug,
                        "result": [],
                    }
                )
            else:
                duplicate_results.append(
                    {
                        "card_slug": new_card.slug,
                        "result": result.model_dump(),
                    }
                )

                if result.is_duplicate:
                    total_duplicates_found += 1
                    logger.warning(
                        "duplicate_card_detected",
                        new_slug=new_card.slug,
                        best_match=(
                            result.best_match.card_slug if result.best_match else None
                        ),
                        similarity=(
                            result.best_match.similarity_score
                            if result.best_match
                            else 0.0
                        ),
                        recommendation=result.recommendation,
                    )

        # Store all results
        detection_time = time.time() - start_time
        detection_summary = {
            "total_cards_checked": len(new_cards),
            "duplicates_found": total_duplicates_found,
            "results": duplicate_results,
            "detection_time": detection_time,
        }

        state["duplicate_detection"] = detection_summary
        state["stage_times"]["duplicate_detection"] = detection_time

        logger.info(
            "langgraph_duplicate_detection_complete",
            cards_checked=len(new_cards),
            duplicates_found=total_duplicates_found,
            time=detection_summary["detection_time"],
        )

    except (ValueError, KeyError) as e:
        logger.warning("duplicate_detection_failed", error=str(e))
        state["duplicate_detection"] = None
        state["stage_times"]["duplicate_detection"] = time.time() - start_time

    # Move to complete
    state["current_stage"] = "complete"
    duplicate_detection = state.get("duplicate_detection") or {}
    state["messages"].append(
        f"Duplicate detection: {duplicate_detection.get('duplicates_found', 0)} duplicates found"
    )

    return state
