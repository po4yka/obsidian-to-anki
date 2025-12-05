"""Enhancement pipeline node functions for LangGraph workflow.

This module implements nodes that enhance generated cards with additional
context and quality assessments:
- Context enrichment: Adds examples, mnemonics, and helpful context
- Memorization quality: Evaluates cards for spaced repetition effectiveness
"""

import asyncio
import time

from obsidian_anki_sync.agents.models import (
    ContextEnrichmentResult,
    GeneratedCard,
    MemorizationQualityResult,
)
from obsidian_anki_sync.agents.pydantic import (
    ContextEnrichmentAgentAI,
    MemorizationQualityAgentAI,
)
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.providers.pydantic_ai_models import (
    create_openrouter_model_from_env,
)
from obsidian_anki_sync.utils.logging import get_logger

from .node_helpers import increment_step_count
from .state import PipelineState, get_config, get_model

logger = get_logger(__name__)


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
        model = get_model(state, "context_enrichment")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = get_config(state).get_model_for_agent("context_enrichment")
            model = create_openrouter_model_from_env(model_name=model_name)

        # Create enrichment agent
        enrichment_agent = ContextEnrichmentAgentAI(model=model, temperature=0.3)

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
        state["generation"]["cards"] = [card.model_dump() for card in enriched_cards]

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
        model = get_model(state, "memorization_quality")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = get_config(state).get_model_for_agent("memorization_quality")
            model = create_openrouter_model_from_env(model_name=model_name)

        # Create memorization quality agent
        quality_agent = MemorizationQualityAgentAI(model=model, temperature=0.0)

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
            suggested_improvements=[f"Assessment failed: {e!s}"],
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
            state["messages"].append(f"Memorization quality: score={mem_score:.2f}")

    return state
