"""Detection pipeline node functions for LangGraph workflow.

This module implements nodes responsible for detecting issues and duplicates:
- Duplicate detection: Checks newly generated cards against existing Anki cards
- Highlight agent: Suggests candidate Q&A pairs when generation fails
"""

import time

from obsidian_anki_sync.agents.exceptions import HighlightError
from obsidian_anki_sync.agents.pydantic import (
    DuplicateDetectionAgentAI,
    HighlightAgentAI,
)
from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.providers.pydantic_ai_models import (
    create_openrouter_model_from_env,
)
from obsidian_anki_sync.utils.logging import get_logger

from .node_helpers import increment_step_count
from .state import PipelineState, get_agent_selector, get_config, get_model, get_rag_integration

logger = get_logger(__name__)


async def duplicate_detection_node(state: PipelineState) -> PipelineState:
    """Execute duplicate detection stage.

    Checks newly generated cards against existing cards from Anki.
    If RAG is enabled, uses vector-based semantic similarity search first,
    then falls back to LLM-based detection for remaining cards.

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

    # Deserialize cards for checking
    new_cards = [
        GeneratedCard(**card_dict) for card_dict in state["generation"]["cards"]
    ]

    # RAG-based duplicate detection (faster, semantic similarity)
    rag_duplicate_results = []
    cards_to_check_with_llm = []

    if (
        state.get("enable_rag")
        and state.get("rag_duplicate_detection")
        and get_rag_integration(state)
    ):
        rag_integration = get_rag_integration(state)
        logger.info("rag_duplicate_detection_start", cards_count=len(new_cards))

        for card in new_cards:
            try:
                # Extract question and answer from APF HTML (simplified extraction)
                question = card.slug  # Use slug as question proxy
                answer = card.apf_html[:500] if card.apf_html else ""

                rag_result = await rag_integration.check_for_duplicates(
                    question=question,
                    answer=answer,
                )

                rag_duplicate_results.append(
                    {
                        "card_slug": card.slug,
                        "is_duplicate": rag_result.get("is_duplicate", False),
                        "confidence": rag_result.get("confidence", 0.0),
                        "recommendation": rag_result.get("recommendation", ""),
                        "similar_count": rag_result.get("similar_count", 0),
                        "source": "rag",
                    }
                )

                if rag_result.get("is_duplicate"):
                    logger.info(
                        "rag_duplicate_detected",
                        slug=card.slug,
                        confidence=rag_result.get("confidence"),
                        recommendation=rag_result.get("recommendation"),
                    )
                elif rag_result.get("confidence", 0) < 0.5:
                    # Low confidence from RAG, add to LLM check list
                    cards_to_check_with_llm.append(card)

            except Exception as e:
                logger.warning(
                    "rag_duplicate_check_failed",
                    slug=card.slug,
                    error=str(e),
                )
                cards_to_check_with_llm.append(card)

        state["rag_duplicate_results"] = rag_duplicate_results
        logger.info(
            "rag_duplicate_detection_complete",
            duplicates_found=sum(
                1 for r in rag_duplicate_results if r.get("is_duplicate")
            ),
            total_checked=len(new_cards),
            remaining_for_llm=len(cards_to_check_with_llm),
        )

        # If all cards checked by RAG and no LLM fallback needed
        if not cards_to_check_with_llm:
            detection_time = time.time() - start_time
            detection_summary = {
                "total_cards_checked": len(new_cards),
                "duplicates_found": sum(
                    1 for r in rag_duplicate_results if r.get("is_duplicate")
                ),
                "results": rag_duplicate_results,
                "detection_time": detection_time,
                "method": "rag",
            }
            state["duplicate_detection"] = detection_summary
            state["stage_times"]["duplicate_detection"] = detection_time
            state["current_stage"] = "complete"
            state["messages"].append(
                f"Duplicate detection (RAG): {detection_summary['duplicates_found']} duplicates found"
            )
            return state
    else:
        # No RAG, use LLM for all cards
        cards_to_check_with_llm = new_cards

    # Check if we have existing cards to compare against (for LLM-based detection)
    if not state.get("existing_cards_dicts") and cards_to_check_with_llm:
        logger.info("duplicate_detection_skipped", reason="no_existing_cards_for_llm")
        # Return RAG results if available
        if rag_duplicate_results:
            detection_time = time.time() - start_time
            detection_summary = {
                "total_cards_checked": len(new_cards),
                "duplicates_found": sum(
                    1 for r in rag_duplicate_results if r.get("is_duplicate")
                ),
                "results": rag_duplicate_results,
                "detection_time": detection_time,
                "method": "rag",
            }
            state["duplicate_detection"] = detection_summary
            state["stage_times"]["duplicate_detection"] = detection_time
        else:
            state["duplicate_detection"] = None
        state["current_stage"] = "complete"
        return state

    # LLM-based detection for remaining cards
    llm_duplicate_results = []
    llm_duplicates_found = 0

    if cards_to_check_with_llm:
        try:
            # Use cached model from state, or create on demand as fallback
            model = get_model(state, "duplicate_detection")
            if model is None:
                # Fallback: create model on demand if not cached
                model_name = get_config(state).get_model_for_agent(
                    "duplicate_detection"
                )
                model = create_openrouter_model_from_env(model_name=model_name)

            # Create duplicate detection agent
            detection_agent = DuplicateDetectionAgentAI(model=model, temperature=0.0)

            # Get existing cards for LLM comparison
            existing_cards_dicts = state.get("existing_cards_dicts")
            assert existing_cards_dicts is not None, (
                "existing_cards_dicts should not be None"
            )
            existing_cards = [
                GeneratedCard(**card_dict) for card_dict in existing_cards_dicts
            ]

            # Check each card against existing cards in parallel
            tasks = [
                detection_agent.find_duplicates(card, existing_cards)
                for card in cards_to_check_with_llm
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                card = cards_to_check_with_llm[i]
                if isinstance(result, Exception):
                    logger.exception(
                        "card_duplicate_check_failed", slug=card.slug, error=str(result)
                    )
                    llm_duplicate_results.append(
                        {
                            "card_slug": card.slug,
                            "is_duplicate": False,
                            "confidence": 0.0,
                            "source": "llm",
                            "result": [],
                        }
                    )
                else:
                    llm_duplicate_results.append(
                        {
                            "card_slug": card.slug,
                            "is_duplicate": result.is_duplicate,
                            "confidence": (
                                result.best_match.similarity_score
                                if result.best_match
                                else 0.0
                            ),
                            "recommendation": result.recommendation,
                            "source": "llm",
                            "result": result.model_dump(),
                        }
                    )

                    if result.is_duplicate:
                        llm_duplicates_found += 1
                        logger.warning(
                            "llm_duplicate_card_detected",
                            new_slug=card.slug,
                            best_match=(
                                result.best_match.card_slug
                                if result.best_match
                                else None
                            ),
                            similarity=(
                                result.best_match.similarity_score
                                if result.best_match
                                else 0.0
                            ),
                            recommendation=result.recommendation,
                        )

            logger.info(
                "llm_duplicate_detection_complete",
                cards_checked=len(cards_to_check_with_llm),
                duplicates_found=llm_duplicates_found,
            )

        except (ValueError, KeyError) as e:
            logger.warning("llm_duplicate_detection_failed", error=str(e))

    # Merge RAG and LLM results
    all_results = rag_duplicate_results + llm_duplicate_results
    total_duplicates = sum(1 for r in all_results if r.get("is_duplicate"))

    # Store all results
    detection_time = time.time() - start_time
    detection_summary = {
        "total_cards_checked": len(new_cards),
        "duplicates_found": total_duplicates,
        "rag_duplicates": sum(
            1 for r in rag_duplicate_results if r.get("is_duplicate")
        ),
        "llm_duplicates": llm_duplicates_found,
        "results": all_results,
        "detection_time": detection_time,
        "method": (
            "hybrid"
            if rag_duplicate_results and llm_duplicate_results
            else ("rag" if rag_duplicate_results else "llm")
        ),
    }

    state["duplicate_detection"] = detection_summary
    state["stage_times"]["duplicate_detection"] = detection_time

    logger.info(
        "langgraph_duplicate_detection_complete",
        cards_checked=len(new_cards),
        duplicates_found=total_duplicates,
        method=detection_summary["method"],
        time=detection_summary["detection_time"],
    )

    # Move to complete
    state["current_stage"] = "complete"
    state["messages"].append(
        f"Duplicate detection ({detection_summary['method']}): {total_duplicates} duplicates found"
    )

    return state


async def highlight_node(state: PipelineState) -> PipelineState:
    """Execute highlight agent to suggest candidate Q&A when generation fails."""

    if not increment_step_count(state, "highlight"):
        return state

    highlight_enabled = state.get("enable_highlight_agent", True)
    if not highlight_enabled:
        logger.info("highlight_agent_disabled")
        state["highlight_result"] = None
        state["stage_times"]["highlight"] = 0.0
        state["current_stage"] = "failed"
        state["messages"].append("Highlight agent disabled")
        return state

    config = get_config(state)
    note_content: str = state.get("note_content", "")
    metadata = NoteMetadata(**state["metadata_dict"])

    model = get_model(state, "highlight")
    if model is None:
        try:
            model_name = config.get_model_for_agent("highlight")
            model = create_openrouter_model_from_env(model_name=model_name)
        except (ValueError, KeyError) as exc:
            logger.warning("highlight_model_unavailable", error=str(exc))
            state["highlight_result"] = None
            state["stage_times"]["highlight"] = 0.0
            state["current_stage"] = "failed"
            state["messages"].append("Highlight agent unavailable (model missing)")
            return state

    max_candidates = getattr(config, "highlight_max_candidates", 3)
    highlight_agent = HighlightAgentAI(model=model, temperature=0.0)
    logger.info(
        "highlight_agent_start",
        title=metadata.title,
        max_candidates=max_candidates,
    )

    start_time = time.time()
    try:
        highlight_result = await highlight_agent.highlight(
            note_content=note_content,
            metadata=metadata,
            max_candidates=max_candidates,
        )
    except HighlightError as exc:
        logger.error("highlight_agent_failed", error=str(exc))
        state["highlight_result"] = None
        state["stage_times"]["highlight"] = time.time() - start_time
        state["current_stage"] = "failed"
        state["messages"].append("Highlight agent failed to analyze the note")
        return state

    highlight_time = highlight_result.analysis_time or (time.time() - start_time)
    state["highlight_result"] = highlight_result.model_dump()
    state["stage_times"]["highlight"] = highlight_time
    state["messages"].append(
        f"Highlight agent generated {len(highlight_result.qa_candidates)} candidate Q&A pairs"
    )
    state["current_stage"] = "failed"

    logger.info(
        "highlight_agent_complete",
        candidates=len(highlight_result.qa_candidates),
        confidence=highlight_result.confidence,
        time=highlight_time,
    )

    return state
