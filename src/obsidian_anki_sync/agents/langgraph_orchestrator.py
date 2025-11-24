"""LangGraph-based orchestrator for the card generation pipeline.

This module implements a state machine workflow using LangGraph to coordinate:
1. Pre-Validator Agent - structure and format validation
2. Generator Agent - card generation with LLM
3. Post-Validator Agent - quality validation with retry logic

The workflow supports conditional routing, automatic retries, and state persistence.
"""

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..config import Config
from ..models import NoteMetadata, QAPair
from ..utils.logging import get_logger
from .exceptions import (
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from .models import (
    AgentPipelineResult,
    CardSplittingResult,
    ContextEnrichmentResult,
    GeneratedCard,
    GenerationResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from .slug_utils import generate_agent_slug_base

logger = get_logger(__name__)


# ============================================================================
# State Definition
# ============================================================================


class PipelineState(TypedDict):
    """State for the card generation pipeline workflow.

    This state is passed between nodes and tracks the entire pipeline execution.
    """

    # Input data
    note_content: str
    metadata_dict: dict  # Serialized NoteMetadata
    qa_pairs_dicts: list[dict]  # Serialized QAPair list
    file_path: str | None
    slug_base: str
    config: Config  # Service configuration for model selection
    existing_cards_dicts: (
        list[dict] | None
    )  # Serialized existing cards for duplicate check

    # Cached models (for performance - created once during orchestrator init)
    pre_validator_model: Any | None  # PydanticAI OpenAIModel instance
    card_splitting_model: Any | None
    generator_model: Any | None
    post_validator_model: Any | None
    context_enrichment_model: Any | None
    memorization_quality_model: Any | None
    duplicate_detection_model: Any | None

    # Pipeline stage results
    pre_validation: dict | None  # Serialized PreValidationResult
    card_splitting: dict | None  # Serialized CardSplittingResult
    generation: dict | None  # Serialized GenerationResult
    post_validation: dict | None  # Serialized PostValidationResult
    context_enrichment: dict | None  # Serialized ContextEnrichmentResult
    memorization_quality: dict | None  # Serialized MemorizationQualityResult
    # Serialized DuplicateDetectionResult (per card)
    duplicate_detection: dict | None

    # Workflow control
    current_stage: Literal[
        "pre_validation",
        "card_splitting",
        "generation",
        "post_validation",
        "context_enrichment",
        "memorization_quality",
        "duplicate_detection",
        "complete",
        "failed",
    ]
    enable_card_splitting: bool
    enable_context_enrichment: bool
    enable_memorization_quality: bool
    enable_duplicate_detection: bool
    retry_count: int
    max_retries: int
    auto_fix_enabled: bool
    strict_mode: bool

    # Timing
    start_time: float
    stage_times: dict[str, float]

    # Messages (for debugging/logging)
    messages: Annotated[list[str], add_messages]


# ============================================================================
# Node Functions
# ============================================================================


async def pre_validation_node(state: PipelineState) -> PipelineState:
    """Execute pre-validation stage.

    Validates note structure, formatting, and frontmatter before generation.
    Can auto-fix common issues if enabled.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with pre-validation results
    """
    from ..models import NoteMetadata, QAPair
    from ..providers.pydantic_ai_models import (
        create_openrouter_model_from_env,
    )
    from .pydantic_ai_agents import PreValidatorAgentAI

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
    from ..models import NoteMetadata, QAPair
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import CardSplittingAgentAI

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

    # Run splitting analysis
    try:
        splitting_result = await splitting_agent.analyze(
            note_content=state["note_content"],
            metadata=metadata,
            qa_pairs=qa_pairs,
        )
        splitting_result.decision_time = time.time() - start_time
    except Exception as e:
        logger.exception("langgraph_card_splitting_error", error=str(e))
        # Fallback: assume no splitting needed
        from .models import CardSplittingResult

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
    from ..models import NoteMetadata, QAPair
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import GeneratorAgentAI

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
    except Exception as e:
        logger.exception("langgraph_generation_error", error=str(e))
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
    from ..models import NoteMetadata
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import PostValidatorAgentAI

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
    elif state["retry_count"] < state["max_retries"] and state["auto_fix_enabled"]:
        # Apply corrections if available
        if post_result.corrected_cards and state["generation"] is not None:
            # Update generation with corrected cards
            corrected_dicts = [
                card.model_dump() for card in post_result.corrected_cards
            ]
            state["generation"]["cards"] = corrected_dicts
            state["generation"]["total_cards"] = len(corrected_dicts)
            logger.info("applied_corrected_cards", count=len(corrected_dicts))

        state["retry_count"] += 1
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
    from ..models import NoteMetadata
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import ContextEnrichmentAgentAI

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
    from ..models import NoteMetadata
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import MemorizationQualityAgentAI

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
    if state["memorization_quality"] is not None:
        state["messages"].append(
            f"Memorization quality: score={state['memorization_quality']['memorization_score']:.2f}"
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
    from ..providers.pydantic_ai_models import create_openrouter_model_from_env
    from .pydantic_ai_agents import DuplicateDetectionAgentAI

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


# ============================================================================
# Conditional Routing
# ============================================================================


def should_continue_after_pre_validation(
    state: PipelineState,
) -> Literal["card_splitting", "generation", "failed"]:
    """Determine next node after pre-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    pre_validation = state.get("pre_validation")
    if pre_validation and pre_validation["is_valid"]:
        # Route to card splitting if enabled, otherwise directly to generation
        if state.get("enable_card_splitting", True):
            return "card_splitting"
        return "generation"
    return "failed"


def should_continue_after_post_validation(
    state: PipelineState,
) -> Literal["context_enrichment", "generation", "failed"]:
    """Determine next node after post-validation.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "failed")

    if current_stage == "context_enrichment":
        return "context_enrichment"
    elif current_stage == "generation":
        return "generation"  # Retry
    else:
        return "failed"


def should_continue_after_enrichment(
    state: PipelineState,
) -> Literal["memorization_quality", "complete"]:
    """Determine next node after context enrichment.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "complete")

    if current_stage == "memorization_quality":
        return "memorization_quality"
    else:
        return "complete"


def should_continue_after_memorization_quality(
    state: PipelineState,
) -> Literal["duplicate_detection", "complete"]:
    """Determine next node after memorization quality.

    Args:
        state: Current pipeline state

    Returns:
        Next node name
    """
    current_stage = state.get("current_stage", "complete")

    if current_stage == "duplicate_detection":
        return "duplicate_detection"
    else:
        return "complete"


# ============================================================================
# Workflow Builder
# ============================================================================


class LangGraphOrchestrator:
    """LangGraph-based orchestrator for the card generation pipeline.

    Uses a state machine workflow with conditional routing, automatic retries,
    and state persistence via checkpoints.
    """

    def __init__(
        self,
        config: Config,
        max_retries: int | None = None,
        auto_fix_enabled: bool | None = None,
        strict_mode: bool | None = None,
        enable_card_splitting: bool | None = None,
        enable_context_enrichment: bool | None = None,
        enable_memorization_quality: bool | None = None,
        enable_duplicate_detection: bool | None = None,
    ):
        """Initialize LangGraph orchestrator.

        Args:
            config: Service configuration
            max_retries: Maximum post-validation retry attempts (uses config if None)
            auto_fix_enabled: Enable automatic error fixing (uses config if None)
            strict_mode: Use strict validation mode (uses config if None)
            enable_card_splitting: Enable card splitting agent (uses config if None)
            enable_context_enrichment: Enable context enrichment agent (uses config if None)
            enable_memorization_quality: Enable memorization quality agent (uses config if None)
            enable_duplicate_detection: Enable duplicate detection agent (uses config if None)
        """
        self.config = config
        # Use config values as defaults if not explicitly provided
        self.max_retries = (
            max_retries if max_retries is not None else config.langgraph_max_retries
        )
        self.auto_fix_enabled = (
            auto_fix_enabled
            if auto_fix_enabled is not None
            else config.langgraph_auto_fix
        )
        self.strict_mode = (
            strict_mode if strict_mode is not None else config.langgraph_strict_mode
        )
        self.enable_card_splitting = (
            enable_card_splitting
            if enable_card_splitting is not None
            # Default to True
            else getattr(config, "enable_card_splitting", True)
        )
        self.enable_context_enrichment = (
            enable_context_enrichment
            if enable_context_enrichment is not None
            else config.enable_context_enrichment
        )
        self.enable_memorization_quality = (
            enable_memorization_quality
            if enable_memorization_quality is not None
            else config.enable_memorization_quality
        )
        self.enable_duplicate_detection = (
            enable_duplicate_detection
            if enable_duplicate_detection is not None
            else getattr(
                config, "enable_duplicate_detection", False
            )  # Default to False
        )

        # Create and cache PydanticAI models once during initialization
        # This avoids recreating models (and HTTP clients) on every node execution
        from ..providers.pydantic_ai_models import (
            PydanticAIModelFactory,
        )

        try:
            self.pre_validator_model = PydanticAIModelFactory.create_from_config(
                config, model_name=config.get_model_for_agent("pre_validator")
            )
            self.card_splitting_model = PydanticAIModelFactory.create_from_config(
                config, model_name=config.get_model_for_agent("card_splitting")
            )
            self.generator_model = PydanticAIModelFactory.create_from_config(
                config, model_name=config.get_model_for_agent("generator")
            )
            self.post_validator_model = PydanticAIModelFactory.create_from_config(
                config, model_name=config.get_model_for_agent("post_validator")
            )
            self.context_enrichment_model = (
                PydanticAIModelFactory.create_from_config(
                    config, model_name=config.get_model_for_agent(
                        "context_enrichment")
                )
            )
            self.memorization_quality_model = (
                PydanticAIModelFactory.create_from_config(
                    config,
                    model_name=config.get_model_for_agent(
                        "memorization_quality"),
                )
            )
            self.duplicate_detection_model = (
                PydanticAIModelFactory.create_from_config(
                    config, model_name=config.get_model_for_agent(
                        "duplicate_detection")
                )
            )
            logger.info("pydantic_ai_models_cached", models_created=7)
        except Exception as e:
            logger.warning(
                "failed_to_cache_models_will_create_on_demand", error=str(e)
            )
            # Set to None - nodes will create models on demand as fallback
            self.pre_validator_model = None
            self.card_splitting_model = None
            self.generator_model = None
            self.post_validator_model = None
            self.context_enrichment_model = None
            self.memorization_quality_model = None
            self.duplicate_detection_model = None

        # Initialize memory store if enabled
        self.memory_store = None
        if getattr(config, "enable_agent_memory", True) and AgentMemoryStore:
            try:
                memory_storage_path = getattr(
                    config, "memory_storage_path", Path(".agent_memory"))
                enable_semantic_search = getattr(
                    config, "enable_semantic_search", True)
                embedding_model = getattr(
                    config, "embedding_model", "text-embedding-3-small")

                self.memory_store = AgentMemoryStore(
                    storage_path=memory_storage_path,
                    embedding_model=embedding_model,
                    enable_semantic_search=enable_semantic_search,
                )
                logger.info("langgraph_memory_store_initialized",
                            path=str(memory_storage_path))
            except Exception as e:
                logger.warning(
                    "langgraph_memory_store_init_failed", error=str(e))

        # Build the workflow graph
        self.workflow = self._build_workflow()

        # Initialize checkpoint saver for state persistence
        self.checkpointer = MemorySaver()

        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        logger.info(
            "langgraph_orchestrator_initialized",
            max_retries=max_retries,
            auto_fix=auto_fix_enabled,
            strict_mode=strict_mode,
            card_splitting=enable_card_splitting,
            context_enrichment=enable_context_enrichment,
            memorization_quality=enable_memorization_quality,
            duplicate_detection=enable_duplicate_detection,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Configured StateGraph instance
        """
        # Create workflow graph
        workflow = StateGraph(PipelineState)

        # Add core nodes
        workflow.add_node("pre_validation", pre_validation_node)
        workflow.add_node("card_splitting", card_splitting_node)
        workflow.add_node("generation", generation_node)
        workflow.add_node("post_validation", post_validation_node)

        # Add enhancement nodes
        workflow.add_node("context_enrichment", context_enrichment_node)
        workflow.add_node("memorization_quality", memorization_quality_node)
        workflow.add_node("duplicate_detection", duplicate_detection_node)

        # Set entry point
        workflow.set_entry_point("pre_validation")

        # Add conditional edges
        workflow.add_conditional_edges(
            "pre_validation",
            should_continue_after_pre_validation,
            {
                "card_splitting": "card_splitting",
                "generation": "generation",
                "failed": END,
            },
        )

        # Card splitting always goes to generation
        workflow.add_edge("card_splitting", "generation")

        workflow.add_edge("generation", "post_validation")

        workflow.add_conditional_edges(
            "post_validation",
            should_continue_after_post_validation,
            {
                "context_enrichment": "context_enrichment",
                "generation": "generation",  # Retry loop
                "failed": END,
            },
        )

        # Add enrichment to quality routing
        workflow.add_conditional_edges(
            "context_enrichment",
            should_continue_after_enrichment,
            {
                "memorization_quality": "memorization_quality",
                "complete": END,
            },
        )

        # Memorization quality to duplicate detection routing
        workflow.add_conditional_edges(
            "memorization_quality",
            should_continue_after_memorization_quality,
            {
                "duplicate_detection": "duplicate_detection",
                "complete": END,
            },
        )

        # Duplicate detection always goes to END
        workflow.add_edge("duplicate_detection", END)

        return workflow

    async def process_note(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
        existing_cards: list[GeneratedCard] | None = None,
    ) -> AgentPipelineResult:
        """Process a note through the LangGraph workflow.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path for validation
            existing_cards: Optional list of existing cards for duplicate detection

        Returns:
            AgentPipelineResult with all pipeline stages
        """
        import uuid

        start_time = time.time()

        logger.info(
            "langgraph_pipeline_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
        )

        # Generate slug base
        slug_base = self._generate_slug_base(metadata)

        # Initialize state
        initial_state: PipelineState = {
            "note_content": note_content,
            "metadata_dict": asdict(metadata),
            "qa_pairs_dicts": [asdict(qa) for qa in qa_pairs],
            "file_path": str(file_path) if file_path else None,
            "slug_base": slug_base,
            "config": self.config,  # Pass config for model selection
            "existing_cards_dicts": (
                [card.model_dump() for card in existing_cards]
                if existing_cards
                else None
            ),
            # Pass cached models through state for reuse
            "pre_validator_model": self.pre_validator_model,
            "card_splitting_model": self.card_splitting_model,
            "generator_model": self.generator_model,
            "post_validator_model": self.post_validator_model,
            "context_enrichment_model": self.context_enrichment_model,
            "memorization_quality_model": self.memorization_quality_model,
            "duplicate_detection_model": self.duplicate_detection_model,
            "pre_validation": None,
            "card_splitting": None,
            "generation": None,
            "post_validation": None,
            "context_enrichment": None,
            "memorization_quality": None,
            "duplicate_detection": None,
            "current_stage": "pre_validation",
            "enable_card_splitting": self.enable_card_splitting,
            "enable_context_enrichment": self.enable_context_enrichment,
            "enable_memorization_quality": self.enable_memorization_quality,
            "enable_duplicate_detection": self.enable_duplicate_detection,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "auto_fix_enabled": self.auto_fix_enabled,
            "strict_mode": self.strict_mode,
            "start_time": start_time,
            "stage_times": {},
            "messages": [],
        }

        # Execute workflow with async invocation
        # Use unique thread ID with UUID to avoid collisions
        thread_id = f"note-{metadata.title}-{uuid.uuid4().hex[:8]}"
        # LangGraph's ainvoke type checking is imperfect with TypedDict states
        final_state = await self.app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Build result
        total_time = time.time() - start_time
        success = final_state["current_stage"] == "complete"

        # Deserialize results
        pre_validation = (
            PreValidationResult(**final_state["pre_validation"])
            if final_state.get("pre_validation")
            else None
        )
        (
            CardSplittingResult(**final_state["card_splitting"])
            if final_state.get("card_splitting")
            else None
        )
        generation = (
            GenerationResult(**final_state["generation"])
            if final_state.get("generation")
            else None
        )
        post_validation = (
            PostValidationResult(**final_state["post_validation"])
            if final_state.get("post_validation")
            else None
        )
        (
            ContextEnrichmentResult(**final_state["context_enrichment"])
            if final_state.get("context_enrichment")
            else None
        )
        memorization_quality = (
            MemorizationQualityResult(**final_state["memorization_quality"])
            if final_state.get("memorization_quality")
            else None
        )

        result = AgentPipelineResult(
            success=success,
            pre_validation=pre_validation
            or PreValidationResult(
                is_valid=False, error_type="none", validation_time=0.0
            ),
            generation=generation,
            post_validation=post_validation,
            memorization_quality=memorization_quality,
            total_time=total_time,
            retry_count=final_state["retry_count"],
        )

        logger.info(
            "langgraph_pipeline_complete",
            success=success,
            retry_count=final_state["retry_count"],
            total_time=total_time,
            messages=final_state["messages"],
        )

        return result

    def _generate_slug_base(self, metadata: NoteMetadata) -> str:
        """Generate base slug from note metadata using collision-safe helper."""

        return generate_agent_slug_base(metadata)
