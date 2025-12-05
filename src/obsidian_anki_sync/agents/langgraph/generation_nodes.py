"""Generation pipeline node functions for LangGraph workflow.

This module implements nodes responsible for generating Anki cards from notes:
- Card splitting: Determines optimal card count and splitting strategy
- Generation: Core card generation using LLMs with parallel processing
"""

import asyncio
import time

from obsidian_anki_sync.agents.models import (
    CardSplittingResult,
    GenerationResult,
)
from obsidian_anki_sync.agents.pydantic import CardSplittingAgentAI
from obsidian_anki_sync.error_codes import ErrorCode
from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.providers.pydantic_ai_models import (
    create_openrouter_model_from_env,
)
from obsidian_anki_sync.utils.logging import get_logger, log_state_transition

from .node_helpers import increment_step_count
from .state import PipelineState, get_agent_selector, get_config, get_model, get_rag_integration

logger = get_logger(__name__)


def _validate_generation_inputs(state: PipelineState) -> list[str]:
    """Validate inputs before generation to catch issues early.

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not state.get("note_content"):
        errors.append("note_content is empty or None")

    if not state.get("qa_pairs_dicts"):
        errors.append("qa_pairs_dicts is empty or None")

    metadata = state.get("metadata_dict", {})
    if not metadata:
        errors.append("metadata_dict is empty or None")
    elif not metadata.get("title"):
        errors.append("metadata.title is missing")

    if not state.get("slug_base"):
        errors.append("slug_base is empty or None")

    return errors


def _get_reasoning_recommendations(state: PipelineState) -> list[str]:
    """Get recommendations from the current reasoning trace if available.

    This helper allows action nodes to consume recommendations from the
    preceding reasoning node (if CoT is enabled).

    Args:
        state: Current pipeline state

    Returns:
        List of recommendations from reasoning, empty if none available
    """
    current_reasoning = state.get("current_reasoning")
    if not current_reasoning:
        return []

    recommendations = current_reasoning.get("recommendations", [])
    if recommendations:
        logger.debug(
            "cot_recommendations_available",
            count=len(recommendations),
            stage=current_reasoning.get("stage", "unknown"),
        )
    return recommendations  # type: ignore[no-any-return]


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
    model = get_model(state, "card_splitting")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            config = get_config(state)
            model_name = config.get_model_for_agent("card_splitting")
            model = create_openrouter_model_from_env(model_name=model_name)
        except (ValueError, KeyError) as e:
            logger.warning("failed_to_create_card_splitting_model", error=str(e))
            # Fallback: skip card splitting analysis
            state["card_splitting"] = None
            state["current_stage"] = "generation"
            state["stage_times"]["card_splitting"] = time.time() - start_time
            state["messages"].append("Card splitting skipped (model unavailable)")
            return state

    # Create card splitting agent
    splitting_agent = CardSplittingAgentAI(model=model, temperature=0.0)

    # Get splitting preferences from config
    config = get_config(state)
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
            logger.debug("card_splitting_preferred_size_small_encouraging_split")
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
    Supports parallel generation for better performance with many Q/A pairs.

    If Chain of Thought (CoT) reasoning is enabled, this node will consume
    recommendations from the preceding think_before_generation_node.

    If RAG is enabled, this node will:
    - Enrich context with related concepts from the knowledge base
    - Retrieve few-shot examples for improved generation quality

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

    # Log state transition if enabled
    config = get_config(state)
    if getattr(config, "log_state_transitions", True):
        log_state_transition(
            logger,
            pipeline_id=state.get("pipeline_id"),
            from_stage=state.get("current_stage", "unknown"),
            to_stage="generation",
            reason="starting generation",
        )

    # Input validation - catch issues early with clear error messages
    validation_errors = _validate_generation_inputs(state)
    if validation_errors:
        logger.error(
            "generation_input_validation_failed",
            errors=validation_errors,
            error_code=ErrorCode.GEN_INPUT_INVALID.value,
        )
        gen_result = GenerationResult(
            cards=[],
            total_cards=0,
            generation_time=time.time() - start_time,
            model_used="none",
            error_code=ErrorCode.GEN_INPUT_INVALID.value,
        )
        state["generation"] = gen_result.model_dump()
        state["stage_times"]["generation"] = gen_result.generation_time
        state["current_stage"] = "failed"
        state["last_error"] = f"Input validation failed: {'; '.join(validation_errors)}"
        state["messages"].append(f"Generation failed: {'; '.join(validation_errors)}")
        return state

    # Get CoT reasoning recommendations if available
    reasoning_recommendations = _get_reasoning_recommendations(state)
    if reasoning_recommendations:
        logger.info(
            "cot_generation_using_recommendations",
            recommendations_count=len(reasoning_recommendations),
        )
        # Log a summary of the recommendations
        # Log first 3
        for i, rec in enumerate(reasoning_recommendations[:3], 1):
            logger.debug(f"cot_recommendation_{i}", recommendation=rec[:100])

    # RAG: Enrich context and get few-shot examples before generation
    rag_enrichment = None
    rag_examples = None
    rag_integration = get_rag_integration(state) if state.get("enable_rag") else None

    if state.get("enable_rag") and rag_integration:
        metadata_dict = state["metadata_dict"]

        # Get context enrichment
        if state.get("rag_context_enrichment", True):
            try:
                rag_enrichment = await rag_integration.enrich_generation_context(
                    note_content=state["note_content"],
                    metadata=metadata_dict,
                )
                if rag_enrichment:
                    state["rag_enrichment"] = rag_enrichment
                    logger.info(
                        "rag_context_enrichment_complete",
                        related_concepts=len(
                            rag_enrichment.get("related_concepts", [])
                        ),
                        few_shot_examples=len(
                            rag_enrichment.get("few_shot_examples", [])
                        ),
                    )
            except Exception as e:
                # Upgrade to error level - RAG failures affect generation quality
                logger.error(
                    "rag_context_enrichment_failed",
                    error=str(e),
                    error_code=ErrorCode.PRV_RAG_FAILED.value,
                    recoverable=True,
                )

        # Get few-shot examples
        if state.get("rag_few_shot_examples", True):
            try:
                topic = metadata_dict.get("topic", "")
                difficulty = metadata_dict.get("difficulty")
                if topic:
                    rag_examples = await rag_integration.get_examples_for_generation(
                        topic=topic,
                        difficulty=difficulty,
                    )
                    if rag_examples:
                        state["rag_examples"] = rag_examples
                        logger.info(
                            "rag_few_shot_examples_retrieved",
                            count=len(rag_examples),
                            topic=topic,
                        )
            except Exception as e:
                # Upgrade to error level - RAG failures affect generation quality
                logger.error(
                    "rag_few_shot_examples_failed",
                    error=str(e),
                    error_code=ErrorCode.PRV_RAG_FAILED.value,
                    recoverable=True,
                )

    # Deserialize data
    metadata = NoteMetadata(**state["metadata_dict"])
    qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

    config = get_config(state)

    # Use cached model from state, or create on demand as fallback
    model = get_model(state, "generator")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = config.get_model_for_agent("generator")
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

    # NEW: Use unified agent interface for framework switching
    agent_framework = state.get("agent_framework", "pydantic_ai")
    agent_selector = get_agent_selector(state)

    if not agent_selector:
        logger.error("agent_selector_missing_in_state")
        state["current_stage"] = "failed"
        state["messages"].append("Generation failed: Agent selector missing")
        return state

    # Use unified agent interface
    generator_agent = agent_selector.get_agent(agent_framework, "generator")
    logger.info(
        "using_unified_agent", framework=agent_framework, agent_type="generator"
    )

    # Run generation
    try:
        # Determine if we should parallelize
        # Default batch size for parallel generation
        BATCH_SIZE = getattr(config, "generation_batch_size", 5)

        if len(qa_pairs) > BATCH_SIZE:
            # Parallel generation
            logger.info(
                "langgraph_generation_parallel",
                total_pairs=len(qa_pairs),
                batch_size=BATCH_SIZE,
            )

            # Split Q&A pairs into chunks
            chunks = [
                qa_pairs[i : i + BATCH_SIZE]
                for i in range(0, len(qa_pairs), BATCH_SIZE)
            ]

            # Create tasks for each chunk
            tasks = []
            for chunk in chunks:
                # Use unified agent interface
                qa_dicts = [qa.model_dump() for qa in chunk]
                tasks.append(
                    generator_agent.generate_cards(
                        note_content=state["note_content"],
                        metadata=metadata.model_dump(),
                        qa_pairs=qa_dicts,
                        slug_base=state["slug_base"],
                    )
                )

            # Run tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            all_cards = []
            total_gen_time = 0.0
            all_warnings = []
            failed_chunks = 0
            total_chunks = len(chunks)

            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    failed_chunks += 1
                    logger.error(
                        "langgraph_generation_chunk_failed",
                        chunk_index=i,
                        total_chunks=total_chunks,
                        error=str(res),
                        error_code=ErrorCode.GEN_CHUNK_FAILED.value,
                    )
                    all_warnings.append(f"Chunk {i} failed: {str(res)[:100]}")
                # Handle different result types
                elif hasattr(res, "data") and hasattr(res.data, "cards"):
                    # UnifiedAgentResult with GenerationResult data
                    all_cards.extend(res.data.cards)
                    total_gen_time = max(
                        total_gen_time, getattr(res.data, "generation_time", 0)
                    )
                    if hasattr(res, "warnings") and res.warnings:
                        all_warnings.extend(res.warnings)

            # Fail the entire note if any chunks failed (per user preference)
            if failed_chunks > 0:
                logger.error(
                    "generation_partial_failure",
                    failed_chunks=failed_chunks,
                    total_chunks=total_chunks,
                    error_code=ErrorCode.GEN_PARTIAL_RESULT.value,
                )
                gen_result = GenerationResult(
                    cards=[],
                    total_cards=0,
                    generation_time=time.time() - start_time,
                    model_used=str(model),
                    is_partial=True,
                    failed_chunk_count=failed_chunks,
                    total_chunk_count=total_chunks,
                    error_code=ErrorCode.GEN_PARTIAL_RESULT.value,
                    warnings=all_warnings,
                )
                state["generation"] = gen_result.model_dump()
                state["stage_times"]["generation"] = gen_result.generation_time
                state["current_stage"] = "failed"
                state["last_error"] = (
                    f"Partial generation failure: {failed_chunks}/{total_chunks} "
                    "chunks failed"
                )
                state["messages"].append(
                    f"Generation failed: {failed_chunks}/{total_chunks} chunks failed"
                )
                return state

            # Create merged result (all chunks succeeded)
            gen_result = GenerationResult(
                cards=all_cards,
                total_cards=len(all_cards),
                generation_time=time.time() - start_time,  # Total wall time
                model_used=str(model),
                total_chunk_count=total_chunks,
                warnings=all_warnings,
            )

        # Sequential generation (single batch)
        else:
            # Use unified agent interface
            qa_dicts = [qa.model_dump() for qa in qa_pairs]
            unified_result = await generator_agent.generate_cards(
                note_content=state["note_content"],
                metadata=metadata.model_dump(),
                qa_pairs=qa_dicts,
                slug_base=state["slug_base"],
                source_path=state.get("file_path"),
            )
            # Convert unified result to GenerationResult
            gen_result = unified_result.data
            # Handle case where generation failed and data is None
            if gen_result is None:
                logger.error(
                    "langgraph_generation_returned_none",
                    success=unified_result.success,
                    reasoning=unified_result.reasoning,
                )
                # Create empty result to allow graceful continuation
                gen_result = GenerationResult(
                    cards=[],
                    total_cards=0,
                    generation_time=time.time() - start_time,
                    model_used=str(model),
                )
            else:
                # Add warnings from unified result
                if unified_result.warnings:
                    if (
                        not hasattr(gen_result, "warnings")
                        or gen_result.warnings is None
                    ):
                        gen_result.warnings = []
                    gen_result.warnings.extend(unified_result.warnings)
                gen_result.generation_time = time.time() - start_time

        # Validate against split plan if available
        if expected_card_count is not None:
            actual_count = gen_result.total_cards

            # Adjust expected count for bilingual notes
            # Each concept (card in split plan) generates one card per language
            num_languages = len(metadata.language_tags) if metadata.language_tags else 1
            adjusted_expected_count = expected_card_count * num_languages

            if actual_count != adjusted_expected_count:
                logger.warning(
                    "langgraph_generation_card_count_mismatch",
                    expected=adjusted_expected_count,
                    actual=actual_count,
                    strategy=(
                        splitting_result.splitting_strategy if card_splitting else None
                    ),
                    num_languages=num_languages,
                    original_expected=expected_card_count,
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
        "linter_validation" if gen_result.total_cards > 0 else "failed"
    )
    state["messages"].append(f"Generated {gen_result.total_cards} cards")

    return state
