"""Pipeline node functions for LangGraph workflow.

This module implements all 9 pipeline nodes that process notes through
the card generation workflow:
0. Auto-Fix - Fix note issues before processing (permanent step)
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

from obsidian_anki_sync.agents.exceptions import (
    HighlightError,
    ModelError,
    PreValidationError,
    StructuredOutputError,
)
from obsidian_anki_sync.agents.models import (
    AutoFixResult,
    CardSplittingResult,
    ContextEnrichmentResult,
    GeneratedCard,
    GenerationResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.agents.pydantic import (
    CardSplittingAgentAI,
    ContextEnrichmentAgentAI,
    DuplicateDetectionAgentAI,
    GeneratorAgentAI,
    HighlightAgentAI,
    MemorizationQualityAgentAI,
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

from .node_helpers import increment_step_count
from .state import (
    PipelineState,
    get_agent_selector,
    get_config,
    get_model,
    get_rag_integration,
)

logger = get_logger(__name__)


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


# ============================================================================
# Node Functions
# ============================================================================


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
        file_path = Path(state["file_path"]) if state.get(
            "file_path") else None

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
        state["autofix"] = AutoFixResult(
            file_modified=False,
            issues_found=[],
            issues_fixed=0,
            issues_skipped=0,
            fix_time=time.time() - start_time,
        ).model_dump()
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
    from obsidian_anki_sync.agents.models import NoteCorrectionResult
    from obsidian_anki_sync.agents.parser_repair import ParserRepairAgent

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
            model_name = getattr(config, "parser_repair_model", "qwen3:8b")

        # Create provider for note correction (reuse parser repair infrastructure)
        from obsidian_anki_sync.providers.factory import ProviderFactory

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

        correction_result = await correction_agent.analyze_and_correct_proactively_async(
            content=note_content, file_path=file_path
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
    file_path = Path(state["file_path"]) if state["file_path"] else None

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
            error_details=f"Model error: {e!s}",
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
            error_details=f"Unexpected error: {e!s}",
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
    model = get_model(state, "card_splitting")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            config = get_config(state)
            model_name = config.get_model_for_agent("card_splitting")
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
    metadata = NoteMetadata(**state["metadata_dict"])
    splitting_result = CardSplittingResult(**card_splitting)

    # Use cached model or create on demand
    model = get_model(state, "split_validator")
    if model is None:
        try:
            model_name = get_config(
                state).get_model_for_agent("split_validator")
            model = create_openrouter_model_from_env(model_name=model_name)
        except Exception as e:
            logger.warning(
                "failed_to_create_split_validator_model", error=str(e))
            # Skip validation if model unavailable
            state["current_stage"] = "generation"
            return state

    # Create agent
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
    rag_integration = get_rag_integration(
        state) if state.get("enable_rag") else None

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
                logger.warning("rag_context_enrichment_failed", error=str(e))

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
                logger.warning("rag_few_shot_examples_failed", error=str(e))

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

    if agent_selector:
        # Use unified agent interface
        generator_agent = agent_selector.get_agent(
            agent_framework, "generator")
        logger.info(
            "using_unified_agent", framework=agent_framework, agent_type="generator"
        )
    else:
        # Fallback to direct PydanticAI agent (legacy behavior)
        logger.warning("agent_selector_not_available_fallback_to_pydantic_ai")
        generator = GeneratorAgentAI(model=model, temperature=0.3)

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
                qa_pairs[i: i + BATCH_SIZE]
                for i in range(0, len(qa_pairs), BATCH_SIZE)
            ]

            # Create tasks for each chunk
            tasks = []
            for chunk in chunks:
                if agent_selector:
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
                else:
                    # Legacy PydanticAI agent (with RAG context)
                    tasks.append(
                        generator.generate_cards(
                            note_content=state["note_content"],
                            metadata=metadata,
                            qa_pairs=chunk,
                            slug_base=state["slug_base"],
                            rag_enrichment=rag_enrichment,
                            rag_examples=rag_examples,
                        )
                    )

            # Run tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            all_cards = []
            total_gen_time = 0.0
            all_warnings = []

            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error(
                        "langgraph_generation_chunk_failed",
                        chunk_index=i,
                        error=str(res),
                    )
                    # We continue with partial results if some chunks fail
                # Handle different result types
                elif hasattr(res, "data") and hasattr(res.data, "cards"):
                    # UnifiedAgentResult with GenerationResult data
                    all_cards.extend(res.data.cards)
                    total_gen_time = max(
                        total_gen_time, getattr(res.data, "generation_time", 0)
                    )
                    if hasattr(res, "warnings") and res.warnings:
                        all_warnings.extend(res.warnings)
                else:
                    # Legacy GenerationResult
                    all_cards.extend(res.cards)
                    total_gen_time = max(total_gen_time, res.generation_time)
                    if hasattr(res, "warnings") and res.warnings:
                        all_warnings.extend(res.warnings)

            # Create merged result
            gen_result = GenerationResult(
                cards=all_cards,
                total_cards=len(all_cards),
                generation_time=time.time() - start_time,  # Total wall time
                model_used=str(model),
            )

        else:
            # Sequential generation (single batch)
            if agent_selector:
                # Use unified agent interface
                qa_dicts = [qa.model_dump() for qa in qa_pairs]
                unified_result = await generator_agent.generate_cards(
                    note_content=state["note_content"],
                    metadata=metadata.model_dump(),
                    qa_pairs=qa_dicts,
                    slug_base=state["slug_base"],
                )
                # Convert unified result to GenerationResult
                gen_result = unified_result.data
                # Add warnings from unified result
                if hasattr(gen_result, "warnings") and unified_result.warnings:
                    if (
                        not hasattr(gen_result, "warnings")
                        or gen_result.warnings is None
                    ):
                        gen_result.warnings = []
                    gen_result.warnings.extend(unified_result.warnings)
            else:
                # Legacy PydanticAI agent (with RAG context)
                gen_result = await generator.generate_cards(
                    note_content=state["note_content"],
                    metadata=metadata,
                    qa_pairs=qa_pairs,
                    slug_base=state["slug_base"],
                    rag_enrichment=rag_enrichment,
                    rag_examples=rag_examples,
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
        "linter_validation" if gen_result.total_cards > 0 else "failed"
    )
    state["messages"].append(f"Generated {gen_result.total_cards} cards")

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

    for card in cards:
        try:
            result = validate_apf(card.apf_html, card.slug)
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
    model = get_model(state, "post_validator")
    if model is None:
        try:
            # Fallback: create model on demand if not cached
            model_name = get_config(
                state).get_model_for_agent("post_validator")
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
    logger.info(
        "langgraph_post_validation_complete",
        is_valid=post_result.is_valid,
        retry_count=state["retry_count"],
        time=post_result.validation_time,
        linter_valid=linter_valid,
        llm_error_type=post_result.error_type if not post_result.is_valid else "none",
        llm_template_overridden=llm_template_overridden,
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
    elif (state.get("retry_count") or 0) < (
        state.get("max_retries") or 3
    ) and state.get("auto_fix_enabled", False):
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
        model = get_model(state, "context_enrichment")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = get_config(state).get_model_for_agent(
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
        model = get_model(state, "memorization_quality")
        if model is None:
            # Fallback: create model on demand if not cached
            model_name = get_config(state).get_model_for_agent(
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
            state["messages"].append(
                f"Memorization quality: score={mem_score:.2f}")

    return state


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
        logger.info("rag_duplicate_detection_start",
                    cards_count=len(new_cards))

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
        logger.info("duplicate_detection_skipped",
                    reason="no_existing_cards_for_llm")
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
            detection_agent = DuplicateDetectionAgentAI(
                model=model, temperature=0.0)

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
            state["messages"].append(
                "Highlight agent unavailable (model missing)")
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

    highlight_time = highlight_result.analysis_time or (
        time.time() - start_time)
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
