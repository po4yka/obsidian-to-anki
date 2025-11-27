"""Chain of Thought (CoT) reasoning nodes for LangGraph pipeline.

This module implements reasoning nodes that perform explicit thinking BEFORE
each action node. Reasoning nodes analyze the current state and provide
recommendations that are passed to action nodes via state["current_reasoning"].

Error handling: On failure, log warning and continue to action node.
"""

import time
from typing import Any

from pydantic_ai import Agent

from ...models import NoteMetadata, QAPair
from ...utils.logging import get_logger
from .node_helpers import increment_step_count
from .reasoning_models import (
    CardSplittingReasoningOutput,
    DuplicateReasoningOutput,
    EnrichmentReasoningOutput,
    GenerationReasoningOutput,
    MemorizationReasoningOutput,
    PostValidationReasoningOutput,
    PreValidationReasoningOutput,
    ReasoningTrace,
)
from .reasoning_prompts import (
    CARD_SPLITTING_REASONING_PROMPT,
    DUPLICATE_REASONING_PROMPT,
    ENRICHMENT_REASONING_PROMPT,
    GENERATION_REASONING_PROMPT,
    MEMORIZATION_REASONING_PROMPT,
    POST_VALIDATION_REASONING_PROMPT,
    PRE_VALIDATION_REASONING_PROMPT,
)
from .state import PipelineState

logger = get_logger(__name__)


def _should_skip_reasoning(state: PipelineState, stage: str) -> bool:
    """Check if reasoning should be skipped for this stage.

    Args:
        state: Current pipeline state
        stage: Stage name to check

    Returns:
        True if reasoning should be skipped
    """
    # Master toggle
    if not state.get("enable_cot_reasoning", False):
        return True

    # Stage-specific toggle
    enabled_stages = state.get("cot_enabled_stages", [])
    if enabled_stages and stage not in enabled_stages:
        return True

    return False


def _store_reasoning_trace(
    state: PipelineState,
    stage: str,
    output: Any,
    reasoning_time: float,
    stage_specific_data: dict | None = None,
) -> None:
    """Store reasoning trace in state if configured.

    Args:
        state: Current pipeline state
        stage: Stage name
        output: Reasoning output from agent
        reasoning_time: Time taken for reasoning
        stage_specific_data: Additional stage-specific data
    """
    if not state.get("store_reasoning_traces", True):
        return

    # Initialize reasoning_traces if needed
    if "reasoning_traces" not in state or state["reasoning_traces"] is None:
        state["reasoning_traces"] = {}

    # Create trace from output
    trace = ReasoningTrace.from_output(
        stage=stage,
        output=output,
        reasoning_time=reasoning_time,
        timestamp=time.time(),
        stage_specific_data=stage_specific_data,
    )

    # Store trace
    state["reasoning_traces"][stage] = trace.model_dump()

    # Set current_reasoning for action node to consume
    state["current_reasoning"] = trace.model_dump()

    # Log if configured
    if state.get("log_reasoning_traces", False):
        reasoning_preview = (
            output.reasoning[:200] + "..."
            if len(output.reasoning) > 200
            else output.reasoning
        )
        logger.info(
            f"cot_reasoning_{stage}",
            reasoning_preview=reasoning_preview,
            key_observations=output.key_observations,
            confidence=output.confidence,
            reasoning_time=reasoning_time,
        )


async def think_before_pre_validation_node(state: PipelineState) -> PipelineState:
    """Reason about pre-validation before executing it.

    Analyzes note structure, frontmatter, and content quality to provide
    recommendations for the pre-validation stage.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "pre_validation"

    if _should_skip_reasoning(state, stage):
        return state

    # Check step limit (cycle protection)
    if not increment_step_count(state, "think_before_pre_validation"):
        return state

    logger.info("cot_think_before_pre_validation_start")
    start_time = time.time()

    # Get reasoning model from state
    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        # Create reasoning agent
        agent: Agent[None, PreValidationReasoningOutput] = Agent(
            model=model,
            result_type=PreValidationReasoningOutput,
            system_prompt=PRE_VALIDATION_REASONING_PROMPT,
        )

        # Build context for reasoning
        metadata = NoteMetadata(**state["metadata_dict"])
        qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

        # Create prompt with note context
        note_preview = state["note_content"][:1500]
        if len(state["note_content"]) > 1500:
            note_preview += "..."

        prompt = f"""Analyze this note before pre-validation:

Title: {metadata.title}
Topic: {metadata.topic}
Tags: {', '.join(metadata.tags)}
Language Tags: {', '.join(metadata.language_tags)}
Q&A Pairs Count: {len(qa_pairs)}

Note Content:
{note_preview}

Think through: What structural issues, frontmatter problems, or content quality concerns exist?
What should the pre-validator focus on?"""

        # Run reasoning
        result = await agent.run(prompt)
        output = result.data

        # Store trace with stage-specific data
        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "structure_assessment": output.structure_assessment,
                "frontmatter_assessment": output.frontmatter_assessment,
                "content_quality_assessment": output.content_quality_assessment,
                "validation_focus": output.validation_focus,
            },
        )

        state["stage_times"]["think_before_pre_validation"] = time.time() - start_time
        state["messages"].append(
            f"CoT pre-validation reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        # Log and continue - reasoning failure should not block pipeline
        logger.warning("cot_pre_validation_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_generation_node(state: PipelineState) -> PipelineState:
    """Reason about card generation before executing it.

    Analyzes Q&A pairs and content complexity to provide recommendations
    for the generation stage.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "generation"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_generation"):
        return state

    logger.info("cot_think_before_generation_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, GenerationReasoningOutput] = Agent(
            model=model,
            result_type=GenerationReasoningOutput,
            system_prompt=GENERATION_REASONING_PROMPT,
        )

        metadata = NoteMetadata(**state["metadata_dict"])
        qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

        # Format Q&A pairs for analysis
        qa_summary = []
        for i, qa in enumerate(qa_pairs[:5], 1):  # Limit to first 5
            qa_summary.append(f"  Q{i}: {qa.question[:100]}...")
            qa_summary.append(f"  A{i}: {qa.answer[:100]}...")
        qa_text = "\n".join(qa_summary)
        if len(qa_pairs) > 5:
            qa_text += f"\n  ... and {len(qa_pairs) - 5} more Q&A pairs"

        prompt = f"""Analyze this content before card generation:

Title: {metadata.title}
Topic: {metadata.topic}
Total Q&A Pairs: {len(qa_pairs)}

Q&A Pairs Preview:
{qa_text}

Think through: What card types work best? How complex is the content?
What formatting considerations are needed?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "card_type_recommendation": output.card_type_recommendation,
                "complexity_assessment": output.complexity_assessment,
                "formatting_recommendations": output.formatting_recommendations,
            },
        )

        state["stage_times"]["think_before_generation"] = time.time() - start_time
        state["messages"].append(
            f"CoT generation reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_generation_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_post_validation_node(state: PipelineState) -> PipelineState:
    """Reason about post-validation before executing it.

    Analyzes generated cards to identify potential quality issues and
    guide the validation focus.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "post_validation"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_post_validation"):
        return state

    logger.info("cot_think_before_post_validation_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, PostValidationReasoningOutput] = Agent(
            model=model,
            result_type=PostValidationReasoningOutput,
            system_prompt=POST_VALIDATION_REASONING_PROMPT,
        )

        # Get generation results
        generation = state.get("generation")
        if not generation:
            logger.warning("cot_post_validation_no_generation")
            return state

        cards = generation.get("cards", [])
        linter_results = state.get("linter_results", [])

        # Summarize cards and linter results
        card_summary = f"Generated {len(cards)} cards"
        if linter_results:
            errors = sum(1 for r in linter_results if r.get("errors"))
            warnings = sum(1 for r in linter_results if r.get("warnings"))
            card_summary += f" (linter: {errors} errors, {warnings} warnings)"

        prompt = f"""Analyze generated cards before post-validation:

{card_summary}
Retry Count: {state.get('retry_count', 0)}
Auto-fix Enabled: {state.get('auto_fix_enabled', False)}
Strict Mode: {state.get('strict_mode', True)}

Think through: What quality issues might exist? What validation strategy?
What issues are expected based on linting?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "quality_concerns": output.quality_concerns,
                "validation_strategy": output.validation_strategy,
                "expected_issues": output.expected_issues,
            },
        )

        state["stage_times"]["think_before_post_validation"] = time.time() - start_time
        state["messages"].append(
            f"CoT post-validation reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_post_validation_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_card_splitting_node(state: PipelineState) -> PipelineState:
    """Reason about card splitting before executing it.

    Analyzes content complexity to guide the splitting decision.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "card_splitting"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_card_splitting"):
        return state

    logger.info("cot_think_before_card_splitting_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, CardSplittingReasoningOutput] = Agent(
            model=model,
            result_type=CardSplittingReasoningOutput,
            system_prompt=CARD_SPLITTING_REASONING_PROMPT,
        )

        metadata = NoteMetadata(**state["metadata_dict"])
        qa_pairs = [QAPair(**qa_dict) for qa_dict in state["qa_pairs_dicts"]]

        note_preview = state["note_content"][:1000]

        prompt = f"""Analyze this content for card splitting decision:

Title: {metadata.title}
Topic: {metadata.topic}
Q&A Pairs: {len(qa_pairs)}

Content Preview:
{note_preview}

Think through: Should this be split into multiple cards?
What are the concept boundaries? What are the trade-offs?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "complexity_indicators": output.complexity_indicators,
                "split_recommendation": output.split_recommendation,
                "concept_boundaries": output.concept_boundaries,
            },
        )

        state["stage_times"]["think_before_card_splitting"] = time.time() - start_time
        state["messages"].append(
            f"CoT card-splitting reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_card_splitting_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_enrichment_node(state: PipelineState) -> PipelineState:
    """Reason about context enrichment before executing it.

    Identifies opportunities for adding examples, mnemonics, and context.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "context_enrichment"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_enrichment"):
        return state

    logger.info("cot_think_before_enrichment_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, EnrichmentReasoningOutput] = Agent(
            model=model,
            result_type=EnrichmentReasoningOutput,
            system_prompt=ENRICHMENT_REASONING_PROMPT,
        )

        metadata = NoteMetadata(**state["metadata_dict"])
        generation = state.get("generation", {})
        cards_count = len(generation.get("cards", []))

        prompt = f"""Analyze cards for context enrichment:

Title: {metadata.title}
Topic: {metadata.topic}
Tags: {', '.join(metadata.tags)}
Cards Count: {cards_count}

Think through: What enrichment opportunities exist?
What examples, mnemonics, or context would help learning?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "enrichment_opportunities": output.enrichment_opportunities,
                "mnemonic_suggestions": output.mnemonic_suggestions,
                "example_types": output.example_types,
            },
        )

        state["stage_times"]["think_before_enrichment"] = time.time() - start_time
        state["messages"].append(
            f"CoT enrichment reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_enrichment_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_memorization_node(state: PipelineState) -> PipelineState:
    """Reason about memorization quality before checking it.

    Analyzes factors affecting retention and SRS effectiveness.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "memorization_quality"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_memorization"):
        return state

    logger.info("cot_think_before_memorization_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, MemorizationReasoningOutput] = Agent(
            model=model,
            result_type=MemorizationReasoningOutput,
            system_prompt=MEMORIZATION_REASONING_PROMPT,
        )

        metadata = NoteMetadata(**state["metadata_dict"])
        generation = state.get("generation", {})
        cards_count = len(generation.get("cards", []))

        prompt = f"""Analyze cards for memorization quality:

Title: {metadata.title}
Topic: {metadata.topic}
Cards Count: {cards_count}

Think through: What factors affect retention?
What is the cognitive load? Will these cards work with SRS?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "retention_factors": output.retention_factors,
                "cognitive_load_assessment": output.cognitive_load_assessment,
            },
        )

        state["stage_times"]["think_before_memorization"] = time.time() - start_time
        state["messages"].append(
            f"CoT memorization reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_memorization_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state


async def think_before_duplicate_node(state: PipelineState) -> PipelineState:
    """Reason about duplicate detection before executing it.

    Identifies potential similarity indicators and comparison strategy.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reasoning trace
    """
    stage = "duplicate_detection"

    if _should_skip_reasoning(state, stage):
        return state

    if not increment_step_count(state, "think_before_duplicate"):
        return state

    logger.info("cot_think_before_duplicate_start")
    start_time = time.time()

    model = state.get("reasoning_model")
    if model is None:
        logger.warning("cot_reasoning_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, DuplicateReasoningOutput] = Agent(
            model=model,
            result_type=DuplicateReasoningOutput,
            system_prompt=DUPLICATE_REASONING_PROMPT,
        )

        metadata = NoteMetadata(**state["metadata_dict"])
        generation = state.get("generation", {})
        new_cards_count = len(generation.get("cards", []))
        existing_cards = state.get("existing_cards_dicts", [])
        existing_count = len(existing_cards) if existing_cards else 0

        prompt = f"""Analyze for duplicate detection:

Title: {metadata.title}
Topic: {metadata.topic}
New Cards: {new_cards_count}
Existing Cards to Compare: {existing_count}

Think through: What similarity indicators exist?
What comparison strategy should be used?"""

        result = await agent.run(prompt)
        output = result.data

        _store_reasoning_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "similarity_indicators": output.similarity_indicators,
                "comparison_strategy": output.comparison_strategy,
            },
        )

        state["stage_times"]["think_before_duplicate"] = time.time() - start_time
        state["messages"].append(
            f"CoT duplicate reasoning: confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("cot_duplicate_reasoning_failed", error=str(e))
        state["current_reasoning"] = None

    return state
