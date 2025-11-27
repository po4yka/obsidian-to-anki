"""Self-Reflection nodes for LangGraph pipeline.

This module implements reflection nodes that evaluate outputs AFTER action nodes
execute and determine if revision is needed. Reflection nodes have access to
the original CoT reasoning traces to check if the plan was followed.

Self-Reflection pattern:
1. Action node produces output (e.g., generation, enrichment)
2. Validation passes
3. Reflection node evaluates the output
4. If revision_needed and revision_count < max_revisions:
   - Revision node improves the output
   - Loop back to action node
5. Continue to next stage

Error handling: On failure, log warning and continue (non-blocking).
"""

import time
from typing import Any

from pydantic_ai import Agent

from ...utils.logging import get_logger
from .node_helpers import increment_step_count
from .reflection_models import (
    EnrichmentReflectionOutput,
    GenerationReflectionOutput,
    ReflectionTrace,
    RevisionOutput,
)
from .reflection_prompts import (
    ENRICHMENT_REFLECTION_PROMPT,
    GENERATION_REFLECTION_PROMPT,
    REVISION_PROMPT_ENRICHMENT,
    REVISION_PROMPT_GENERATION,
)
from .state import PipelineState

logger = get_logger(__name__)


def _should_skip_reflection(state: PipelineState, stage: str) -> bool:
    """Check if reflection should be skipped for this stage.

    Args:
        state: Current pipeline state
        stage: Stage name to check

    Returns:
        True if reflection should be skipped
    """
    # Master toggle
    if not state.get("enable_self_reflection", False):
        return True

    # Stage-specific toggle
    enabled_stages = state.get("reflection_enabled_stages", [])
    if enabled_stages and stage not in enabled_stages:
        return True

    return False


def _store_reflection_trace(
    state: PipelineState,
    stage: str,
    output: Any,
    reflection_time: float,
    stage_specific_data: dict | None = None,
) -> None:
    """Store reflection trace in state if configured.

    Args:
        state: Current pipeline state
        stage: Stage name
        output: Reflection output from agent
        reflection_time: Time taken for reflection
        stage_specific_data: Additional stage-specific data
    """
    if not state.get("store_reflection_traces", True):
        return

    # Initialize reflection_traces if needed
    if "reflection_traces" not in state or state["reflection_traces"] is None:
        state["reflection_traces"] = {}

    # Create trace from output
    trace = ReflectionTrace.from_output(
        stage=stage,
        output=output,
        reflection_time=reflection_time,
        timestamp=time.time(),
        stage_specific_data=stage_specific_data,
    )

    # Store trace
    state["reflection_traces"][stage] = trace.model_dump()

    # Set current_reflection for revision node to consume
    state["current_reflection"] = trace.model_dump()

    # Log if configured
    if state.get("log_reflection_traces", False):
        reflection_preview = (
            output.reflection[:200] + "..."
            if len(output.reflection) > 200
            else output.reflection
        )
        logger.info(
            f"self_reflection_{stage}",
            reflection_preview=reflection_preview,
            issues_found=output.issues_found[:3],  # Limit log output
            revision_needed=output.revision_needed,
            confidence=output.confidence,
            reflection_time=reflection_time,
        )


def _can_revise(state: PipelineState, stage: str) -> bool:
    """Check if revision is allowed for this stage.

    Args:
        state: Current pipeline state
        stage: Stage name

    Returns:
        True if revision is allowed
    """
    max_revisions = state.get("max_revisions", 2)
    if max_revisions <= 0:
        return False

    # Get stage-specific revision count
    stage_revision_counts = state.get("stage_revision_counts", {})
    current_count = stage_revision_counts.get(stage, 0)

    return current_count < max_revisions


def _increment_revision_count(state: PipelineState, stage: str) -> None:
    """Increment revision count for a stage.

    Args:
        state: Current pipeline state
        stage: Stage name
    """
    if "stage_revision_counts" not in state or state["stage_revision_counts"] is None:
        state["stage_revision_counts"] = {}

    current_count = state["stage_revision_counts"].get(stage, 0)
    state["stage_revision_counts"][stage] = current_count + 1

    # Also increment total revision count
    state["revision_count"] = state.get("revision_count", 0) + 1


def _get_cot_reasoning_for_stage(state: PipelineState, stage: str) -> dict | None:
    """Get the CoT reasoning trace for a stage (if available).

    Args:
        state: Current pipeline state
        stage: Stage name (e.g., "generation", "context_enrichment")

    Returns:
        CoT reasoning trace dict or None
    """
    reasoning_traces = state.get("reasoning_traces", {})
    return reasoning_traces.get(stage)


async def reflect_after_generation_node(state: PipelineState) -> PipelineState:
    """Reflect on generated cards after post-validation passes.

    Evaluates card quality, format compliance, and CoT plan adherence.
    Determines if revision is needed.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reflection trace and revision decision
    """
    stage = "generation"

    if _should_skip_reflection(state, stage):
        return state

    # Check step limit (cycle protection)
    if not increment_step_count(state, "reflect_after_generation"):
        return state

    logger.info("self_reflection_after_generation_start")
    start_time = time.time()

    # Get reflection model from state
    model = state.get("reflection_model")
    if model is None:
        logger.warning("self_reflection_model_not_available", stage=stage)
        return state

    try:
        # Create reflection agent
        agent: Agent[None, GenerationReflectionOutput] = Agent(
            model=model,
            result_type=GenerationReflectionOutput,
            system_prompt=GENERATION_REFLECTION_PROMPT,
        )

        # Get generation results and validation info
        generation = state.get("generation")
        if not generation:
            logger.warning("self_reflection_no_generation_output")
            return state

        post_validation = state.get("post_validation")
        linter_results = state.get("linter_results", [])

        # Get CoT reasoning for context
        cot_reasoning = _get_cot_reasoning_for_stage(state, stage)

        # Build context for reflection
        cards = generation.get("cards", [])
        card_summary = []
        for i, card in enumerate(cards[:5], 1):  # Limit to first 5
            card_summary.append(f"Card {i}:")
            card_summary.append(f"  Type: {card.get('card_type', 'unknown')}")
            card_summary.append(f"  Question preview: {str(card.get('question', ''))[:100]}...")
            card_summary.append(f"  Answer preview: {str(card.get('answer', ''))[:100]}...")
        card_text = "\n".join(card_summary)

        if len(cards) > 5:
            card_text += f"\n... and {len(cards) - 5} more cards"

        # Summarize validation
        validation_summary = "Post-validation: "
        if post_validation:
            validation_summary += f"valid={post_validation.get('is_valid', False)}"
            if post_validation.get("issues"):
                validation_summary += f", issues={len(post_validation['issues'])}"

        if linter_results:
            errors = sum(1 for r in linter_results if r.get("errors"))
            warnings = sum(1 for r in linter_results if r.get("warnings"))
            validation_summary += f"\nLinter: {errors} errors, {warnings} warnings"

        # Include CoT context if available
        cot_context = ""
        if cot_reasoning:
            cot_context = f"""

Original CoT Reasoning Plan:
- Planned approach: {cot_reasoning.get('planned_approach', 'N/A')}
- Recommendations: {', '.join(cot_reasoning.get('recommendations', [])[:5])}
- Potential issues identified: {', '.join(cot_reasoning.get('potential_issues', [])[:3])}
- Confidence: {cot_reasoning.get('confidence', 0):.2f}
"""

        prompt = f"""Reflect on these generated flashcards:

Generated Cards:
{card_text}

{validation_summary}
{cot_context}

Evaluate the quality of these cards. Consider:
1. Are questions clear and specific?
2. Are answers complete and well-formatted?
3. Is APF format correctly followed?
4. If CoT plan was provided, was it followed?
5. Will these cards be effective for memorization?

Determine if revision is needed and provide specific suggestions."""

        # Run reflection
        result = await agent.run(prompt)
        output = result.data

        # Store trace with stage-specific data
        _store_reflection_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "card_quality_scores": output.card_quality_scores,
                "format_compliance": output.format_compliance,
                "content_accuracy": output.content_accuracy,
                "question_clarity": output.question_clarity,
                "answer_completeness": output.answer_completeness,
                "memorization_potential": output.memorization_potential,
                "recommended_card_changes": output.recommended_card_changes,
            },
        )

        state["stage_times"]["reflect_after_generation"] = time.time() - start_time
        state["messages"].append(
            f"Self-reflection generation: revision_needed={output.revision_needed}, "
            f"confidence={output.confidence:.2f}"
        )

    except Exception as e:
        # Log and continue - reflection failure should not block pipeline
        logger.warning("self_reflection_generation_failed", error=str(e))
        state["current_reflection"] = None

    return state


async def reflect_after_enrichment_node(state: PipelineState) -> PipelineState:
    """Reflect on enriched cards after context enrichment.

    Evaluates enrichment quality, over-enrichment risk, and CoT plan adherence.
    Determines if revision is needed.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with reflection trace and revision decision
    """
    stage = "context_enrichment"

    if _should_skip_reflection(state, stage):
        return state

    if not increment_step_count(state, "reflect_after_enrichment"):
        return state

    logger.info("self_reflection_after_enrichment_start")
    start_time = time.time()

    model = state.get("reflection_model")
    if model is None:
        logger.warning("self_reflection_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, EnrichmentReflectionOutput] = Agent(
            model=model,
            result_type=EnrichmentReflectionOutput,
            system_prompt=ENRICHMENT_REFLECTION_PROMPT,
        )

        # Get enrichment results
        context_enrichment = state.get("context_enrichment")
        if not context_enrichment:
            logger.warning("self_reflection_no_enrichment_output")
            return state

        generation = state.get("generation")

        # Get CoT reasoning for context
        cot_reasoning = _get_cot_reasoning_for_stage(state, stage)

        # Build enrichment summary
        enrichments = context_enrichment.get("enrichments", [])
        enrichment_summary = []
        for i, enrichment in enumerate(enrichments[:5], 1):
            enrichment_summary.append(f"Enrichment {i}:")
            enrichment_summary.append(f"  Examples: {len(enrichment.get('examples', []))}")
            enrichment_summary.append(f"  Mnemonics: {len(enrichment.get('mnemonics', []))}")
            enrichment_summary.append(f"  Context added: {bool(enrichment.get('context'))}")
        enrichment_text = "\n".join(enrichment_summary)

        if len(enrichments) > 5:
            enrichment_text += f"\n... and {len(enrichments) - 5} more enrichments"

        # Include CoT context if available
        cot_context = ""
        if cot_reasoning:
            cot_context = f"""

Original CoT Reasoning Plan:
- Enrichment opportunities: {', '.join(cot_reasoning.get('stage_specific_data', {}).get('enrichment_opportunities', [])[:3])}
- Mnemonic suggestions: {', '.join(cot_reasoning.get('stage_specific_data', {}).get('mnemonic_suggestions', [])[:3])}
- Example types recommended: {', '.join(cot_reasoning.get('stage_specific_data', {}).get('example_types', [])[:3])}
- Confidence: {cot_reasoning.get('confidence', 0):.2f}
"""

        prompt = f"""Reflect on these enriched flashcards:

Enrichment Summary:
{enrichment_text}

Original cards count: {len(generation.get('cards', [])) if generation else 0}
{cot_context}

Evaluate the enrichment quality. Consider:
1. Are examples relevant and helpful?
2. Are mnemonics effective and memorable?
3. Is there risk of over-enrichment (cognitive overload)?
4. If CoT plan was provided, was it followed?
5. Will enrichments improve long-term retention?

Determine if revision is needed and provide specific suggestions."""

        result = await agent.run(prompt)
        output = result.data

        _store_reflection_trace(
            state,
            stage,
            output,
            time.time() - start_time,
            stage_specific_data={
                "example_quality": output.example_quality,
                "mnemonic_effectiveness": output.mnemonic_effectiveness,
                "context_relevance": output.context_relevance,
                "enrichment_impact": output.enrichment_impact,
                "over_enrichment_risk": output.over_enrichment_risk,
                "recommended_enrichment_changes": output.recommended_enrichment_changes,
            },
        )

        state["stage_times"]["reflect_after_enrichment"] = time.time() - start_time
        state["messages"].append(
            f"Self-reflection enrichment: revision_needed={output.revision_needed}, "
            f"confidence={output.confidence:.2f}"
        )

    except Exception as e:
        logger.warning("self_reflection_enrichment_failed", error=str(e))
        state["current_reflection"] = None

    return state


async def revise_generation_node(state: PipelineState) -> PipelineState:
    """Revise generated cards based on reflection feedback.

    Uses reflection analysis to make targeted improvements to generated cards.
    Increments revision count and updates generation output.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with revised generation
    """
    stage = "generation"

    if not increment_step_count(state, "revise_generation"):
        return state

    # Check if revision is allowed
    if not _can_revise(state, stage):
        logger.info("revise_generation_max_revisions_reached")
        return state

    # Get reflection feedback
    current_reflection = state.get("current_reflection")
    if not current_reflection or not current_reflection.get("revision_needed", False):
        logger.info("revise_generation_no_revision_needed")
        return state

    logger.info("revise_generation_start")
    start_time = time.time()

    model = state.get("reflection_model")  # Use same model as reflection
    if model is None:
        logger.warning("revision_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, RevisionOutput] = Agent(
            model=model,
            result_type=RevisionOutput,
            system_prompt=REVISION_PROMPT_GENERATION,
        )

        generation = state.get("generation")
        if not generation:
            return state

        # Get CoT reasoning for additional context
        cot_reasoning = _get_cot_reasoning_for_stage(state, stage)

        # Build revision context
        issues = current_reflection.get("issues_found", [])
        suggestions = current_reflection.get("revision_suggestions", [])

        prompt = f"""Revise these generated flashcards based on reflection feedback:

Original Output:
{generation}

Issues Found:
{chr(10).join(f'- {issue}' for issue in issues[:10])}

Revision Suggestions:
{chr(10).join(f'- {s.get("suggestion", "")} (severity: {s.get("severity", "medium")})' for s in suggestions[:10])}

Reflection Quality Assessment: {current_reflection.get('quality_assessment', 'N/A')}
Revision Priority: {current_reflection.get('revision_priority', 'medium')}

{f"CoT Plan: {cot_reasoning.get('planned_approach', 'N/A')}" if cot_reasoning else ""}

Make targeted improvements to address the identified issues.
Preserve what is working well. Focus on the highest priority issues first."""

        result = await agent.run(prompt)
        output = result.data

        # Update generation with revised output
        if output.revised_output:
            state["generation"] = output.revised_output

        # Increment revision count
        _increment_revision_count(state, stage)

        state["stage_times"]["revise_generation"] = time.time() - start_time
        state["messages"].append(
            f"Revised generation: changes={len(output.changes_made)}, "
            f"issues_addressed={len(output.issues_addressed)}, "
            f"confidence={output.revision_confidence:.2f}"
        )

        # Log revision details
        logger.info(
            "revise_generation_complete",
            changes_made=len(output.changes_made),
            issues_addressed=len(output.issues_addressed),
            issues_remaining=len(output.issues_remaining),
            revision_confidence=output.revision_confidence,
            further_revision_recommended=output.further_revision_recommended,
        )

    except Exception as e:
        logger.warning("revise_generation_failed", error=str(e))

    return state


async def revise_enrichment_node(state: PipelineState) -> PipelineState:
    """Revise enriched cards based on reflection feedback.

    Uses reflection analysis to improve enrichments.
    Increments revision count and updates enrichment output.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with revised enrichment
    """
    stage = "context_enrichment"

    if not increment_step_count(state, "revise_enrichment"):
        return state

    if not _can_revise(state, stage):
        logger.info("revise_enrichment_max_revisions_reached")
        return state

    current_reflection = state.get("current_reflection")
    if not current_reflection or not current_reflection.get("revision_needed", False):
        logger.info("revise_enrichment_no_revision_needed")
        return state

    logger.info("revise_enrichment_start")
    start_time = time.time()

    model = state.get("reflection_model")
    if model is None:
        logger.warning("revision_model_not_available", stage=stage)
        return state

    try:
        agent: Agent[None, RevisionOutput] = Agent(
            model=model,
            result_type=RevisionOutput,
            system_prompt=REVISION_PROMPT_ENRICHMENT,
        )

        context_enrichment = state.get("context_enrichment")
        if not context_enrichment:
            return state

        cot_reasoning = _get_cot_reasoning_for_stage(state, stage)

        issues = current_reflection.get("issues_found", [])
        suggestions = current_reflection.get("revision_suggestions", [])

        prompt = f"""Revise these enriched flashcards based on reflection feedback:

Current Enrichment Output:
{context_enrichment}

Issues Found:
{chr(10).join(f'- {issue}' for issue in issues[:10])}

Revision Suggestions:
{chr(10).join(f'- {s.get("suggestion", "")} (severity: {s.get("severity", "medium")})' for s in suggestions[:10])}

Over-enrichment Risk: {current_reflection.get('stage_specific_data', {}).get('over_enrichment_risk', False)}
Enrichment Impact Assessment: {current_reflection.get('stage_specific_data', {}).get('enrichment_impact', 'N/A')}

{f"CoT Plan: {cot_reasoning.get('planned_approach', 'N/A')}" if cot_reasoning else ""}

Improve the enrichments based on feedback.
If over-enrichment risk is high, consider trimming excessive content.
Focus on quality over quantity."""

        result = await agent.run(prompt)
        output = result.data

        if output.revised_output:
            state["context_enrichment"] = output.revised_output

        _increment_revision_count(state, stage)

        state["stage_times"]["revise_enrichment"] = time.time() - start_time
        state["messages"].append(
            f"Revised enrichment: changes={len(output.changes_made)}, "
            f"issues_addressed={len(output.issues_addressed)}, "
            f"confidence={output.revision_confidence:.2f}"
        )

        logger.info(
            "revise_enrichment_complete",
            changes_made=len(output.changes_made),
            issues_addressed=len(output.issues_addressed),
            issues_remaining=len(output.issues_remaining),
            revision_confidence=output.revision_confidence,
            further_revision_recommended=output.further_revision_recommended,
        )

    except Exception as e:
        logger.warning("revise_enrichment_failed", error=str(e))

    return state


def should_revise_generation(state: PipelineState) -> bool:
    """Routing function to determine if generation should be revised.

    Args:
        state: Current pipeline state

    Returns:
        True if revision should occur
    """
    current_reflection = state.get("current_reflection")
    if not current_reflection:
        return False

    if not current_reflection.get("revision_needed", False):
        return False

    return _can_revise(state, "generation")


def should_revise_enrichment(state: PipelineState) -> bool:
    """Routing function to determine if enrichment should be revised.

    Args:
        state: Current pipeline state

    Returns:
        True if revision should occur
    """
    current_reflection = state.get("current_reflection")
    if not current_reflection:
        return False

    if not current_reflection.get("revision_needed", False):
        return False

    return _can_revise(state, "context_enrichment")
