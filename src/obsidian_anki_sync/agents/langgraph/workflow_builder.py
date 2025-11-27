from typing import Literal

from langgraph.graph import END, StateGraph

from ...config import Config
from .nodes import (
    card_splitting_node,
    context_enrichment_node,
    duplicate_detection_node,
    generation_node,
    memorization_quality_node,
    note_correction_node,
    post_validation_node,
    pre_validation_node,
)
from .retry_policies import TRANSIENT_RETRY_POLICY, VALIDATION_RETRY_POLICY
from .state import PipelineState


# Conditional Routing Functions
def should_continue_after_pre_validation(
    state: PipelineState,
) -> Literal["card_splitting", "generation", "failed"]:
    """Determine next node after pre-validation."""
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
    """Determine next node after post-validation."""
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
    """Determine next node after context enrichment."""
    current_stage = state.get("current_stage", "complete")

    if current_stage == "memorization_quality":
        return "memorization_quality"
    else:
        return "complete"


def should_continue_after_memorization_quality(
    state: PipelineState,
) -> Literal["duplicate_detection", "complete"]:
    """Determine next node after memorization quality."""
    current_stage = state.get("current_stage", "complete")

    if current_stage == "duplicate_detection":
        return "duplicate_detection"
    else:
        return "complete"


class WorkflowBuilder:
    """Builder for the LangGraph workflow."""

    def __init__(self, config: Config):
        """Initialize the workflow builder.

        Args:
            config: Service configuration
        """
        self.config = config

    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Configured StateGraph instance
        """
        # Create workflow graph
        workflow = StateGraph(PipelineState)

        # Add optional note correction node (if enabled)
        enable_note_correction = getattr(
            self.config, "enable_note_correction", False)
        if enable_note_correction:
            workflow.add_node(
                "note_correction",
                note_correction_node,
                retry=TRANSIENT_RETRY_POLICY,
            )

        # Add core nodes with appropriate retry policies
        workflow.add_node(
            "pre_validation",
            pre_validation_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        workflow.add_node(
            "card_splitting",
            card_splitting_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        workflow.add_node(
            "generation",
            generation_node,
            retry=TRANSIENT_RETRY_POLICY,
        )

        workflow.add_node(
            "post_validation",
            post_validation_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Add enhancement nodes
        workflow.add_node(
            "context_enrichment",
            context_enrichment_node,
            retry=VALIDATION_RETRY_POLICY,
        )
        workflow.add_node(
            "memorization_quality",
            memorization_quality_node,
            retry=VALIDATION_RETRY_POLICY,
        )
        workflow.add_node(
            "duplicate_detection",
            duplicate_detection_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        # Set entry point
        if enable_note_correction:
            workflow.set_entry_point("note_correction")
            workflow.add_edge("note_correction", "pre_validation")
        else:
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
