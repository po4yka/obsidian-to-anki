from typing import Literal

from langgraph.graph import END, StateGraph

from ...config import Config
from .nodes import (
    card_splitting_node,
    context_enrichment_node,
    duplicate_detection_node,
    generation_node,
    linter_validation_node,
    memorization_quality_node,
    note_correction_node,
    post_validation_node,
    pre_validation_node,
    split_validation_node,
)
from .reasoning_nodes import (
    think_before_card_splitting_node,
    think_before_duplicate_node,
    think_before_enrichment_node,
    think_before_generation_node,
    think_before_memorization_node,
    think_before_post_validation_node,
    think_before_pre_validation_node,
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

        # Check if CoT reasoning is enabled
        enable_cot = getattr(self.config, "enable_cot_reasoning", False)

        # Add optional note correction node (if enabled)
        enable_note_correction = getattr(
            self.config, "enable_note_correction", False)
        if enable_note_correction:
            workflow.add_node(
                "note_correction",
                note_correction_node,
                retry=TRANSIENT_RETRY_POLICY,
            )

        # ====================================================================
        # Add Chain of Thought (CoT) reasoning nodes (if enabled)
        # Reasoning nodes run BEFORE their corresponding action nodes
        # They do NOT retry - reasoning failure should not block pipeline
        # ====================================================================
        if enable_cot:
            workflow.add_node(
                "think_before_pre_validation",
                think_before_pre_validation_node,
                retry=None,  # Advisory only - no retry
            )
            workflow.add_node(
                "think_before_card_splitting",
                think_before_card_splitting_node,
                retry=None,
            )
            workflow.add_node(
                "think_before_generation",
                think_before_generation_node,
                retry=None,
            )
            workflow.add_node(
                "think_before_post_validation",
                think_before_post_validation_node,
                retry=None,
            )
            workflow.add_node(
                "think_before_enrichment",
                think_before_enrichment_node,
                retry=None,
            )
            workflow.add_node(
                "think_before_memorization",
                think_before_memorization_node,
                retry=None,
            )
            workflow.add_node(
                "think_before_duplicate",
                think_before_duplicate_node,
                retry=None,
            )

        # Add core action nodes with appropriate retry policies
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
            "split_validation",
            split_validation_node,
            retry=VALIDATION_RETRY_POLICY,
        )

        workflow.add_node(
            "generation",
            generation_node,
            retry=TRANSIENT_RETRY_POLICY,
        )

        # Linter validation - deterministic APF template compliance check
        # This runs between generation and post_validation to provide
        # authoritative template validation (no LLM hallucinations)
        workflow.add_node(
            "linter_validation",
            linter_validation_node,
            retry=None,  # Deterministic - no retry needed
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

        # ====================================================================
        # Set entry point and routing
        # ====================================================================
        if enable_cot:
            # CoT enabled: Route through reasoning nodes
            if enable_note_correction:
                workflow.set_entry_point("note_correction")
                workflow.add_edge("note_correction", "think_before_pre_validation")
            else:
                workflow.set_entry_point("think_before_pre_validation")

            # Reasoning nodes always proceed to their action nodes
            workflow.add_edge("think_before_pre_validation", "pre_validation")
            workflow.add_edge("think_before_card_splitting", "card_splitting")
            workflow.add_edge("think_before_generation", "generation")
            workflow.add_edge("think_before_post_validation", "post_validation")
            workflow.add_edge("think_before_enrichment", "context_enrichment")
            workflow.add_edge("think_before_memorization", "memorization_quality")
            workflow.add_edge("think_before_duplicate", "duplicate_detection")

            # Pre-validation routes to thinking nodes
            workflow.add_conditional_edges(
                "pre_validation",
                self._route_after_pre_validation_with_cot,
                {
                    "think_card_splitting": "think_before_card_splitting",
                    "think_generation": "think_before_generation",
                    "failed": END,
                },
            )

            # Card splitting -> split validation -> think_before_generation
            workflow.add_edge("card_splitting", "split_validation")
            workflow.add_edge("split_validation", "think_before_generation")

            # Generation -> Linter -> think_before_post_validation
            workflow.add_edge("generation", "linter_validation")
            workflow.add_edge("linter_validation", "think_before_post_validation")

            # Post-validation routes to thinking nodes
            workflow.add_conditional_edges(
                "post_validation",
                self._route_after_post_validation_with_cot,
                {
                    "think_enrichment": "think_before_enrichment",
                    "think_generation": "think_before_generation",  # Retry
                    "failed": END,
                },
            )

            # Enrichment routes to thinking nodes
            workflow.add_conditional_edges(
                "context_enrichment",
                self._route_after_enrichment_with_cot,
                {
                    "think_memorization": "think_before_memorization",
                    "complete": END,
                },
            )

            # Memorization routes to thinking nodes
            workflow.add_conditional_edges(
                "memorization_quality",
                self._route_after_memorization_with_cot,
                {
                    "think_duplicate": "think_before_duplicate",
                    "complete": END,
                },
            )

            # Duplicate detection always goes to END
            workflow.add_edge("duplicate_detection", END)

        else:
            # CoT disabled: Original routing (no reasoning nodes)
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

            # Card splitting goes to split validation
            workflow.add_edge("card_splitting", "split_validation")

            # Split validation goes to generation
            workflow.add_edge("split_validation", "generation")

            # Generation -> Linter validation -> Post-validation
            workflow.add_edge("generation", "linter_validation")
            workflow.add_edge("linter_validation", "post_validation")

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

    # ========================================================================
    # CoT-specific routing functions
    # ========================================================================

    def _route_after_pre_validation_with_cot(
        self, state: PipelineState
    ) -> Literal["think_card_splitting", "think_generation", "failed"]:
        """Route after pre-validation when CoT is enabled."""
        pre_validation = state.get("pre_validation")
        if pre_validation and pre_validation["is_valid"]:
            if state.get("enable_card_splitting", True):
                return "think_card_splitting"
            return "think_generation"
        return "failed"

    def _route_after_post_validation_with_cot(
        self, state: PipelineState
    ) -> Literal["think_enrichment", "think_generation", "failed"]:
        """Route after post-validation when CoT is enabled."""
        current_stage = state.get("current_stage", "failed")

        if current_stage == "context_enrichment":
            return "think_enrichment"
        elif current_stage == "generation":
            return "think_generation"  # Retry through reasoning
        else:
            return "failed"

    def _route_after_enrichment_with_cot(
        self, state: PipelineState
    ) -> Literal["think_memorization", "complete"]:
        """Route after enrichment when CoT is enabled."""
        current_stage = state.get("current_stage", "complete")

        if current_stage == "memorization_quality":
            return "think_memorization"
        return "complete"

    def _route_after_memorization_with_cot(
        self, state: PipelineState
    ) -> Literal["think_duplicate", "complete"]:
        """Route after memorization quality when CoT is enabled."""
        current_stage = state.get("current_stage", "complete")

        if current_stage == "duplicate_detection":
            return "think_duplicate"
        return "complete"
