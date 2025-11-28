"""LangGraph-based orchestrator for the card generation pipeline.

DEPRECATED: This module is now a facade for backward compatibility.
Import from `agents.langgraph` package instead:

    from agents.langgraph import LangGraphOrchestrator, PipelineState

This module implements a state machine workflow using LangGraph to coordinate:
1. Pre-Validator Agent - structure and format validation
2. Generator Agent - card generation with LLM
3. Post-Validator Agent - quality validation with retry logic

The workflow supports conditional routing, automatic retries, and state persistence.

Best Practices Applied (2025):
- RetryPolicy with exponential backoff for transient failures
- Typed state with Pydantic validation
- Max steps counter for cycle protection
- Graceful error degradation with severity-based routing
- Checkpointing for fault tolerance

References:
- https://www.swarnendu.de/blog/langgraph-best-practices/
- https://langchain-ai.github.io/langgraph/how-tos/persistence/
"""

# Re-export everything from the langgraph package for backward compatibility
from .langgraph import (
    DEFAULT_RETRY_POLICY,
    TRANSIENT_RETRY_POLICY,
    VALIDATION_RETRY_POLICY,
    ErrorSeverity,
    LangGraphOrchestrator,
    PipelineState,
    classify_error_severity,
    is_transient_error,
)

# Re-export node helpers for backward compatibility (though these were private)
from .langgraph.node_helpers import (
    handle_node_error,
    increment_step_count,
    record_error,
)

# Re-export node functions for backward compatibility (though these were private)
from .langgraph.nodes import (
    card_splitting_node,
    context_enrichment_node,
    duplicate_detection_node,
    generation_node,
    linter_validation_node,
    memorization_quality_node,
    note_correction_node,
    post_validation_node,
    pre_validation_node,
)

# Re-export routing functions for backward compatibility (though these were private)
from .langgraph.workflow_builder import (
    should_continue_after_enrichment,
    should_continue_after_memorization_quality,
    should_continue_after_post_validation,
    should_continue_after_pre_validation,
)

__all__ = [
    # Main orchestrator
    "LangGraphOrchestrator",
    # State
    "PipelineState",
    # Retry policies
    "DEFAULT_RETRY_POLICY",
    "VALIDATION_RETRY_POLICY",
    "TRANSIENT_RETRY_POLICY",
    "is_transient_error",
    # Error handling
    "ErrorSeverity",
    "classify_error_severity",
    # Node functions (private, but exported for compatibility)
    "note_correction_node",
    "pre_validation_node",
    "card_splitting_node",
    "generation_node",
    "linter_validation_node",
    "post_validation_node",
    "context_enrichment_node",
    "memorization_quality_node",
    "duplicate_detection_node",
    # Node helpers (private, but exported for compatibility)
    "increment_step_count",
    "record_error",
    "handle_node_error",
    # Routing functions (private, but exported for compatibility)
    "should_continue_after_pre_validation",
    "should_continue_after_post_validation",
    "should_continue_after_enrichment",
    "should_continue_after_memorization_quality",
]
