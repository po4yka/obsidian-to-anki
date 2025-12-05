"""LangGraph-based orchestrator for card generation pipeline.

This package provides a state machine workflow using LangGraph to coordinate
the multi-agent card generation pipeline with automatic retries, error handling,
and state persistence.

Public API:
- LangGraphOrchestrator: Main orchestrator class
- PipelineState: State TypedDict for the workflow
- Retry policies and error handling utilities
- Chain of Thought (CoT) reasoning models and nodes
- Self-Reflection models and nodes for output evaluation and revision
"""

from .orchestrator import LangGraphOrchestrator
from .reasoning_models import (
    CardSplittingReasoningOutput,
    DuplicateReasoningOutput,
    EnrichmentReasoningOutput,
    GenerationReasoningOutput,
    MemorizationReasoningOutput,
    PostValidationReasoningOutput,
    PreValidationReasoningOutput,
    ReasoningTrace,
    ReasoningTraceOutput,
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
from .reflection_models import (
    EnrichmentReflectionOutput,
    GenerationReflectionOutput,
    ReflectionOutput,
    ReflectionTrace,
    RevisionInput,
    RevisionOutput,
    RevisionSuggestion,
)
from .reflection_nodes import (
    reflect_after_enrichment_node,
    reflect_after_generation_node,
    revise_enrichment_node,
    revise_generation_node,
    should_revise_enrichment,
    should_revise_generation,
)
from .retry_policies import (
    DEFAULT_RETRY_POLICY,
    TRANSIENT_RETRY_POLICY,
    VALIDATION_RETRY_POLICY,
    ErrorSeverity,
    classify_error_severity,
    is_transient_error,
)
from .state import PipelineState

# Node modules (following SOLID principles and Clean Architecture)
from . import (
    correction_nodes,
    detection_nodes,
    enhancement_nodes,
    generation_nodes,
    validation_nodes,
)

# Optional: Swarm pattern (requires langgraph-swarm package)
try:
    from .swarm_orchestrator import LangGraphSwarmOrchestrator, SwarmResult

    _SWARM_AVAILABLE = True
except ImportError:
    LangGraphSwarmOrchestrator = None  # type: ignore[assignment, misc]
    SwarmResult = None  # type: ignore[assignment, misc]
    _SWARM_AVAILABLE = False

__all__ = [
    # Retry policies
    "DEFAULT_RETRY_POLICY",
    "TRANSIENT_RETRY_POLICY",
    "VALIDATION_RETRY_POLICY",
    "CardSplittingReasoningOutput",
    "DuplicateReasoningOutput",
    "EnrichmentReasoningOutput",
    "EnrichmentReflectionOutput",
    # Error handling
    "ErrorSeverity",
    "GenerationReasoningOutput",
    "GenerationReflectionOutput",
    # Main orchestrator
    "LangGraphOrchestrator",
    "MemorizationReasoningOutput",
    # State
    "PipelineState",
    "PostValidationReasoningOutput",
    "PreValidationReasoningOutput",
    "ReasoningTrace",
    # Chain of Thought (CoT) reasoning models
    "ReasoningTraceOutput",
    # Self-Reflection models
    "ReflectionOutput",
    "ReflectionTrace",
    "RevisionInput",
    "RevisionOutput",
    "RevisionSuggestion",
    "classify_error_severity",
    "is_transient_error",
    "reflect_after_enrichment_node",
    # Self-Reflection nodes
    "reflect_after_generation_node",
    "revise_enrichment_node",
    "revise_generation_node",
    "should_revise_enrichment",
    "should_revise_generation",
    "think_before_card_splitting_node",
    "think_before_duplicate_node",
    "think_before_enrichment_node",
    "think_before_generation_node",
    "think_before_memorization_node",
    "think_before_post_validation_node",
    # CoT reasoning nodes
    "think_before_pre_validation_node",
]

# Only export swarm classes if available
if _SWARM_AVAILABLE:
    __all__ += ["LangGraphSwarmOrchestrator", "SwarmResult"]
