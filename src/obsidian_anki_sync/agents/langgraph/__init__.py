"""LangGraph-based orchestrator for card generation pipeline.

This package provides a state machine workflow using LangGraph to coordinate
the multi-agent card generation pipeline with automatic retries, error handling,
and state persistence.

Public API:
- LangGraphOrchestrator: Main orchestrator class
- PipelineState: State TypedDict for the workflow
- Retry policies and error handling utilities
"""

from .orchestrator import LangGraphOrchestrator
from .retry_policies import (
    DEFAULT_RETRY_POLICY,
    TRANSIENT_RETRY_POLICY,
    VALIDATION_RETRY_POLICY,
    ErrorSeverity,
    classify_error_severity,
    is_transient_error,
)
from .state import PipelineState

# Optional: Swarm pattern (requires langgraph-swarm package)
try:
    from .swarm_orchestrator import LangGraphSwarmOrchestrator, SwarmResult
    _SWARM_AVAILABLE = True
except ImportError:
    LangGraphSwarmOrchestrator = None  # type: ignore[assignment, misc]
    SwarmResult = None  # type: ignore[assignment, misc]
    _SWARM_AVAILABLE = False

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
]

# Only export swarm classes if available
if _SWARM_AVAILABLE:
    __all__.extend(["LangGraphSwarmOrchestrator", "SwarmResult"])
