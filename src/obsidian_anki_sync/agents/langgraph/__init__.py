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
