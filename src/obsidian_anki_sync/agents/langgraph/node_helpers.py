"""Helper functions for LangGraph pipeline nodes.

This module provides utility functions for node execution including
step counting, error recording, and error handling.
"""

import time
from typing import Any

from ...utils.logging import get_logger
from .retry_policies import ErrorSeverity, classify_error_severity
from .state import PipelineState

logger = get_logger(__name__)


# ============================================================================
# Node Helper Functions
# ============================================================================


def increment_step_count(state: PipelineState, stage_name: str) -> bool:
    """Increment step count and check for max steps limit.

    Best practice: Add hard stops to prevent infinite loops in cycles.

    Args:
        state: Current pipeline state
        stage_name: Name of the current stage for logging

    Returns:
        True if within limits, False if max steps exceeded
    """
    state["step_count"] = state.get("step_count", 0) + 1
    max_steps = state.get("max_steps", 20)

    if state["step_count"] > max_steps:
        logger.error(
            "max_steps_exceeded",
            step_count=state["step_count"],
            max_steps=max_steps,
            stage=stage_name,
        )
        state["current_stage"] = "failed"
        state["last_error"] = f"Max steps ({max_steps}) exceeded at stage {stage_name}"
        state["last_error_severity"] = ErrorSeverity.CRITICAL
        record_error(state, stage_name, state["last_error"], ErrorSeverity.CRITICAL)
        return False

    logger.debug(
        "step_count_incremented",
        step_count=state["step_count"],
        max_steps=max_steps,
        stage=stage_name,
    )
    return True


def record_error(
    state: PipelineState,
    stage: str,
    error: str,
    severity: str,
) -> None:
    """Record an error in the state for debugging and routing.

    Best practice: Track all errors with context for debugging.

    Args:
        state: Current pipeline state
        stage: Stage where error occurred
        error: Error message
        severity: Error severity level
    """
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    state["errors"].append(
        {
            "stage": stage,
            "error": error,
            "severity": severity,
            "timestamp": time.time(),
            "step_count": state.get("step_count", 0),
        }
    )

    state["last_error"] = error
    state["last_error_severity"] = severity

    logger.info(
        "error_recorded",
        stage=stage,
        severity=severity,
        error_count=len(state["errors"]),
    )


def handle_node_error(
    state: PipelineState,
    stage: str,
    error: Exception,
    fallback_result: Any = None,
) -> tuple[PipelineState, str]:
    """Handle an error in a node with appropriate routing.

    Best practice: Different errors need different handling strategies.

    Args:
        state: Current pipeline state
        stage: Stage where error occurred
        error: The exception that occurred
        fallback_result: Optional fallback result to use

    Returns:
        Tuple of (updated state, next stage)
    """
    severity = classify_error_severity(error)
    record_error(state, stage, str(error), severity)

    if severity == ErrorSeverity.CRITICAL:
        # Critical errors stop the pipeline
        logger.error(
            "critical_error_stopping_pipeline",
            stage=stage,
            error=str(error),
        )
        state["current_stage"] = "failed"
        return state, "failed"

    elif severity == ErrorSeverity.WARNING:
        # Warnings allow continuation with degraded output
        logger.warning(
            "warning_continuing_with_fallback",
            stage=stage,
            error=str(error),
        )
        # Don't change stage - let caller decide

    else:  # RECOVERABLE
        # Recoverable errors may retry if retries available
        if state["retry_count"] < state["max_retries"]:
            logger.info(
                "recoverable_error_will_retry",
                stage=stage,
                error=str(error),
                retry_count=state["retry_count"],
            )
        else:
            logger.warning(
                "recoverable_error_no_retries_left",
                stage=stage,
                error=str(error),
            )

    return state, state.get("current_stage", "failed")
