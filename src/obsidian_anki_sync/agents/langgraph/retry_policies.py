"""Retry policies and error handling for LangGraph pipeline.

This module defines retry policies with exponential backoff and error
classification logic for routing decisions.
"""

from langgraph.types import RetryPolicy

# ============================================================================
# Retry Policy Configuration
# ============================================================================

# Default retry policy for LLM-based nodes (handles transient API failures)
# Best practice: Retry on transient errors (5xx, 429, network issues)
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,  # Start with 1 second delay
    backoff_factor=2.0,  # Double the delay each retry
    max_interval=30.0,  # Cap at 30 seconds
    jitter=True,  # Add randomization to prevent thundering herd
)

# Lighter retry policy for validation nodes (faster, less critical)
VALIDATION_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
)


# ============================================================================
# Error Classification
# ============================================================================


def is_transient_error(exc: Exception) -> bool:
    """Determine if an exception is transient and should be retried.

    Best practice: Only retry on server errors (5xx), rate limits (429),
    and network/timeout errors. Don't retry validation or logic errors.

    Args:
        exc: Exception to classify

    Returns:
        True if error is transient and should be retried
    """
    error_msg = str(exc).lower()

    # Network/timeout errors
    if any(
        term in error_msg
        for term in [
            "timeout",
            "timed out",
            "connection",
            "network",
            "temporarily unavailable",
            "service unavailable",
        ]
    ):
        return True

    # Rate limiting
    if (
        "429" in error_msg
        or "rate limit" in error_msg
        or "too many requests" in error_msg
    ):
        return True

    # Server errors (5xx)
    if any(f"{code}" in error_msg for code in range(500, 600)):
        return True

    # API-specific transient errors
    if any(
        term in error_msg for term in ["overloaded", "capacity", "retry", "temporary"]
    ):
        return True

    return False


# Retry policy with custom retry condition
TRANSIENT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
    retry_on=is_transient_error,
)


# ============================================================================
# Error Severity Classification
# ============================================================================


class ErrorSeverity:
    """Error severity levels for routing decisions."""

    CRITICAL = "critical"  # Unrecoverable, stop pipeline
    RECOVERABLE = "recoverable"  # Can retry or use fallback
    WARNING = "warning"  # Log and continue


def classify_error_severity(error: Exception) -> str:
    """Classify error severity for routing decisions.

    Best practice: Different errors need different handling strategies.
    - Critical: Stop and escalate
    - Recoverable: Retry with backoff or use fallback
    - Warning: Log and continue with degraded output

    Args:
        error: Exception to classify

    Returns:
        Error severity level (critical/recoverable/warning)
    """
    error_msg = str(error).lower()

    # Critical errors - cannot continue
    if any(
        term in error_msg
        for term in [
            "api key",
            "authentication",
            "authorization",
            "forbidden",
            "invalid model",
            "model not found",
            "quota exceeded",
        ]
    ):
        return ErrorSeverity.CRITICAL

    # Recoverable errors - can retry
    if is_transient_error(error):
        return ErrorSeverity.RECOVERABLE

    # Validation errors - warning, continue with fallback
    if any(
        term in error_msg
        for term in ["validation", "format", "parse", "json", "schema"]
    ):
        return ErrorSeverity.WARNING

    # Default to recoverable for unknown errors
    return ErrorSeverity.RECOVERABLE
