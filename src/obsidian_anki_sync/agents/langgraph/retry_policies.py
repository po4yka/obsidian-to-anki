"""Retry policies and error handling for LangGraph pipeline.

This module defines retry policies with exponential backoff and error
classification logic for routing decisions.
"""

from langgraph.types import RetryPolicy

from obsidian_anki_sync.agents.models import RepairStrategy

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
    return bool(
        any(
            term in error_msg
            for term in ["overloaded", "capacity", "retry", "temporary"]
        )
    )


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


def classify_error_category(error: Exception) -> str:
    """Classify error into specific categories for repair routing.

    Expanded categories:
    - syntax: YAML/JSON syntax errors, malformed structure
    - structure: Missing sections, incorrect hierarchy
    - content: Missing or invalid content, quality issues
    - quality: Grammar, clarity, completeness issues
    - frontmatter: YAML frontmatter specific issues
    - unknown: Unclassified errors

    Args:
        error: Exception to classify

    Returns:
        Error category string
    """
    error_msg = str(error).lower()

    # Syntax errors
    if any(
        term in error_msg
        for term in [
            "syntax",
            "parse",
            "json",
            "yaml",
            "malformed",
            "invalid format",
            "unexpected token",
            "invalid character",
        ]
    ):
        return "syntax"

    # Structure errors
    if any(
        term in error_msg
        for term in [
            "missing section",
            "missing field",
            "required field",
            "structure",
            "hierarchy",
            "incorrect level",
            "missing header",
        ]
    ):
        return "structure"

    # Frontmatter errors
    if any(
        term in error_msg
        for term in [
            "frontmatter",
            "metadata",
            "yaml frontmatter",
            "header",
            "language_tags",
            "missing id",
            "missing title",
        ]
    ):
        return "frontmatter"

    # Content errors
    if any(
        term in error_msg
        for term in [
            "missing content",
            "empty",
            "no content",
            "invalid content",
            "content error",
        ]
    ):
        return "content"

    # Quality errors
    if any(
        term in error_msg
        for term in [
            "grammar",
            "clarity",
            "quality",
            "incomplete",
            "truncated",
            "bilingual",
            "consistency",
        ]
    ):
        return "quality"

    # Default to unknown
    return "unknown"


def select_repair_strategy(error: Exception) -> RepairStrategy:
    """Select repair strategy based on error classification.

    Strategies:
    - deterministic: Fast rule-based fixes (syntax errors)
    - rule_based: Pattern matching fixes (structure errors)
    - llm_based: LLM-powered fixes (content/quality errors)
    - multi_stage: Multiple repair stages (complex errors)
    - partial: Partial fixes with flagging (uncertain errors)
    - skip: Skip repair (critical/unrecoverable errors)

    Args:
        error: Exception to analyze

    Returns:
        RepairStrategy with selected approach
    """
    severity = classify_error_severity(error)
    category = classify_error_category(error)

    # Critical errors: skip repair
    if severity == ErrorSeverity.CRITICAL:
        return RepairStrategy(
            strategy_type="skip",
            priority=10,
            stages=[],
            confidence_threshold=0.0,
        )

    # Syntax errors: deterministic fixes
    if category == "syntax":
        return RepairStrategy(
            strategy_type="deterministic",
            priority=1,
            stages=["syntax"],
            confidence_threshold=0.9,
        )

    # Structure errors: rule-based fixes
    if category == "structure":
        return RepairStrategy(
            strategy_type="rule_based",
            priority=2,
            stages=["structure"],
            confidence_threshold=0.8,
        )

    # Frontmatter errors: rule-based with LLM fallback
    if category == "frontmatter":
        return RepairStrategy(
            strategy_type="multi_stage",
            priority=2,
            stages=["deterministic", "rule_based", "llm_based"],
            confidence_threshold=0.7,
        )

    # Content/quality errors: LLM-based fixes
    if category in ["content", "quality"]:
        return RepairStrategy(
            strategy_type="llm_based",
            priority=3,
            stages=["llm_based"],
            confidence_threshold=0.7,
        )

    # Unknown errors: multi-stage with lower confidence
    return RepairStrategy(
        strategy_type="multi_stage",
        priority=5,
        stages=["deterministic", "rule_based", "llm_based"],
        confidence_threshold=0.6,
    )


def get_repair_priority(error: Exception) -> int:
    """Get repair priority ranking (1=highest, 10=lowest).

    Args:
        error: Exception to rank

    Returns:
        Priority integer (1-10)
    """
    strategy = select_repair_strategy(error)
    return strategy.priority


# ============================================================================
# Repair-Specific Retry Policies
# ============================================================================


# Repair retry policy with exponential backoff
REPAIR_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.5,  # Start with 0.5 seconds
    backoff_factor=2.0,  # Double each retry
    max_interval=10.0,  # Cap at 10 seconds
    jitter=True,
)


def calculate_repair_backoff(attempt: int, base_interval: float = 0.5) -> float:
    """Calculate exponential backoff delay for repair attempts.

    Args:
        attempt: Current attempt number (0-indexed)
        base_interval: Base interval in seconds

    Returns:
        Backoff delay in seconds
    """
    import random

    delay = base_interval * (2.0**attempt)
    # Add jitter (randomization) to prevent thundering herd
    jitter = random.uniform(0.0, delay * 0.1)
    return min(delay + jitter, 10.0)  # Cap at 10 seconds


# CircuitBreaker implementation is in obsidian_anki_sync.utils.resilience
# Import from there for consistent tenacity-based circuit breaker pattern:
# from obsidian_anki_sync.utils.resilience import CircuitBreaker, CircuitBreakerConfig


def get_adaptive_retry_count(error: Exception, base_max_retries: int = 3) -> int:
    """Get adaptive retry count based on error type.

    More recoverable errors get more retries.

    Args:
        error: Exception to analyze
        base_max_retries: Base maximum retries

    Returns:
        Adaptive retry count
    """
    severity = classify_error_severity(error)
    category = classify_error_category(error)

    # Critical errors: no retries
    if severity == ErrorSeverity.CRITICAL:
        return 0

    # Syntax/structure errors: more retries (easier to fix)
    if category in ["syntax", "structure"]:
        return base_max_retries + 1

    # Content/quality errors: standard retries
    if category in ["content", "quality"]:
        return base_max_retries

    # Unknown errors: fewer retries (less likely to succeed)
    return max(1, base_max_retries - 1)
