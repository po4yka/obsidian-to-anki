"""Enhanced logging utilities for LLM operations.

This module provides structured logging helpers for LLM operations,
including token usage, timing, cost estimation, and performance metrics.
"""

import time
from collections import defaultdict
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)

# Global session tracking for cumulative metrics
_session_metrics: dict[str, dict[str, Any]] = defaultdict(lambda: {
    "total_cost": 0.0,
    "total_tokens": 0,
    "total_requests": 0,
    "total_duration": 0.0,
    "slow_requests": 0,
    "errors": 0,
    "by_model": defaultdict(lambda: {
        "cost": 0.0,
        "tokens": 0,
        "requests": 0,
        "duration": 0.0,
    }),
})

# OpenRouter pricing (per 1M tokens) - update as needed
# Source: https://openrouter.ai/models (approximate pricing)
OPENROUTER_PRICING: dict[str, dict[str, float]] = {
    "moonshotai/kimi-k2": {"prompt": 0.25, "completion": 0.25},
    "moonshotai/kimi-k2-thinking": {"prompt": 0.50, "completion": 0.50},
    "qwen/qwen-2.5-72b-instruct": {"prompt": 0.55, "completion": 0.55},
    "qwen/qwen-2.5-32b-instruct": {"prompt": 0.20, "completion": 0.20},
    "deepseek/deepseek-chat-v3.1": {"prompt": 0.14, "completion": 0.28},
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "minimax/minimax-m2": {"prompt": 0.30, "completion": 0.30},
    "qwen/qwen3-max": {"prompt": 0.50, "completion": 0.50},
    # Qwen3 series models
    "qwen/qwen3-235b-a22b-2507": {"prompt": 0.08, "completion": 0.55},
    "qwen/qwen3-235b-a22b-thinking-2507": {"prompt": 0.11, "completion": 0.60},
    "qwen/qwen3-next-80b-a3b-instruct": {"prompt": 0.10, "completion": 0.80},
    "qwen/qwen3-32b": {"prompt": 0.05, "completion": 0.20},
    "qwen/qwen3-30b-a3b": {"prompt": 0.06, "completion": 0.22},
}

DEFAULT_PRICING = {"prompt": 0.50, "completion": 0.50}  # Conservative default


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    """Estimate cost for an LLM request.

    Args:
        model: Model identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Dictionary with cost breakdown and total
    """
    pricing = OPENROUTER_PRICING.get(model, DEFAULT_PRICING)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_cost_usd": round(prompt_cost, 6),
        "completion_cost_usd": round(completion_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "pricing_source": "openrouter",
    }


def calculate_tokens_per_second(completion_tokens: int, duration_seconds: float) -> float:
    """Calculate tokens per second for completion.

    Args:
        completion_tokens: Number of completion tokens
        duration_seconds: Duration in seconds

    Returns:
        Tokens per second (0 if duration is 0)
    """
    if duration_seconds <= 0:
        return 0.0
    return round(completion_tokens / duration_seconds, 2)


def calculate_context_window_usage(
    prompt_tokens: int, completion_tokens: int, context_window: int
) -> dict[str, Any]:
    """Calculate context window usage statistics.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        context_window: Total context window size

    Returns:
        Dictionary with usage statistics
    """
    total_used = prompt_tokens + completion_tokens
    usage_percent = (total_used / context_window * 100) if context_window > 0 else 0
    prompt_percent = (prompt_tokens / context_window * 100) if context_window > 0 else 0
    remaining = context_window - total_used

    return {
        "total_tokens_used": total_used,
        "context_window_size": context_window,
        "usage_percent": round(usage_percent, 2),
        "prompt_percent": round(prompt_percent, 2),
        "remaining_tokens": remaining,
        "is_near_limit": usage_percent > 80,
    }


def log_llm_request(
    model: str,
    operation: str,
    prompt_length: int,
    system_length: int = 0,
    prompt_tokens_estimate: int | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    has_json_schema: bool = False,
    **extra_context: Any,
) -> float:
    """Log LLM request start and return start time.

    Args:
        model: Model identifier
        operation: Operation name (e.g., "qa_extraction", "card_generation")
        prompt_length: Length of prompt in characters
        system_length: Length of system prompt in characters
        prompt_tokens_estimate: Estimated prompt tokens (if available)
        temperature: Sampling temperature
        max_tokens: Maximum tokens requested
        has_json_schema: Whether JSON schema is used
        **extra_context: Additional context to log

    Returns:
        Start time for duration calculation
    """
    start_time = time.time()

    # Estimate tokens if not provided (rough: 1 token ≈ 4 chars)
    if prompt_tokens_estimate is None:
        prompt_tokens_estimate = (prompt_length + system_length) // 4

    logger.info(
        "llm_request_start",
        model=model,
        operation=operation,
        prompt_length=prompt_length,
        system_length=system_length,
        prompt_tokens_estimate=prompt_tokens_estimate,
        temperature=temperature,
        max_tokens=max_tokens,
        has_json_schema=has_json_schema,
        **extra_context,
    )

    return start_time


def _get_session_id() -> str:
    """Get current session ID for metrics tracking."""
    # Use a simple session identifier - could be enhanced with thread-local storage
    return "default"


def log_llm_success(
    model: str,
    operation: str,
    start_time: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    response_length: int,
    finish_reason: str = "stop",
    context_window: int | None = None,
    estimate_cost_flag: bool = True,
    session_id: str | None = None,
    **extra_context: Any,
) -> None:
    """Log successful LLM response with comprehensive metrics.

    Args:
        model: Model identifier
        operation: Operation name
        start_time: Request start time (from log_llm_request)
        prompt_tokens: Actual prompt tokens used
        completion_tokens: Actual completion tokens used
        total_tokens: Total tokens used
        response_length: Response length in characters
        finish_reason: Finish reason from API
        context_window: Model context window size (for usage calculation)
        estimate_cost_flag: Whether to estimate cost
        **extra_context: Additional context to log
    """
    duration = time.time() - start_time
    tokens_per_second = calculate_tokens_per_second(completion_tokens, duration)

    # Track slow requests (>60 seconds)
    is_slow = duration > 60.0
    if is_slow:
        logger.warning(
            "llm_slow_request",
            model=model,
            operation=operation,
            duration_seconds=round(duration, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            suggestion="Consider using a faster model or reducing input size",
        )

    log_data: dict[str, Any] = {
        "model": model,
        "operation": operation,
        "duration_seconds": round(duration, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_length": response_length,
        "finish_reason": finish_reason,
        "tokens_per_second": tokens_per_second,
        "is_slow": is_slow,
        **extra_context,
    }

    # Add cost estimation if enabled
    cost_info = {}
    if estimate_cost_flag:
        cost_info = estimate_cost(model, prompt_tokens, completion_tokens)
        log_data.update(cost_info)

        # Update session metrics
        sess_id = session_id or _get_session_id()
        metrics = _session_metrics[sess_id]
        metrics["total_cost"] += cost_info["total_cost_usd"]
        metrics["total_tokens"] += total_tokens
        metrics["total_requests"] += 1
        metrics["total_duration"] += duration
        if is_slow:
            metrics["slow_requests"] += 1

        # Per-model metrics
        model_metrics = metrics["by_model"][model]
        model_metrics["cost"] += cost_info["total_cost_usd"]
        model_metrics["tokens"] += total_tokens
        model_metrics["requests"] += 1
        model_metrics["duration"] += duration

    # Add context window usage if available
    if context_window:
        usage_info = calculate_context_window_usage(
            prompt_tokens, completion_tokens, context_window
        )
        log_data.update(usage_info)

        # Warn if near context limit
        if usage_info["is_near_limit"]:
            logger.warning(
                "llm_context_window_near_limit",
                model=model,
                operation=operation,
                usage_percent=usage_info["usage_percent"],
                remaining_tokens=usage_info["remaining_tokens"],
            )

    logger.info("llm_request_success", **log_data)

    # Log warning if response was truncated
    if finish_reason == "length":
        logger.warning(
            "llm_response_truncated",
            model=model,
            operation=operation,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            suggestion="Consider increasing max_tokens or reducing input size",
        )


def log_llm_retry(
    model: str,
    operation: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    error: Exception | None = None,
    **extra_context: Any,
) -> None:
    """Log LLM retry attempt.

    Args:
        model: Model identifier
        operation: Operation name
        attempt: Current attempt number
        max_attempts: Maximum attempts
        reason: Reason for retry
        error: Exception that triggered retry (if any)
        **extra_context: Additional context to log
    """
    logger.warning(
        "llm_retry_attempt",
        model=model,
        operation=operation,
        attempt=attempt,
        max_attempts=max_attempts,
        reason=reason,
        error=str(error) if error else None,
        error_type=type(error).__name__ if error else None,
        **extra_context,
    )


def log_llm_error(
    model: str,
    operation: str,
    start_time: float | None,
    error: Exception,
    error_type: str | None = None,
    status_code: int | None = None,
    retryable: bool = False,
    **extra_context: Any,
) -> None:
    """Log LLM error with context.

    Args:
        model: Model identifier
        operation: Operation name
        start_time: Request start time (if available)
        error: Exception that occurred
        error_type: Type of error (if known)
        status_code: HTTP status code (if applicable)
        retryable: Whether error is retryable
        **extra_context: Additional context to log
    """
    duration = time.time() - start_time if start_time else None

    log_data: dict[str, Any] = {
        "model": model,
        "operation": operation,
        "error": str(error),
        "error_type": error_type or type(error).__name__,
        "retryable": retryable,
        **extra_context,
    }

    if duration is not None:
        log_data["duration_seconds"] = round(duration, 3)

    if status_code:
        log_data["status_code"] = status_code

    logger.error("llm_request_error", **log_data)

    # Update error count in session metrics
    sess_id = _get_session_id()
    _session_metrics[sess_id]["errors"] += 1


def log_session_summary(session_id: str | None = None) -> None:
    """Log summary statistics for the current session.

    Args:
        session_id: Optional session ID (uses default if not provided)
    """
    sess_id = session_id or _get_session_id()
    metrics = _session_metrics[sess_id]

    if metrics["total_requests"] == 0:
        return  # No requests to summarize

    avg_duration = metrics["total_duration"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0
    avg_cost_per_request = metrics["total_cost"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0

    logger.info(
        "llm_session_summary",
        session_id=sess_id,
        total_requests=metrics["total_requests"],
        total_tokens=metrics["total_tokens"],
        total_cost_usd=round(metrics["total_cost"], 6),
        total_duration_seconds=round(metrics["total_duration"], 2),
        avg_duration_seconds=round(avg_duration, 2),
        avg_cost_per_request_usd=round(avg_cost_per_request, 6),
        slow_requests=metrics["slow_requests"],
        errors=metrics["errors"],
        models_used=list(metrics["by_model"].keys()),
    )

    # Log per-model breakdown
    for model, model_metrics in metrics["by_model"].items():
        model_avg_duration = model_metrics["duration"] / model_metrics["requests"] if model_metrics["requests"] > 0 else 0
        logger.info(
            "llm_model_summary",
            model=model,
            requests=model_metrics["requests"],
            tokens=model_metrics["tokens"],
            cost_usd=round(model_metrics["cost"], 6),
            avg_duration_seconds=round(model_avg_duration, 2),
        )


def reset_session_metrics(session_id: str | None = None) -> None:
    """Reset metrics for a session.

    Args:
        session_id: Optional session ID (uses default if not provided)
    """
    sess_id = session_id or _get_session_id()
    if sess_id in _session_metrics:
        del _session_metrics[sess_id]

