"""LLM error handling utilities for agents."""

from enum import Enum
from typing import Any

import httpx

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMErrorType(Enum):
    """Categories of LLM errors for better handling."""

    TIMEOUT = "timeout"
    NETWORK = "network"
    MODEL_NOT_FOUND = "model_not_found"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"
    CONTEXT_LENGTH = "context_length"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


class LLMError(Exception):
    """Enhanced LLM error with categorization and context."""

    def __init__(
        self,
        message: str,
        error_type: LLMErrorType,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
        retryable: bool = False,
    ):
        """Initialize LLM error.

        Args:
            message: Human-readable error message
            error_type: Category of error
            original_error: Original exception if any
            context: Additional context (model, prompt length, etc.)
            retryable: Whether this error is safe to retry
        """
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error
        self.context = context or {}
        self.retryable = retryable


def categorize_llm_error(
    error: Exception, model: str, operation: str, duration: float
) -> LLMError:
    """Categorize an LLM error for better handling.

    Args:
        error: Original exception
        model: Model being used
        operation: Operation being performed (e.g., "generation", "validation")
        duration: How long the operation took before failing

    Returns:
        Categorized LLMError
    """
    context = {"model": model, "operation": operation, "duration": round(duration, 2)}

    # Timeout errors
    if isinstance(
        error, (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout)
    ):
        return LLMError(
            message=f"LLM request timed out after {duration:.1f}s for {operation}",
            error_type=LLMErrorType.TIMEOUT,
            original_error=error,
            context=context,
            retryable=True,
        )

    # Network errors
    if isinstance(error, (httpx.NetworkError, httpx.ConnectError)):
        return LLMError(
            message=f"Network error during {operation}: {str(error)}",
            error_type=LLMErrorType.NETWORK,
            original_error=error,
            context=context,
            retryable=True,
        )

    # HTTP status errors
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        response_text = error.response.text

        # Model not found (404)
        if status_code == 404:
            return LLMError(
                message=f"Model '{model}' not found. Pull it with: ollama pull {model}",
                error_type=LLMErrorType.MODEL_NOT_FOUND,
                original_error=error,
                context={**context, "status_code": status_code},
                retryable=False,
            )

        # Rate limiting (429)
        if status_code == 429:
            return LLMError(
                message=f"Rate limited during {operation}. Try again in a moment.",
                error_type=LLMErrorType.RATE_LIMIT,
                original_error=error,
                context={**context, "status_code": status_code},
                retryable=True,
            )

        # Context length exceeded (usually 400 with specific message)
        if status_code == 400 and (
            "context length" in response_text.lower()
            or "too long" in response_text.lower()
        ):
            return LLMError(
                message=f"Prompt exceeds model's context length for {operation}",
                error_type=LLMErrorType.CONTEXT_LENGTH,
                original_error=error,
                context={**context, "status_code": status_code},
                retryable=False,
            )

        # Server errors (5xx) - retryable
        if 500 <= status_code < 600:
            return LLMError(
                message=f"Server error (HTTP {status_code}) during {operation}",
                error_type=LLMErrorType.SERVER_ERROR,
                original_error=error,
                context={**context, "status_code": status_code},
                retryable=True,
            )

    # JSON parsing errors or empty responses
    if isinstance(error, ValueError):
        error_str = str(error).lower()
        if "empty" in error_str or "no json" in error_str:
            return LLMError(
                message=f"LLM returned empty or invalid response for {operation}",
                error_type=LLMErrorType.EMPTY_RESPONSE,
                original_error=error,
                context=context,
                retryable=True,  # Sometimes retryable
            )

    # Unknown/uncategorized error
    return LLMError(
        message=f"Unexpected error during {operation}: {str(error)}",
        error_type=LLMErrorType.UNKNOWN,
        original_error=error,
        context=context,
        retryable=False,
    )


def log_llm_error(error: LLMError, **extra_context: Any) -> None:
    """Log an LLM error with appropriate level and context.

    Args:
        error: The LLM error to log
        **extra_context: Additional context to include in log
    """
    full_context = {**error.context, **extra_context}

    # Critical errors (non-retryable, likely user action needed)
    if error.error_type in (LLMErrorType.MODEL_NOT_FOUND, LLMErrorType.CONTEXT_LENGTH):
        logger.error(
            "llm_critical_error",
            error_type=error.error_type.value,
            message=str(error),
            retryable=error.retryable,
            **full_context,
        )

    # Transient errors (retryable)
    elif error.error_type in (
        LLMErrorType.TIMEOUT,
        LLMErrorType.NETWORK,
        LLMErrorType.RATE_LIMIT,
        LLMErrorType.SERVER_ERROR,
    ):
        logger.warning(
            "llm_transient_error",
            error_type=error.error_type.value,
            message=str(error),
            retryable=error.retryable,
            **full_context,
        )

    # Other errors
    else:
        logger.error(
            "llm_error",
            error_type=error.error_type.value,
            message=str(error),
            retryable=error.retryable,
            **full_context,
        )


def should_retry_llm_error(error: LLMError, attempt: int, max_attempts: int) -> bool:
    """Determine if an LLM error should be retried.

    Args:
        error: The LLM error
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts allowed

    Returns:
        True if should retry, False otherwise
    """
    # Don't retry if we've exhausted attempts
    if attempt >= max_attempts:
        logger.info(
            "max_retry_attempts_reached",
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=error.error_type.value,
        )
        return False

    # Only retry if error is marked as retryable
    if not error.retryable:
        logger.info(
            "error_not_retryable",
            error_type=error.error_type.value,
            reason="Error type is not safe to retry",
        )
        return False

    logger.info(
        "retrying_llm_operation",
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=error.error_type.value,
    )
    return True


def format_llm_error_for_user(error: LLMError) -> str:
    """Format an LLM error into a user-friendly message.

    Args:
        error: The LLM error

    Returns:
        User-friendly error message with actionable guidance
    """
    base_msg = str(error)

    # Add specific guidance based on error type
    if error.error_type == LLMErrorType.MODEL_NOT_FOUND:
        model = error.context.get("model", "unknown")
        return f"{base_msg}\n\nTo fix: Run 'ollama pull {model}' to download the model."

    if error.error_type == LLMErrorType.TIMEOUT:
        return (
            f"{base_msg}\n\n"
            "The model is taking too long to respond. This could be due to:\n"
            "- Large prompt size\n"
            "- Slow hardware\n"
            "- Model loading time (first request)\n\n"
            "Consider using a smaller/faster model or increasing the timeout."
        )

    if error.error_type == LLMErrorType.CONTEXT_LENGTH:
        return (
            f"{base_msg}\n\n"
            "The input is too long for this model. Try:\n"
            "- Using a model with larger context window\n"
            "- Reducing note size\n"
            "- Processing in smaller batches"
        )

    if error.error_type == LLMErrorType.NETWORK:
        return (
            f"{base_msg}\n\n"
            "Cannot reach the LLM service. Check:\n"
            "- Is Ollama running? (ollama list)\n"
            "- Network connectivity\n"
            "- Base URL configuration"
        )

    # Default message
    return base_msg
