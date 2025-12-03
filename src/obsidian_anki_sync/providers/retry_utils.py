"""Shared retry utilities for LLM providers.

Provides consistent retry logic with Retry-After header parsing
and exponential backoff with jitter for rate limit handling.
"""

import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

# HTTP status codes that should be retried
HTTP_STATUS_RETRYABLE = {429, 500, 502, 503, 504}


def is_retryable_status(status_code: int) -> bool:
    """Check if HTTP status code should be retried.

    Args:
        status_code: HTTP status code

    Returns:
        True if status code is retryable (429, 5xx)
    """
    return status_code in HTTP_STATUS_RETRYABLE


def parse_retry_after_header(response: httpx.Response) -> float | None:
    """Parse Retry-After header from response.

    Supports both numeric seconds and HTTP date formats.

    Args:
        response: HTTP response object

    Returns:
        Wait time in seconds, or None if header is missing/invalid
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        # Try to parse as seconds (numeric value)
        wait_seconds = float(retry_after)
        if wait_seconds > 0:
            return wait_seconds
    except ValueError:
        # Try to parse as HTTP date
        try:
            retry_datetime = parsedate_to_datetime(retry_after)
            if retry_datetime.tzinfo is None:
                retry_datetime = retry_datetime.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (retry_datetime - now).total_seconds()
            if delta > 0:
                return float(delta)
        except Exception:
            pass

    return None


def calculate_retry_wait(
    status_code: int,
    attempt: int,
    response: httpx.Response | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate wait time before retrying based on status code and headers.

    For 429 (rate limit), respects Retry-After header if present.
    For other retryable errors, uses exponential backoff with optional jitter.

    Args:
        status_code: HTTP status code
        attempt: Current retry attempt number (0-indexed)
        response: HTTP response object (optional, for Retry-After header)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Whether to add jitter to prevent thundering herd

    Returns:
        Wait time in seconds
    """
    # For rate limiting, prefer Retry-After header
    if status_code == 429 and response is not None:
        retry_after = parse_retry_after_header(response)
        if retry_after is not None:
            # Add small jitter to Retry-After to prevent thundering herd
            if jitter:
                retry_after += random.uniform(0.1, 1.0)
            return min(retry_after, max_delay)

    # Exponential backoff: base_delay * 2^attempt
    delay = base_delay * (2**attempt)

    # Add jitter if enabled (random 0-25% of delay)
    if jitter:
        jitter_amount = delay * random.uniform(0, 0.25)
        delay += jitter_amount

    return float(min(delay, max_delay))


def log_retry(
    provider_name: str,
    attempt: int,
    max_retries: int,
    status_code: int | None,
    wait_time: float,
    error: str | None = None,
) -> None:
    """Log retry attempt with consistent format.

    Args:
        provider_name: Name of the provider (openai, anthropic, etc.)
        attempt: Current attempt number (1-indexed for display)
        max_retries: Maximum retries configured
        status_code: HTTP status code (if available)
        wait_time: Wait time before retry
        error: Error message (if available)
    """
    log_data: dict[str, Any] = {
        "attempt": attempt,
        "max_retries": max_retries,
        "wait_seconds": round(wait_time, 2),
    }
    if status_code is not None:
        log_data["status_code"] = status_code
    if error is not None:
        log_data["error"] = error

    logger.warning(f"{provider_name}_retry", **log_data)
