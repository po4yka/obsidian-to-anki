"""Retry logic and backoff calculation for OpenRouter API."""

from datetime import datetime, UTC
from email.utils import parsedate_to_datetime

import httpx

from .models import HTTP_STATUS_RETRYABLE


def calculate_retry_backoff(
    status_code: int,
    attempt: int,
    response: httpx.Response | None = None,
) -> float:
    """Calculate wait time before retrying based on status code and headers.

    Args:
        status_code: HTTP status code
        attempt: Current retry attempt number (0-indexed)
        response: HTTP response object (optional)

    Returns:
        Wait time in seconds
    """
    if status_code == 429 and response is not None:
        wait_seconds = parse_retry_after_header(response)
        if wait_seconds is not None:
            return wait_seconds
        # Default for 429 when Retry-After missing or invalid
        return min(60.0, float(2**attempt))

    # Generic exponential backoff for other retryable statuses
    return min(60.0, float(2**attempt))


def is_retryable_status(status_code: int) -> bool:
    """Check if HTTP status code should be retried.

    Args:
        status_code: HTTP status code

    Returns:
        True if status code is retryable
    """
    return status_code in HTTP_STATUS_RETRYABLE


def parse_retry_after_header(response: httpx.Response) -> float | None:
    """Parse Retry-After header from response.

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
                retry_datetime = retry_datetime.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            delta = (retry_datetime - now).total_seconds()
            if delta > 0:
                return float(delta)
        except Exception:
            pass

    return None
