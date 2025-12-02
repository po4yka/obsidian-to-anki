import asyncio
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from obsidian_anki_sync.utils.logging import get_logger

from .models import HTTP_STATUS_RETRYABLE

logger = get_logger(__name__)


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


class RetryTransport(httpx.AsyncHTTPTransport):
    """Custom transport to handle 429 retries transparently.

    This is used to inject retry logic into clients that don't support it natively
    or where we need consistent behavior (like PydanticAI integration).
    """

    def __init__(
        self,
        *args,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        retries = 0
        delay = self.initial_delay

        while True:
            try:
                # Call the parent implementation to perform the request
                response = await super().handle_async_request(request)

                # Check for 429 Rate Limit
                if response.status_code == 429:
                    if retries >= self.max_retries:
                        # Exhausted retries, return the 429 response
                        return response

                    # Calculate wait time
                    wait_time = parse_retry_after_header(response)
                    if wait_time is None:
                        wait_time = delay

                    # Read response body to log error details if available
                    # We need to be careful not to consume the stream if it's not already read
                    # But handle_async_request returns a response where we can read content
                    # For logging purposes, we peek at it.
                    # Note: PydanticAI/httpx might expect to read the response.
                    # If we read it here, we should ensure it's still available.
                    # httpx response.read() caches content.
                    try:
                        await response.read()
                        error_text = response.text[:500]
                    except Exception:
                        error_text = "(could not read response body)"

                    logger.warning(
                        "openrouter_rate_limit_retry_transport",
                        status_code=429,
                        attempt=retries + 1,
                        max_retries=self.max_retries,
                        wait_time=round(wait_time, 2),
                        url=str(request.url),
                        body_preview=error_text,
                    )

                    # Wait
                    await asyncio.sleep(wait_time)

                    # Prepare for next attempt
                    retries += 1
                    delay *= self.backoff_factor

                    # Close the previous response
                    await response.aclose()

                    continue

                # For other status codes, return immediately
                return response

            except Exception:
                # Let other exceptions bubble up
                raise
