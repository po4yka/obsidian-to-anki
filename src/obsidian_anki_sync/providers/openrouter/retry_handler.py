import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from obsidian_anki_sync.utils.logging import get_logger

from .models import HTTP_STATUS_RETRYABLE

logger = get_logger(__name__)


def is_malformed_chat_completion(response_body: bytes) -> bool:
    """Check if a 200 response contains a malformed ChatCompletion.

    OpenRouter sometimes returns 200 OK but with null/error fields in the response.
    This function detects such cases to trigger retry.

    Args:
        response_body: Raw response body bytes

    Returns:
        True if response is malformed and should be retried
    """
    try:
        data = json.loads(response_body)

        # Check for error field (OpenRouter error format)
        if "error" in data:
            return True

        # Check for required ChatCompletion fields being null/missing
        # Per OpenAI spec, these are required for a valid response
        if data.get("id") is None:
            return True
        if data.get("choices") is None:
            return True
        if data.get("model") is None:
            return True
        return data.get("object") is None
    except (json.JSONDecodeError, TypeError, KeyError):
        # If we can't parse the response, it's malformed
        return True


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
        *args: object,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
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
                    try:
                        await response.aread()
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

                # Check for malformed 200 responses (OpenRouter sometimes returns
                # 200 OK with null/error fields instead of proper error status)
                if response.status_code == 200:
                    try:
                        await response.aread()
                    except Exception as exc:
                        logger.debug(
                            "openrouter_response_check_failed",
                            error=str(exc),
                            url=str(request.url),
                        )
                        with suppress(Exception):
                            await response.aclose()
                        raise

                    if is_malformed_chat_completion(response.content):
                        if retries >= self.max_retries:
                            # Exhausted retries, return the malformed response
                            # Let PydanticAI handle the error
                            logger.error(
                                "openrouter_malformed_response_exhausted",
                                attempt=retries + 1,
                                max_retries=self.max_retries,
                                url=str(request.url),
                                body_preview=response.text[:500],
                            )
                            return response

                        logger.warning(
                            "openrouter_malformed_response_retry",
                            status_code=200,
                            attempt=retries + 1,
                            max_retries=self.max_retries,
                            wait_time=round(delay, 2),
                            url=str(request.url),
                            body_preview=response.text[:500],
                        )

                        # Wait before retry
                        await asyncio.sleep(delay)

                        # Prepare for next attempt
                        retries += 1
                        delay *= self.backoff_factor

                        # Close the previous response
                        await response.aclose()

                        continue

                # For other status codes, return immediately
                return response

            except httpx.RequestError:
                # For network errors, retry if we haven't exhausted attempts
                if retries >= self.max_retries:
                    raise

                logger.warning(
                    "openrouter_request_error_retry",
                    attempt=retries + 1,
                    max_retries=self.max_retries,
                    wait_time=round(delay, 2),
                    url=str(request.url),
                )

                await asyncio.sleep(delay)
                retries += 1
                delay *= self.backoff_factor
