"""Atomic API calls for OpenRouter.

This module provides individual, testable API call functions for OpenRouter.
Each function is atomic and can be tested independently.

Usage:
    from obsidian_anki_sync.providers.openrouter.api_calls import (
        check_connection,
        list_models,
        fetch_key_status,
        chat_completion,
        chat_completion_with_tools,
    )

    # Test connection
    result = check_connection(client, base_url)

    # List available models
    models = list_models(client, base_url)

    # Generate completion
    response = chat_completion(
        client=client,
        base_url=base_url,
        headers=headers,
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from dataclasses import dataclass, field
from typing import Any

import httpx

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class APICallResult:
    """Result of an atomic API call.

    Attributes:
        success: Whether the call succeeded
        data: Response data (if successful)
        error: Error message (if failed)
        status_code: HTTP status code
        latency_ms: Request latency in milliseconds
        raw_response: Raw response object for debugging
    """

    success: bool
    data: Any = None
    error: str | None = None
    status_code: int | None = None
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


@dataclass
class ChatCompletionResult(APICallResult):
    """Result of a chat completion API call.

    Attributes:
        content: Generated text content
        finish_reason: Why generation stopped (stop, length, etc.)
        model: Model that was used
        usage: Token usage statistics
        tool_calls: Tool calls if any were made
    """

    content: str = ""
    finish_reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def check_connection(
    client: httpx.Client,
    base_url: str,
    timeout: float = 10.0,
) -> APICallResult:
    """Check if OpenRouter API is accessible.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        timeout: Request timeout in seconds

    Returns:
        APICallResult with success status
    """
    import time

    start_time = time.perf_counter()

    try:
        response = client.get(
            f"{base_url}/models",
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        success = response.status_code == 200

        logger.debug(
            "openrouter_connection_check",
            success=success,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        return APICallResult(
            success=success,
            status_code=response.status_code,
            latency_ms=latency_ms,
            data={"accessible": success},
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "openrouter_connection_timeout",
            timeout=timeout,
            error=str(e),
        )
        return APICallResult(
            success=False,
            error=f"Connection timeout after {timeout}s",
            latency_ms=latency_ms,
        )

    except httpx.RequestError as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "openrouter_connection_error",
            error=str(e),
        )
        return APICallResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
        )


def list_models(
    client: httpx.Client,
    base_url: str,
    timeout: float = 30.0,
) -> APICallResult:
    """List available models from OpenRouter.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        timeout: Request timeout in seconds

    Returns:
        APICallResult with list of model IDs in data field
    """
    import time

    start_time = time.perf_counter()

    try:
        response = client.get(
            f"{base_url}/models",
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return APICallResult(
                success=False,
                error=f"HTTP {response.status_code}",
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        data = response.json()
        models = [model["id"] for model in data.get("data", [])]

        logger.info(
            "openrouter_list_models",
            model_count=len(models),
            latency_ms=round(latency_ms, 2),
        )

        return APICallResult(
            success=True,
            data={"models": models, "count": len(models)},
            status_code=200,
            latency_ms=latency_ms,
            raw_response=data,
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return APICallResult(
            success=False,
            error=f"Timeout after {timeout}s: {e}",
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("openrouter_list_models_error", error=str(e))
        return APICallResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
        )


def fetch_key_status(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    timeout: float = 10.0,
) -> APICallResult:
    """Fetch API key status and account information.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        headers: Request headers (must include Authorization)
        timeout: Request timeout in seconds

    Returns:
        APICallResult with key status information
    """
    import time

    start_time = time.perf_counter()

    try:
        response = client.get(
            f"{base_url}/auth/key",
            headers=headers,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            return APICallResult(
                success=False,
                error=f"HTTP {response.status_code}",
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        data = response.json()

        logger.debug(
            "openrouter_key_status",
            has_data=bool(data.get("data")),
            latency_ms=round(latency_ms, 2),
        )

        return APICallResult(
            success=True,
            data=data.get("data", data),
            status_code=200,
            latency_ms=latency_ms,
            raw_response=data,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("openrouter_key_status_error", error=str(e))
        return APICallResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
        )


def chat_completion(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> ChatCompletionResult:
    """Make a chat completion API call.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        headers: Request headers (must include Authorization)
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        messages: List of message dicts with role and content
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        response_format: Optional response format (e.g., {"type": "json_object"})
        timeout: Request timeout in seconds

    Returns:
        ChatCompletionResult with generated content
    """
    import time

    start_time = time.perf_counter()

    # Build payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if response_format is not None:
        payload["response_format"] = response_format

    try:
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.warning(
                "openrouter_chat_completion_error",
                status_code=response.status_code,
                error=error_text,
                model=model,
            )
            return ChatCompletionResult(
                success=False,
                error=f"HTTP {response.status_code}: {error_text}",
                status_code=response.status_code,
                latency_ms=latency_ms,
                model=model,
            )

        data = response.json()

        # Extract response content
        choices = data.get("choices", [])
        if not choices:
            return ChatCompletionResult(
                success=False,
                error="No choices in response",
                status_code=200,
                latency_ms=latency_ms,
                model=model,
                raw_response=data,
            )

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "")

        usage = data.get("usage", {})

        logger.info(
            "openrouter_chat_completion",
            model=data.get("model", model),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round(latency_ms, 2),
        )

        return ChatCompletionResult(
            success=True,
            data=data,
            status_code=200,
            latency_ms=latency_ms,
            content=content,
            finish_reason=finish_reason,
            model=data.get("model", model),
            usage=usage,
            raw_response=data,
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "openrouter_chat_completion_timeout",
            model=model,
            timeout=timeout,
        )
        return ChatCompletionResult(
            success=False,
            error=f"Timeout after {timeout}s: {e}",
            latency_ms=latency_ms,
            model=model,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "openrouter_chat_completion_exception",
            model=model,
            error=str(e),
        )
        return ChatCompletionResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
            model=model,
        )


def chat_completion_with_tools(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: float = 120.0,
) -> ChatCompletionResult:
    """Make a chat completion API call with tool/function calling.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        headers: Request headers (must include Authorization)
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        messages: List of message dicts with role and content
        tools: List of tool definitions (OpenAI function calling format)
        tool_choice: Tool selection mode ("auto", "none", "required", or specific)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        ChatCompletionResult with tool_calls populated if tools were called
    """
    import time

    from .retry_handler import sanitize_tool_calls

    start_time = time.perf_counter()

    # Build payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.warning(
                "openrouter_tool_completion_error",
                status_code=response.status_code,
                error=error_text,
                model=model,
            )
            return ChatCompletionResult(
                success=False,
                error=f"HTTP {response.status_code}: {error_text}",
                status_code=response.status_code,
                latency_ms=latency_ms,
                model=model,
            )

        data = response.json()

        # Sanitize tool calls (fix None arguments)
        data, was_sanitized = sanitize_tool_calls(data)
        if was_sanitized:
            logger.info(
                "openrouter_tool_calls_sanitized",
                model=model,
            )

        # Extract response content
        choices = data.get("choices", [])
        if not choices:
            return ChatCompletionResult(
                success=False,
                error="No choices in response",
                status_code=200,
                latency_ms=latency_ms,
                model=model,
                raw_response=data,
            )

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "") or ""
        finish_reason = first_choice.get("finish_reason", "")
        tool_calls = message.get("tool_calls", [])

        usage = data.get("usage", {})

        logger.info(
            "openrouter_tool_completion",
            model=data.get("model", model),
            finish_reason=finish_reason,
            tool_calls_count=len(tool_calls),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round(latency_ms, 2),
        )

        return ChatCompletionResult(
            success=True,
            data=data,
            status_code=200,
            latency_ms=latency_ms,
            content=content,
            finish_reason=finish_reason,
            model=data.get("model", model),
            usage=usage,
            tool_calls=tool_calls,
            raw_response=data,
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "openrouter_tool_completion_timeout",
            model=model,
            timeout=timeout,
        )
        return ChatCompletionResult(
            success=False,
            error=f"Timeout after {timeout}s: {e}",
            latency_ms=latency_ms,
            model=model,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "openrouter_tool_completion_exception",
            model=model,
            error=str(e),
        )
        return ChatCompletionResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
            model=model,
        )


def chat_completion_structured(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, str]],
    json_schema: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = 120.0,
) -> ChatCompletionResult:
    """Make a chat completion API call with structured JSON output.

    Uses OpenRouter's response_format with json_schema for guaranteed
    JSON structure compliance.

    Args:
        client: HTTP client instance
        base_url: OpenRouter API base URL
        headers: Request headers (must include Authorization)
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        messages: List of message dicts with role and content
        json_schema: JSON schema for response format
        temperature: Sampling temperature (0.0 recommended for structured output)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        ChatCompletionResult with content as JSON string
    """
    import json
    import time

    start_time = time.perf_counter()

    # Build payload with response_format
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        response = client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.warning(
                "openrouter_structured_completion_error",
                status_code=response.status_code,
                error=error_text,
                model=model,
            )
            return ChatCompletionResult(
                success=False,
                error=f"HTTP {response.status_code}: {error_text}",
                status_code=response.status_code,
                latency_ms=latency_ms,
                model=model,
            )

        data = response.json()

        # Extract response content
        choices = data.get("choices", [])
        if not choices:
            return ChatCompletionResult(
                success=False,
                error="No choices in response",
                status_code=200,
                latency_ms=latency_ms,
                model=model,
                raw_response=data,
            )

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "")

        # Validate JSON
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(
                "openrouter_structured_invalid_json",
                model=model,
                error=str(e),
                content_preview=content[:200],
            )
            return ChatCompletionResult(
                success=False,
                error=f"Invalid JSON response: {e}",
                status_code=200,
                latency_ms=latency_ms,
                model=model,
                content=content,
                raw_response=data,
            )

        usage = data.get("usage", {})

        logger.info(
            "openrouter_structured_completion",
            model=data.get("model", model),
            finish_reason=finish_reason,
            content_length=len(content),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round(latency_ms, 2),
        )

        return ChatCompletionResult(
            success=True,
            data=data,
            status_code=200,
            latency_ms=latency_ms,
            content=content,
            finish_reason=finish_reason,
            model=data.get("model", model),
            usage=usage,
            raw_response=data,
        )

    except httpx.TimeoutException as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "openrouter_structured_completion_timeout",
            model=model,
            timeout=timeout,
        )
        return ChatCompletionResult(
            success=False,
            error=f"Timeout after {timeout}s: {e}",
            latency_ms=latency_ms,
            model=model,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "openrouter_structured_completion_exception",
            model=model,
            error=str(e),
        )
        return ChatCompletionResult(
            success=False,
            error=str(e),
            latency_ms=latency_ms,
            model=model,
        )


# Convenience function to create a configured client
def create_openrouter_client(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    site_url: str | None = None,
    site_name: str | None = None,
    timeout: float = 120.0,
) -> tuple[httpx.Client, dict[str, str]]:
    """Create an HTTP client and headers for OpenRouter API calls.

    Args:
        api_key: OpenRouter API key
        base_url: API base URL
        site_url: Optional site URL for rankings
        site_name: Optional site name for rankings
        timeout: Default request timeout

    Returns:
        Tuple of (client, headers)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    client = httpx.Client(
        timeout=httpx.Timeout(timeout, connect=30.0),
        limits=httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
            keepalive_expiry=30.0,
        ),
    )

    return client, headers
