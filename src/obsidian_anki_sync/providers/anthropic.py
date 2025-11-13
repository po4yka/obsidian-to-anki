"""Anthropic Claude provider implementation."""

import time
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider.

    Supports:
    - Claude 3 models (Opus, Sonnet, Haiku)
    - Claude 2 models
    - Messages API

    Configuration:
        api_key: Anthropic API key (required)
        base_url: API endpoint URL (default: https://api.anthropic.com)
        timeout: Request timeout in seconds (default: 120.0)
        max_retries: Maximum number of retries (default: 3)
        max_tokens: Maximum tokens to generate (default: 4096)
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = 120.0,
        max_retries: int = 3,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for Anthropic API
            api_version: API version to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration options
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            **kwargs,
        )

        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens

        # Set up headers
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )

        logger.info(
            "anthropic_provider_initialized",
            base_url=base_url,
            api_version=api_version,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def check_connection(self) -> bool:
        """Check if Anthropic API is accessible.

        Returns:
            True if Anthropic API is accessible, False otherwise

        Note:
            Anthropic doesn't have a health check endpoint, so we attempt
            a minimal API call to verify connectivity.
        """
        try:
            # Make a minimal request to check connectivity
            # Using a small model and minimal tokens
            test_payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}],
            }
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                json=test_payload,
            )
            return bool(
                response.status_code in (200, 400)
            )  # 400 means API is accessible
        except Exception as e:
            logger.error(
                "anthropic_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available Claude models.

        Returns:
            List of model identifiers

        Note:
            Anthropic doesn't provide a models endpoint, so we return
            a hardcoded list of known models.
        """
        models = [
            # Claude 3 models
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            # Claude 2 models
            "claude-2.1",
            "claude-2.0",
            # Claude Instant
            "claude-instant-1.2",
        ]
        return models

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate completion from Claude.

        Args:
            model: Model name (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for JSON mode - enforced via system prompt)
            stream: Enable streaming (not implemented)

        Returns:
            Response dictionary with 'response' and token usage info

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported")

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Build payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }

        # Add system prompt if provided
        if system:
            payload["system"] = system

        # For JSON mode, add instruction to system prompt
        if format == "json":
            json_instruction = "\n\nIMPORTANT: Respond with valid JSON only. Do not include any text before or after the JSON."
            if system:
                payload["system"] = system + json_instruction
            else:
                payload["system"] = json_instruction.strip()

        request_start_time = time.time()

        logger.info(
            "anthropic_generate_request",
            model=model,
            prompt_length=len(prompt),
            system_length=len(system),
            temperature=temperature,
            json_mode=format == "json",
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Retry logic
        last_exception: httpx.HTTPStatusError | httpx.RequestError | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_exception = e
                # Check for rate limiting or server errors
                if (
                    e.response.status_code in (429, 500, 502, 503, 504)
                    and attempt < self.max_retries - 1
                ):
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        "anthropic_retry",
                        attempt=attempt + 1,
                        status_code=e.response.status_code,
                        wait_time=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                raise
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "anthropic_retry_request_error",
                        attempt=attempt + 1,
                        error=str(e),
                        wait_time=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                raise
        else:
            # All retries failed
            if last_exception:
                raise last_exception
            raise RuntimeError("All retries failed")

        request_duration = time.time() - request_start_time

        try:
            data = response.json()

            # Extract response text from content blocks
            content = data.get("content", [])
            if not content:
                raise ValueError("No content in response")

            # Combine all text blocks
            response_text = ""
            for block in content:
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Extract token usage
            usage = data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            # Build result in standard format
            result = {
                "response": response_text,
                "model": data.get("model", model),
                "stop_reason": data.get("stop_reason", "end_turn"),
                "_token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            logger.info(
                "anthropic_generate_success",
                model=model,
                response_length=len(response_text),
                request_duration=round(request_duration, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                stop_reason=result["stop_reason"],
            )

            return cast(dict[str, Any], result)

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(
                "anthropic_parse_error",
                model=model,
                error=str(e),
                response_data=str(data) if "data" in locals() else "N/A",
                response_data_length=len(str(data)) if "data" in locals() else 0,
            )
            raise ValueError(f"Failed to parse Anthropic response: {e}")
        except Exception as e:
            request_duration = time.time() - request_start_time
            logger.error(
                "anthropic_unexpected_error",
                error=str(e),
                request_duration=round(request_duration, 2),
            )
            raise
