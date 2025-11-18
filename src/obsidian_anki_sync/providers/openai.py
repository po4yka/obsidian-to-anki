"""OpenAI provider implementation supporting GPT models."""

import time
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider supporting GPT models.

    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other OpenAI models
    - Function calling and JSON mode
    - Chat completions API

    Configuration:
        api_key: OpenAI API key (required)
        base_url: API endpoint URL (default: https://api.openai.com/v1)
        organization: Organization ID (optional)
        timeout: Request timeout in seconds (default: 120.0)
        max_retries: Maximum number of retries (default: 3)
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        organization: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for OpenAI API
            organization: Organization ID (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Additional configuration options
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Initialize HTTP client
        # Use synchronous httpx.Client for compatibility with existing sync code
        # This provider is used in sync contexts, so async client is not needed
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )

        logger.info(
            "openai_provider_initialized",
            base_url=base_url,
            timeout=timeout,
            has_organization=bool(organization),
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def check_connection(self) -> bool:
        """Check if OpenAI API is accessible.

        Returns:
            True if OpenAI API is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            return bool(response.status_code == 200)
        except Exception as e:
            logger.error(
                "openai_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available models from OpenAI.

        Returns:
            List of model IDs
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            logger.info("openai_list_models_success", model_count=len(models))
            return models
        except Exception as e:
            logger.error("openai_list_models_failed", error=str(e))
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
    ) -> dict[str, Any]:
        """Generate completion from OpenAI.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-2.0)
            format: Response format ("json" for JSON mode)
            json_schema: JSON schema for structured output (optional, not yet used)
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Enable JSON mode if requested
        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        request_start_time = time.time()

        logger.info(
            "openai_generate_request",
            model=model,
            prompt_length=len(prompt),
            system_length=len(system),
            temperature=temperature,
            json_mode=format == "json",
            timeout=self.timeout,
        )

        # Retry logic
        last_exception: httpx.HTTPStatusError | httpx.RequestError | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    # Retry on server errors
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        "openai_retry",
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
                        "openai_retry_request_error",
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

            # Extract response
            response_text = data["choices"][0]["message"]["content"]

            # Extract token usage
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Build result in standard format
            result = {
                "response": response_text,
                "model": data.get("model", model),
                "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                "_token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            logger.info(
                "openai_generate_success",
                model=model,
                response_length=len(response_text),
                request_duration=round(request_duration, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=result["finish_reason"],
            )

            return cast(dict[str, Any], result)

        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                "openai_parse_error",
                error=str(e),
                response_data=str(data)[:500],
            )
            raise ValueError(f"Failed to parse OpenAI response: {e}")
        except Exception as e:
            request_duration = time.time() - request_start_time
            logger.error(
                "openai_unexpected_error",
                error=str(e),
                request_duration=round(request_duration, 2),
            )
            raise
