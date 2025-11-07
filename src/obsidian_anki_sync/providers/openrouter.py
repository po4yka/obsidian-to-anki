"""OpenRouter provider implementation."""

import json
import os
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from ..utils.retry import retry
from .base import BaseLLMProvider

logger = get_logger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible API.

    OpenRouter provides access to multiple LLM providers through a unified API.
    Requires an API key from https://openrouter.ai/

    Configuration:
        api_key: OpenRouter API key (required, can use OPENROUTER_API_KEY env var)
        base_url: API endpoint URL (default: https://openrouter.ai/api/v1)
        timeout: Request timeout in seconds (default: 180.0)
        max_tokens: Maximum tokens in response (default: 2048)
        site_url: Your site URL for OpenRouter rankings (optional)
        site_name: Your site name for OpenRouter rankings (optional)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
        max_tokens: int = 2048,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (if not provided, uses OPENROUTER_API_KEY env var)
            base_url: Base URL for OpenRouter API
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            site_url: Your site URL (optional, for rankings)
            site_name: Your site name (optional, for rankings)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it via the api_key parameter "
                "or set the OPENROUTER_API_KEY environment variable."
            )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            site_url=site_url,
            site_name=site_name,
            **kwargs,
        )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.site_url = site_url
        self.site_name = site_name

        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add optional site information for OpenRouter rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers=headers,
        )

        logger.info(
            "openrouter_provider_initialized",
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            has_site_info=bool(site_url and site_name),
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def check_connection(self) -> bool:
        """Check if OpenRouter is accessible.

        Returns:
            True if OpenRouter is accessible, False otherwise
        """
        try:
            # OpenRouter doesn't have a dedicated health endpoint,
            # so we try to list models as a health check
            response = self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(
                "openrouter_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available models from OpenRouter.

        Returns:
            List of model identifiers (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()

            # OpenRouter API format: {"data": [{"id": "model-name"}, ...]}
            models = [model["id"] for model in data.get("data", [])]
            logger.info("openrouter_list_models_success", model_count=len(models))
            return models
        except Exception as e:
            logger.error("openrouter_list_models_failed", error=str(e))
            return []

    @retry(max_attempts=3, initial_delay=2.0, backoff_factor=2.0)
    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate completion from OpenRouter.

        Args:
            model: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            stream: Enable streaming (not implemented)

        Returns:
            Response dictionary with 'response' key containing the completion

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported")

        # Build messages in OpenAI format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        # Handle JSON format request
        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "openrouter_generate_request",
            model=model,
            prompt_length=len(prompt),
            system_length=len(system),
            temperature=temperature,
            format=format,
        )

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Extract completion from OpenAI format
            completion = result["choices"][0]["message"]["content"]

            # Convert to standard format
            standardized_result = {
                "response": completion,
                "model": result.get("model"),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "usage": result.get("usage", {}),
            }

            logger.info(
                "openrouter_generate_success",
                model=model,
                response_length=len(completion),
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
            )

            return standardized_result

        except httpx.HTTPStatusError as e:
            logger.error(
                "openrouter_http_error",
                status_code=e.response.status_code,
                error=str(e),
                response_text=e.response.text[:500],
            )
            raise
        except httpx.RequestError as e:
            logger.error("openrouter_request_error", error=str(e))
            raise
        except Exception as e:
            logger.error("openrouter_unexpected_error", error=str(e))
            raise

    def generate_json(
        self, model: str, prompt: str, system: str = "", temperature: float = 0.7
    ) -> dict[str, Any]:
        """Generate a JSON response from OpenRouter.

        Overrides base implementation to use OpenAI's response_format parameter.

        Args:
            model: Model identifier
            prompt: User prompt (should request JSON format)
            system: System prompt (optional)
            temperature: Sampling temperature

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        result = self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format="json",
        )

        response_text = result.get("response", "{}")
        try:
            return cast(dict[str, Any], json.loads(response_text))
        except json.JSONDecodeError as e:
            logger.error(
                "openrouter_json_parse_error",
                error=str(e),
                response_text=response_text[:500],
            )
            raise
