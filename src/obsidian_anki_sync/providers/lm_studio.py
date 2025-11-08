"""LM Studio provider implementation."""

import json
from typing import Any, cast

import httpx

from ..utils.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class LMStudioProvider(BaseLLMProvider):
    """LM Studio LLM provider using OpenAI-compatible API.

    LM Studio provides a local OpenAI-compatible API server for running
    LLMs on your machine. Default endpoint is http://localhost:1234/v1.

    Configuration:
        base_url: API endpoint URL (default: http://localhost:1234/v1)
        timeout: Request timeout in seconds (default: 600.0)
        max_tokens: Maximum tokens in response (default: 2048)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 600.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """Initialize LM Studio provider.

        Args:
            base_url: Base URL for LM Studio API (OpenAI-compatible)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            **kwargs: Additional configuration options
        """
        super().__init__(
            base_url=base_url, timeout=timeout, max_tokens=max_tokens, **kwargs
        )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Initialize HTTP client
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        logger.info(
            "lm_studio_provider_initialized",
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def check_connection(self) -> bool:
        """Check if LM Studio is running and accessible.

        Returns:
            True if LM Studio is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(
                "lm_studio_connection_check_failed",
                base_url=self.base_url,
                error=str(e),
            )
            return False

    def list_models(self) -> list[str]:
        """List available models in LM Studio.

        Returns:
            List of model identifiers
        """
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()

            # OpenAI API format: {"data": [{"id": "model-name"}, ...]}
            models = [model["id"] for model in data.get("data", [])]
            logger.info("lm_studio_list_models_success", model_count=len(models))
            return models
        except Exception as e:
            logger.error("lm_studio_list_models_failed", error=str(e))
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate completion from LM Studio.

        Args:
            model: Model identifier (use the loaded model ID from LM Studio)
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
            "lm_studio_generate_request",
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
                "lm_studio_generate_success",
                model=model,
                response_length=len(completion),
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
            )

            return standardized_result

        except httpx.HTTPStatusError as e:
            logger.error(
                "lm_studio_http_error",
                status_code=e.response.status_code,
                error=str(e),
                response_text=e.response.text[:500],
            )
            raise
        except httpx.RequestError as e:
            logger.error("lm_studio_request_error", error=str(e))
            raise
        except Exception as e:
            logger.error("lm_studio_unexpected_error", error=str(e))
            raise

    def generate_json(
        self, model: str, prompt: str, system: str = "", temperature: float = 0.7
    ) -> dict[str, Any]:
        """Generate a JSON response from LM Studio.

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
                "lm_studio_json_parse_error",
                error=str(e),
                response_text=response_text[:500],
            )
            raise
