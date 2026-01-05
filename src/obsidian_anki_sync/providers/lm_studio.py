"""LM Studio provider implementation."""

import json
from typing import Any, cast

import httpx

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseLLMProvider

logger = get_logger(__name__)


class LMStudioProvider(BaseLLMProvider):
    """LM Studio LLM provider using OpenAI-compatible API.

    LM Studio provides a local OpenAI-compatible API server for running
    LLMs on your machine. Default endpoint is http://localhost:1234/v1.

    Configuration:
        base_url: API endpoint URL (default: http://localhost:1234/v1)
        timeout: Request timeout in seconds (default: 1800.0)
        max_tokens: Maximum tokens in response (default: 2048)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 1800.0,
        max_tokens: int = 2048,
        verbose_logging: bool = False,
        **kwargs: Any,
    ):
        """Initialize LM Studio provider.

        Args:
            base_url: Base URL for LM Studio API (OpenAI-compatible)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            verbose_logging: Whether to log detailed initialization info
            **kwargs: Additional configuration options
        """
        super().__init__(
            verbose_logging=verbose_logging,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Initialize HTTP client
        # Use synchronous httpx.Client for compatibility with existing sync code
        # This provider is used in sync contexts, so async client is not needed
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
            return bool(response.status_code == 200)
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
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Generate completion from LM Studio.

        Args:
            model: Model identifier (use the loaded model ID from LM Studio)
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            json_schema: JSON schema for structured output (optional, not yet used)
            stream: Enable streaming (not implemented)

        Returns:
            Response dictionary with 'response' key containing the completion

        Raises:
            NotImplementedError: If streaming is requested
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            msg = "Streaming is not yet supported"
            raise NotImplementedError(msg)

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

            # Validate response structure
            choices = result.get("choices", [])
            if not choices:
                msg = f"LM Studio returned empty choices array. Response: {str(result)[:500]}"
                raise ValueError(msg)

            # Extract completion from OpenAI format safely
            first_choice = choices[0]
            message = first_choice.get("message", {})
            completion = message.get("content", "")
            finish_reason = first_choice.get("finish_reason")

            # Convert to standard format
            standardized_result = {
                "response": completion,
                "model": result.get("model"),
                "finish_reason": finish_reason,
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
            # Parse response for structured error details if available
            error_details = {}
            try:
                error_json = e.response.json()
                error_details = error_json
            except (json.JSONDecodeError, ValueError):
                error_details = {"raw_response": e.response.text}

            logger.error(
                "lm_studio_http_error",
                status_code=e.response.status_code,
                model=model,
                prompt_length=len(prompt),
                error=str(e),
                error_details=error_details,
            )
            raise
        except httpx.RequestError as e:
            logger.error("lm_studio_request_error", error=str(e))
            raise
        except Exception as e:
            logger.error("lm_studio_unexpected_error", error=str(e))
            raise

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        json_schema: dict[str, Any] | None = None,
        reasoning_enabled: bool = False,
    ) -> dict[str, Any]:
        """Generate a JSON response from LM Studio.

        Overrides base implementation to use OpenAI's response_format parameter.

        Args:
            model: Model identifier
            prompt: User prompt (should request JSON format)
            system: System prompt (optional)
            temperature: Sampling temperature
            json_schema: JSON schema for structured output (not used by LM Studio)
            reasoning_enabled: Enable reasoning mode (not used by LM Studio)

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
            json_schema=json_schema,
            reasoning_enabled=reasoning_enabled,
        )

        response_text = result.get("response", "{}")
        try:
            return cast("dict[str, Any]", json.loads(response_text))
        except json.JSONDecodeError as e:
            logger.error(
                "lm_studio_json_parse_error",
                model=model,
                error=str(e),
                error_position=(
                    f"line {e.lineno}, col {e.colno}"
                    if hasattr(e, "lineno")
                    else "unknown"
                ),
                response_text=response_text,
                response_length=len(response_text),
            )
            raise
