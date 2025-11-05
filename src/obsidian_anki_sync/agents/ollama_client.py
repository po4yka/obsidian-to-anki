"""Ollama client wrapper for local LLM inference."""

import json
from typing import Any

import httpx

from ..utils.logging import get_logger
from ..utils.retry import retry

logger = get_logger(__name__)


class OllamaClient:
    """Client for Ollama API with retry logic and connection pooling.

    This client provides a simple interface to Ollama's local LLM inference,
    optimized for Apple Silicon with MLX models.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        """Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        logger.info("ollama_client_initialized", base_url=base_url, timeout=timeout)

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            self.client.close()

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error("ollama_connection_check_failed", error=str(e))
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error("ollama_list_models_failed", error=str(e))
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
        """Generate completion from Ollama.

        Args:
            model: Model name (e.g., "qwen3:8b", "qwen3:32b")
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            stream: Enable streaming (not implemented)

        Returns:
            Response dictionary with 'response' and 'context' keys

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        if stream:
            raise NotImplementedError("Streaming is not yet supported")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if system:
            payload["system"] = system

        if format == "json":
            payload["format"] = "json"

        logger.info(
            "ollama_generate_request",
            model=model,
            prompt_length=len(prompt),
            system_length=len(system),
            temperature=temperature,
            format=format,
        )

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            logger.info(
                "ollama_generate_success",
                model=model,
                response_length=len(result.get("response", "")),
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                "ollama_http_error",
                status_code=e.response.status_code,
                error=str(e),
                response_text=e.response.text[:500],
            )
            raise
        except httpx.RequestError as e:
            logger.error("ollama_request_error", error=str(e))
            raise
        except Exception as e:
            logger.error("ollama_unexpected_error", error=str(e))
            raise

    def generate_json(
        self, model: str, prompt: str, system: str = "", temperature: float = 0.7
    ) -> dict[str, Any]:
        """Generate JSON response from Ollama.

        This is a convenience method that ensures JSON format and parses the result.

        Args:
            model: Model name
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature

        Returns:
            Parsed JSON response

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
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(
                "ollama_json_parse_error",
                error=str(e),
                response_text=response_text[:500],
            )
            raise

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model: Model name to pull (e.g., "qwen3:8b")

        Returns:
            True if successful, False otherwise

        Note:
            This is a blocking call that may take several minutes for large models.
        """
        logger.info("ollama_pull_model_start", model=model)

        try:
            # Use a longer timeout for model pulls
            response = self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=httpx.Timeout(600.0),  # 10 minutes
            )
            response.raise_for_status()

            logger.info("ollama_pull_model_success", model=model)
            return True

        except Exception as e:
            logger.error("ollama_pull_model_failed", model=model, error=str(e))
            return False
