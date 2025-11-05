"""Base LLM provider interface."""

import json
from abc import ABC, abstractmethod
from typing import Any, cast

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    This interface defines the contract that all LLM providers must implement,
    allowing for seamless switching between different providers (Ollama, LM Studio,
    OpenRouter, etc.) while maintaining consistent behavior.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the provider with configuration parameters.

        Args:
            **kwargs: Provider-specific configuration options
        """
        self.config = kwargs
        logger.info(
            "provider_initialized",
            provider=self.__class__.__name__,
            config=self._safe_config_for_logging(),
        )

    def _safe_config_for_logging(self) -> dict[str, Any]:
        """Return config with sensitive data redacted for logging.

        Returns:
            Config dictionary with API keys and tokens redacted
        """
        safe_config = self.config.copy()
        for key in ["api_key", "token", "password"]:
            if key in safe_config:
                safe_config[key] = "***REDACTED***"
        return safe_config

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        stream: bool = False,
    ) -> dict[str, Any]:
        """Generate a completion from the LLM.

        Args:
            model: Model identifier (e.g., "qwen3:8b", "gpt-4")
            prompt: User prompt/question
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            stream: Enable streaming (if supported by provider)

        Returns:
            Response dictionary with at least a 'response' key containing the text.
            Additional keys may include 'context', 'tokens', 'finish_reason', etc.

        Raises:
            NotImplementedError: If the provider doesn't support this operation
            Exception: Provider-specific errors (network, API, etc.)
        """
        pass

    def generate_json(
        self, model: str, prompt: str, system: str = "", temperature: float = 0.7
    ) -> dict[str, Any]:
        """Generate a JSON response from the LLM.

        This is a convenience method that ensures JSON format and parses the result.
        Default implementation calls generate() with format="json" and parses response.
        Providers can override this for custom JSON handling.

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
                "json_parse_error",
                provider=self.__class__.__name__,
                error=str(e),
                response_text=response_text[:500],
            )
            raise

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if the provider is accessible and healthy.

        Returns:
            True if the provider is accessible, False otherwise
        """
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models from the provider.

        Returns:
            List of model identifiers/names

        Note:
            Some providers may return an empty list if listing is not supported
            or if authentication fails.
        """
        pass

    def get_provider_name(self) -> str:
        """Get the human-readable name of this provider.

        Returns:
            Provider name (e.g., "Ollama", "LM Studio", "OpenRouter")
        """
        return self.__class__.__name__.replace("Provider", "")

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(config={self._safe_config_for_logging()})"
