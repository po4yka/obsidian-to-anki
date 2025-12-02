"""Interface for LLM generation operations."""

from abc import ABC, abstractmethod
from typing import Any


class IGenerator(ABC):
    """Interface for LLM text generation operations.

    This interface focuses solely on text generation capabilities,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: dict[str, Any | None] | None = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Generate a completion from the LLM.

        Args:
            model: Model identifier
            prompt: User prompt/question
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            format: Response format ("json" for structured output)
            json_schema: JSON schema for structured output
            stream: Enable streaming (if supported)
            reasoning_enabled: Enable reasoning mode
            reasoning_effort: Desired reasoning effort level

        Returns:
            Response with 'response' key containing generated text
        """

    @abstractmethod
    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        json_schema: dict[str, Any | None] | None = None,
        reasoning_enabled: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            model: Model identifier
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            json_schema: JSON schema for structured output
            reasoning_enabled: Enable reasoning mode
            reasoning_effort: Desired reasoning effort level

        Returns:
            Parsed JSON response as dictionary
        """

    @abstractmethod
    def generate_async(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        format: str = "",
        json_schema: dict[str, Any | None] | None = None,
        stream: bool = False,
        reasoning_enabled: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Generate a completion asynchronously.

        Args:
            model: Model identifier
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            format: Response format
            json_schema: JSON schema for structured output
            stream: Enable streaming
            reasoning_enabled: Enable reasoning mode
            reasoning_effort: Desired reasoning effort level

        Returns:
            Response with 'response' key containing generated text
        """
