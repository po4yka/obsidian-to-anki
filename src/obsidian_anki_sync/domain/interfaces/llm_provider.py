"""Interface for LLM provider operations."""

from abc import ABC, abstractmethod
from typing import Any

from .connection_checker import IConnectionChecker
from .llm_generator import IGenerator
from .model_provider import IModelProvider


class ILLMProvider(IGenerator, IConnectionChecker, IModelProvider, ABC):
    """Interface for LLM provider operations.

    This interface combines generation, connection checking, and model
    provider capabilities following the Interface Segregation Principle
    through composition of focused interfaces.
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get human-readable provider name.

        Returns:
            Provider name (e.g., "OpenAI", "Anthropic")
        """

    @abstractmethod
    def get_provider_info(self) -> dict[str, Any]:
        """Get provider information and capabilities.

        Returns:
            Dictionary with provider metadata
        """
