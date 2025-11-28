"""Interface for LLM configuration."""

from abc import ABC, abstractmethod
from typing import Any


class ILLMConfig(ABC):
    """Interface for LLM provider configuration.

    This interface defines the contract for accessing LLM-related
    configuration following the Interface Segregation Principle.
    """

    @property
    @abstractmethod
    def llm_provider(self) -> str:
        """Get the LLM provider name.

        Returns:
            Provider name (e.g., 'ollama', 'openrouter')
        """

    @property
    @abstractmethod
    def llm_timeout(self) -> float:
        """Get the LLM request timeout.

        Returns:
            Timeout in seconds
        """

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Get the default temperature for LLM requests.

        Returns:
            Temperature value (0.0-2.0)
        """

    @property
    @abstractmethod
    def top_p(self) -> float:
        """Get the top-p sampling parameter.

        Returns:
            Top-p value (0.0-1.0)
        """

    @property
    @abstractmethod
    def max_tokens(self) -> int | None:
        """Get the maximum tokens for LLM responses.

        Returns:
            Maximum token count or None for unlimited
        """

    @property
    @abstractmethod
    def reasoning_enabled(self) -> bool:
        """Check if reasoning mode is enabled.

        Returns:
            True if reasoning mode should be used
        """

    @abstractmethod
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model name for a specific agent type.

        Args:
            agent_type: Type of agent (e.g., 'pre_validator', 'generator')

        Returns:
            Model name to use for this agent
        """

    @abstractmethod
    def get_model_config_for_task(self, task: str) -> dict[str, Any]:
        """Get model configuration for a specific task.

        Args:
            task: Task name (e.g., 'pre_validation', 'generation')

        Returns:
            Dictionary with model configuration
        """

    @abstractmethod
    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration specific to a provider.

        Args:
            provider: Provider name

        Returns:
            Provider-specific configuration dictionary
        """
