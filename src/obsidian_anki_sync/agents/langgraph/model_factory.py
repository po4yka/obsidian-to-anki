from typing import Any

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers.pydantic_ai_models import PydanticAIModelFactory
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating and managing PydanticAI models."""

    def __init__(self, config: Config):
        """Initialize the model factory.

        Args:
            config: Service configuration
        """
        self.config = config
        self._model_cache: dict[str, Any] = {}

    def get_model(self, agent_type: str) -> Any:
        """Get a configured model for the specified agent type.

        Uses cached model if available, otherwise creates a new one.

        Args:
            agent_type: Agent type (e.g., "pre_validator", "generator")

        Returns:
            Configured PydanticAI model
        """
        if agent_type in self._model_cache:
            return self._model_cache[agent_type]

        try:
            model = self._create_model_with_config(agent_type)
            self._model_cache[agent_type] = model
            return model
        except Exception as e:
            logger.warning(
                "failed_to_create_model", agent_type=agent_type, error=str(e)
            )
            return None

    def _create_model_with_config(self, agent_type: str) -> Any:
        """Create a PydanticAI model with full configuration including reasoning.

        Args:
            agent_type: Agent type (e.g., "pre_validator", "generator")

        Returns:
            Configured PydanticAI model
        """
        model_name = self.config.get_model_for_agent(agent_type)
        model_config = self.config.get_model_config_for_task(agent_type)

        return PydanticAIModelFactory.create_from_config(
            self.config,
            model_name=model_name,
            reasoning_enabled=model_config.get("reasoning_enabled", False),
            max_tokens=model_config.get("max_tokens"),
            agent_type=agent_type,
            reasoning_effort=model_config.get("reasoning_effort"),
        )

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
