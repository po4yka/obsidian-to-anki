"""Interface for model listing and information operations."""

from abc import ABC, abstractmethod


class IModelProvider(ABC):
    """Interface for LLM model information and listing.

    This interface focuses solely on model discovery and metadata,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models from the provider.

        Returns:
            List of model identifiers/names
        """
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> dict | None:
        """Get detailed information about a specific model.

        Args:
            model: Model identifier

        Returns:
            Model information dictionary or None if not found
        """
        pass

    @abstractmethod
    def list_models_with_info(self) -> list[dict]:
        """List all models with their detailed information.

        Returns:
            List of model information dictionaries
        """
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if a specific model is supported.

        Args:
            model: Model identifier to check

        Returns:
            True if model is supported, False otherwise
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider.

        Returns:
            Default model identifier
        """
        pass
