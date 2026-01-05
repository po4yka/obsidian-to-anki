"""Interface for Anki model operations."""

from abc import ABC, abstractmethod


class IAnkiModelService(ABC):
    """Interface for Anki model (note type) operations.

    Defines operations for working with Anki note types/models,
    including retrieving model information and field definitions.
    """

    @abstractmethod
    def get_model_names(self, use_cache: bool = True) -> list[str]:
        """Get list of available note model names.

        Args:
            use_cache: Whether to use cached value if available

        Returns:
            List of model names
        """

    @abstractmethod
    async def get_model_names_async(self, use_cache: bool = True) -> list[str]:
        """Get list of available note model names (async).

        Args:
            use_cache: Whether to use cached value if available

        Returns:
            List of model names
        """

    @abstractmethod
    def get_model_field_names(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a specific model.

        Args:
            model_name: Name of the model
            use_cache: Whether to use cached value if available

        Returns:
            List of field names
        """

    @abstractmethod
    async def get_model_field_names_async(
        self, model_name: str, use_cache: bool = True
    ) -> list[str]:
        """Get field names for a specific model (async).

        Args:
            model_name: Name of the model
            use_cache: Whether to use cached value if available

        Returns:
            List of field names
        """

    @abstractmethod
    def get_model_names_and_ids(self) -> dict[str, int]:
        """Get note type names and their IDs.

        Returns:
            Dictionary mapping model names to model IDs
        """
