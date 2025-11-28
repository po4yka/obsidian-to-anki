"""Interface for Anki configuration."""

from abc import ABC, abstractmethod


class IAnkiConfig(ABC):
    """Interface for Anki configuration.

    This interface defines the contract for accessing Anki-related
    configuration following the Interface Segregation Principle.
    """

    @property
    @abstractmethod
    def anki_connect_url(self) -> str:
        """Get the AnkiConnect URL.

        Returns:
            URL for AnkiConnect API
        """

    @property
    @abstractmethod
    def anki_deck_name(self) -> str:
        """Get the default Anki deck name.

        Returns:
            Name of the Anki deck to use
        """

    @property
    @abstractmethod
    def anki_note_type(self) -> str:
        """Get the default Anki note type.

        Returns:
            Note type to use for new cards
        """

    @property
    @abstractmethod
    def model_names(self) -> dict[str, str]:
        """Get mapping of internal note types to Anki model names.

        Returns:
            Dictionary mapping internal types to Anki model names
        """

    @property
    @abstractmethod
    def run_mode(self) -> str:
        """Get the run mode for sync operations.

        Returns:
            'apply' or 'dry-run'
        """

    @property
    @abstractmethod
    def delete_mode(self) -> str:
        """Get the delete mode for sync operations.

        Returns:
            'delete' or 'archive'
        """

    @abstractmethod
    def get_model_name(self, internal_type: str) -> str:
        """Get the Anki model name for an internal note type.

        Args:
            internal_type: Internal note type (e.g., 'APF::Simple')

        Returns:
            Actual Anki model name
        """

    @abstractmethod
    def validate_anki_connection(self) -> bool:
        """Validate that AnkiConnect is accessible.

        Returns:
            True if connection is valid, False otherwise
        """
