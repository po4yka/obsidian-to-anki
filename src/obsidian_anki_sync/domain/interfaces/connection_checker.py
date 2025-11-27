"""Interface for connection checking operations."""

from abc import ABC, abstractmethod


class IConnectionChecker(ABC):
    """Interface for checking LLM provider connectivity.

    This interface focuses solely on connection health checking,
    following the Interface Segregation Principle.
    """

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if the provider is accessible and healthy.

        Returns:
            True if connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def get_connection_status(self) -> dict:
        """Get detailed connection status information.

        Returns:
            Dictionary with connection details including:
            - connected: bool
            - latency: float (optional)
            - last_check: datetime (optional)
            - error: str (optional)
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate that authentication credentials are correct.

        Returns:
            True if credentials are valid, False otherwise
        """
        pass
