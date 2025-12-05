"""Interface for HTTP communication with AnkiConnect."""

from abc import ABC, abstractmethod
from typing import Any


class IAnkiHttpClient(ABC):
    """Interface for HTTP communication with AnkiConnect API.

    This interface defines the low-level HTTP operations needed
    to communicate with AnkiConnect, including connection management,
    health checks, and request/response handling.
    """

    @abstractmethod
    def invoke(self, action: str, params: dict[str, Any] | None = None) -> Any:
        """Invoke an AnkiConnect action synchronously.

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Action result

        Raises:
            AnkiConnectError: If the action fails
        """

    @abstractmethod
    async def invoke_async(self, action: str, params: dict[str, Any] | None = None) -> Any:
        """Invoke an AnkiConnect action asynchronously.

        Args:
            action: Action name
            params: Action parameters

        Returns:
            Action result

        Raises:
            AnkiConnectError: If the action fails
        """

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if AnkiConnect is accessible and healthy.

        Returns:
            True if connection is successful, False otherwise
        """

    @abstractmethod
    def close(self) -> None:
        """Close HTTP sessions and cleanup resources."""

    @abstractmethod
    async def aclose(self) -> None:
        """Async cleanup for async contexts."""
