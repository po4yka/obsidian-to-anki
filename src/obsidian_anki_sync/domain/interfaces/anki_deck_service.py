"""Interface for Anki deck operations."""

from abc import ABC, abstractmethod
from typing import Any


class IAnkiDeckService(ABC):
    """Interface for Anki deck operations.

    Defines operations for working with Anki decks, including
    retrieving deck information and statistics.
    """

    @abstractmethod
    def get_deck_names(self, use_cache: bool = True) -> list[str]:
        """Get list of available deck names.

        Args:
            use_cache: Whether to use cached value if available

        Returns:
            List of deck names
        """

    @abstractmethod
    async def get_deck_names_async(self, use_cache: bool = True) -> list[str]:
        """Get list of available deck names (async).

        Args:
            use_cache: Whether to use cached value if available

        Returns:
            List of deck names
        """

    @abstractmethod
    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck.

        Args:
            deck_name: Name of the deck

        Returns:
            Deck statistics
        """
