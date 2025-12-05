"""Interface for Anki card operations."""

from abc import ABC, abstractmethod
from typing import Any


class IAnkiCardService(ABC):
    """Interface for Anki card operations.

    Defines operations for working with Anki cards,
    including suspending, unsuspending, and retrieving card information.
    """

    @abstractmethod
    def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed information about cards.

        Args:
            card_ids: List of card IDs

        Returns:
            List of card information dictionaries
        """

    @abstractmethod
    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards by ID.

        Args:
            card_ids: List of card IDs to suspend
        """

    @abstractmethod
    async def suspend_cards_async(self, card_ids: list[int]) -> None:
        """Suspend cards by ID (async).

        Args:
            card_ids: List of card IDs to suspend
        """

    @abstractmethod
    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID.

        Args:
            card_ids: List of card IDs to unsuspend
        """

    @abstractmethod
    async def unsuspend_cards_async(self, card_ids: list[int]) -> None:
        """Unsuspend cards by ID (async).

        Args:
            card_ids: List of card IDs to unsuspend
        """

    @abstractmethod
    def get_note_id_from_card_id(self, card_id: int) -> int:
        """Get note ID from card ID.

        Args:
            card_id: Card ID

        Returns:
            Note ID
        """

    @abstractmethod
    def get_card_ids_from_note_id(self, note_id: int) -> list[int]:
        """Get card IDs from note ID.

        Args:
            note_id: Note ID

        Returns:
            List of card IDs
        """
