"""Interface for card generation operations."""

from abc import ABC, abstractmethod

from obsidian_anki_sync.domain.entities.card import Card
from obsidian_anki_sync.domain.entities.note import Note, QAPair


class ICardGenerator(ABC):
    """Interface for card generation from notes.

    This interface defines the contract for generating Anki cards
    from Obsidian notes containing Q&A pairs.
    """

    @abstractmethod
    def generate_cards_from_note(self, note: Note) -> list[Card]:
        """Generate cards from a complete note.

        Args:
            note: Note entity to generate cards from

        Returns:
            List of generated cards
        """

    @abstractmethod
    def generate_card_from_qa_pair(
        self, qa_pair: QAPair, note: Note, language: str
    ) -> Card:
        """Generate a single card from a Q&A pair.

        Args:
            qa_pair: Q&A pair to generate card from
            note: Parent note
            language: Target language for the card

        Returns:
            Generated card
        """

    @abstractmethod
    def generate_apf_html(
        self, qa_pair: QAPair, language: str, note_title: str, card_index: int
    ) -> str:
        """Generate APF-formatted HTML for a card.

        Args:
            qa_pair: Q&A pair data
            language: Language code
            note_title: Title of the parent note
            card_index: Index of the card within the note

        Returns:
            APF-formatted HTML string
        """

    @abstractmethod
    def validate_card_content(self, card: Card) -> list[str]:
        """Validate card content and return any issues.

        Args:
            card: Card to validate

        Returns:
            List of validation error messages
        """

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for card generation.

        Returns:
            List of language codes (e.g., ['en', 'ru'])
        """

    @abstractmethod
    def get_note_type_for_card(self, card: Card) -> str:
        """Determine the appropriate Anki note type for a card.

        Args:
            card: Card to determine note type for

        Returns:
            Anki note type name
        """

    @abstractmethod
    def create_manifest(
        self, note: Note, card_index: int, language: str, slug: str, slug_base: str
    ) -> dict:
        """Create manifest data for a card.

        Args:
            note: Parent note
            card_index: Card index within note
            language: Language code
            slug: Full card slug
            slug_base: Base slug without language

        Returns:
            Manifest dictionary
        """
