"""Domain interfaces for card generation.

This module defines the core interfaces for card generation following
Clean Architecture principles.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

from obsidian_anki_sync.agents.models import GeneratedCard, GenerationResult
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair


class ParsedCardStructure(NamedTuple):
    """Parsed structure of an English APF card for translation."""

    title: str
    key_point_code: str | None
    key_point_notes: list[str]
    other_notes: list[str] | None


class ICardGenerator(ABC):
    """Interface for card generation services."""

    @abstractmethod
    def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> GenerationResult:
        """Generate APF cards for all Q/A pairs in a note."""

    @abstractmethod
    def create_manifest(
        self, qa_pair: QAPair, metadata: NoteMetadata, slug_base: str, lang: str
    ) -> Manifest:
        """Create card manifest."""


class ISingleCardGenerator(ABC):
    """Interface for generating individual cards."""

    @abstractmethod
    def generate_single_card(
        self, qa_pair: QAPair, metadata: NoteMetadata, manifest: Manifest, lang: str
    ) -> GeneratedCard:
        """Generate a single APF card."""


class ITranslatedCardGenerator(ABC):
    """Interface for generating translated cards."""

    @abstractmethod
    def generate_translated_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        english_structure: ParsedCardStructure,
        lang: str,
    ) -> GeneratedCard:
        """Generate a translated card using the canonical English structure."""


class ICardStructureParser(ABC):
    """Interface for parsing card structures."""

    @abstractmethod
    def parse_card_structure(self, apf_html: str) -> ParsedCardStructure:
        """Parse the structure of an English APF card for translation."""


class ICardDataExtractor(ABC):
    """Interface for extracting card data from HTML."""

    @abstractmethod
    def extract_card_data_from_html(
        self, apf_html: str, manifest: Manifest
    ) -> dict | None:
        """Extract card data from existing HTML for regeneration."""
