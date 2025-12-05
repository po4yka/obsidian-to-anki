"""Domain interfaces for tag generation.

This module defines interfaces for generating tags following
Clean Architecture principles.
"""

from abc import ABC, abstractmethod

from obsidian_anki_sync.models import NoteMetadata


class ITagGenerator(ABC):
    """Interface for generating deterministic tags from metadata."""

    @abstractmethod
    def generate_tags(self, metadata: NoteMetadata, lang: str) -> list[str]:
        """Generate deterministic tags from metadata.

        This ensures tag taxonomy compliance and consistency.

        Args:
            metadata: Note metadata
            lang: Language code

        Returns:
            List of 3-6 snake_case tags
        """


class ICodeDetector(ABC):
    """Interface for detecting programming languages from code."""

    @abstractmethod
    def detect_code_language(self, code: str) -> str:
        """Detect programming language from code content.

        Uses syntax patterns to automatically detect the language.

        Args:
            code: Code snippet to analyze

        Returns:
            Detected language name or 'text' if unknown
        """

    @abstractmethod
    def get_code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks from metadata."""
