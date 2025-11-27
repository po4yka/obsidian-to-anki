"""Interface for note parsing operations."""

from abc import ABC, abstractmethod
from pathlib import Path


from ..entities.note import NoteMetadata, QAPair


class INoteParser(ABC):
    """Interface for parsing Obsidian notes.

    This interface defines the contract for parsing Obsidian markdown
    files into domain entities (Note, QAPair, etc.).
    """

    @abstractmethod
    def parse_note(self, file_path: Path) -> tuple[NoteMetadata, list[QAPair]]:
        """Parse a note file into metadata and Q&A pairs.

        Args:
            file_path: Path to the note file

        Returns:
            Tuple of (NoteMetadata, list[QAPair])
        """
        pass

    @abstractmethod
    def parse_frontmatter(self, content: str, file_path: Path | None = None) -> NoteMetadata:
        """Parse YAML frontmatter from note content.

        Args:
            content: Raw note content
            file_path: Optional file path for additional context

        Returns:
            Parsed NoteMetadata
        """
        pass

    @abstractmethod
    def extract_qa_pairs(self, content: str) -> list[QAPair]:
        """Extract Q&A pairs from note content.

        Args:
            content: Note content without frontmatter

        Returns:
            List of extracted Q&A pairs
        """
        pass

    @abstractmethod
    def validate_note_structure(
        self,
        metadata: NoteMetadata,
        content: str,
        enforce_bilingual: bool = True,
        check_code_fences: bool = True
    ) -> list[str]:
        """Validate note structure and return issues.

        Args:
            metadata: Note metadata
            content: Note content
            enforce_bilingual: Whether to enforce bilingual content
            check_code_fences: Whether to validate code fences

        Returns:
            List of validation error messages
        """
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if parser supports a specific language.

        Args:
            language: Language code to check

        Returns:
            True if language is supported
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            List of supported language codes
        """
        pass

    @abstractmethod
    def extract_title_from_content(self, content: str) -> str | None:
        """Extract note title from content (fallback method).

        Args:
            content: Note content

        Returns:
            Extracted title if found, None otherwise
        """
        pass

    @abstractmethod
    def get_note_creation_date(self, file_path: Path) -> float | None:
        """Get note creation date from file metadata.

        Args:
            file_path: Path to note file

        Returns:
            Unix timestamp if available, None otherwise
        """
        pass

    @abstractmethod
    def get_note_modification_date(self, file_path: Path) -> float | None:
        """Get note modification date from file metadata.

        Args:
            file_path: Path to note file

        Returns:
            Unix timestamp if available, None otherwise
        """
        pass
