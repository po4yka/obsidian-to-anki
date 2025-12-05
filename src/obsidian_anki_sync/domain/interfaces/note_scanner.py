"""Domain interfaces for note scanning components."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from obsidian_anki_sync.models import Card, QAPair


class INoteScanner(ABC):
    """Interface for note scanning operations."""

    @abstractmethod
    def scan_notes(
        self,
        sample_size: int | None = None,
        incremental: bool = False,
        qa_extractor: Any = None,
        existing_cards_for_duplicate_detection: list | None = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan Obsidian vault and generate cards.

        Args:
            sample_size: Optional number of notes to randomly process
            incremental: If True, only process new notes not yet in database
            qa_extractor: Optional QA extractor for LLM-based extraction
            existing_cards_for_duplicate_detection: Existing cards from Anki
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """


class INoteProcessor(ABC):
    """Interface for processing individual notes."""

    @abstractmethod
    def process_note(
        self,
        file_path: "Path",
        relative_path: str,
        existing_slugs: Collection[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file and generate cards.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            slug_lock: Optional lock for thread-safe slug operations
            existing_cards_for_duplicate_detection: Existing cards from Anki

        Returns:
            Tuple of (cards_dict, new_slugs_set, result_info)
        """


class IParallelProcessor(ABC):
    """Interface for parallel note processing."""

    @abstractmethod
    def scan_notes_parallel(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: dict[str, int],
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan notes using parallel processing.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs (thread-safe updates)
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """


class IQueueProcessor(ABC):
    """Interface for queue-based note processing."""

    @abstractmethod
    def scan_notes_with_queue(
        self,
        note_files: list[tuple[Any, str]],
        obsidian_cards: dict[str, Card],
        existing_slugs: set[str],
        error_by_type: dict[str, int],
        error_samples: dict[str, list[str]],
        qa_extractor: Any = None,
        on_batch_complete: Callable[[list[Card]], None] | None = None,
    ) -> dict[str, Card]:
        """Scan notes using Redis queue for distribution.

        Args:
            note_files: List of (file_path, relative_path) tuples
            obsidian_cards: Dict to populate with cards
            existing_slugs: Set of existing slugs
            error_by_type: Dict to aggregate errors by type
            error_samples: Dict to store sample errors
            qa_extractor: Optional QA extractor
            on_batch_complete: Optional callback for atomic batch processing

        Returns:
            Dict of slug -> Card
        """


class IArchiver(ABC):
    """Interface for archiving problematic notes."""

    @abstractmethod
    def archive_note_safely(
        self,
        file_path: "Path",
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Safely archive a problematic note.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred
            note_content: Optional note content
            card_index: Optional card index
            language: Optional language
        """

    @abstractmethod
    def process_deferred_archives(self) -> None:
        """Process all deferred archival requests sequentially."""

    @abstractmethod
    def set_defer_archival(self, defer: bool) -> None:
        """Set whether to defer archival operations.

        Args:
            defer: If True, defer archival to prevent FD exhaustion
        """
