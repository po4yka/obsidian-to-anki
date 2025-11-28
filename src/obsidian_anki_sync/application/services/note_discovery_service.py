"""Application service for discovering and scanning Obsidian notes."""

import random
from pathlib import Path

from ...domain.entities.note import Note, NoteMetadata
from ...domain.interfaces.note_parser import INoteParser
from ...domain.interfaces.state_repository import IStateRepository
from ...utils.logging import get_logger

logger = get_logger(__name__)


class NoteDiscoveryService:
    """Service for discovering and scanning Obsidian notes.

    This service focuses solely on discovering notes in the vault,
    following the Single Responsibility Principle. It does not
    handle card generation or processing.
    """

    def __init__(
        self,
        vault_path: Path,
        source_dir: str,
        state_repository: IStateRepository,
        note_parser: INoteParser,
    ):
        """Initialize note discovery service.

        Args:
            vault_path: Path to the Obsidian vault
            source_dir: Relative path to source directory within vault
            state_repository: Repository for tracking processed notes
            note_parser: Parser for reading note files
        """
        self.vault_path = vault_path
        self.source_dir = source_dir
        self.state_repository = state_repository
        self.note_parser = note_parser

    def discover_notes(
        self,
        sample_size: int | None = None,
        incremental: bool = False,
        exclude_patterns: list[str | None] = None,
    ) -> list[Note]:
        """Discover notes in the vault.

        Args:
            sample_size: Optional number of notes to randomly sample
            incremental: If True, only return notes not yet processed
            exclude_patterns: Optional patterns to exclude

        Returns:
            List of discovered Note entities
        """
        logger.info(
            "discovering_notes",
            vault_path=str(self.vault_path),
            source_dir=self.source_dir,
            sample_size=sample_size,
            incremental=incremental,
        )

        # Find all note files in the vault
        source_path = self.vault_path / self.source_dir
        if not source_path.exists():
            logger.warning("source_directory_not_found", path=str(source_path))
            return []

        note_files = self._find_note_files(source_path, exclude_patterns)
        logger.debug("found_note_files", count=len(note_files))

        # Filter for incremental processing
        if incremental:
            processed_paths = self._get_processed_paths()
            note_files = [f for f in note_files if str(f) not in processed_paths]
            logger.debug("filtered_incremental", remaining=len(note_files))

        # Apply sampling
        if sample_size and len(note_files) > sample_size:
            note_files = random.sample(note_files, sample_size)
            logger.debug("applied_sampling", sample_size=sample_size)

        # Parse notes into domain entities
        notes = []
        for file_path in note_files:
            try:
                note = self._parse_note_file(file_path)
                if note:
                    notes.append(note)
            except Exception as e:
                logger.warning(
                    "failed_to_parse_note",
                    file_path=str(file_path),
                    error=str(e),
                )
                continue

        logger.info(
            "notes_discovered",
            total_found=len(notes),
            from_files=len(note_files),
        )

        return notes

    def _find_note_files(
        self, source_path: Path, exclude_patterns: list[str | None] = None
    ) -> list[Path]:
        """Find all markdown note files in the source directory.

        Args:
            source_path: Directory to search
            exclude_patterns: Patterns to exclude

        Returns:
            List of markdown file paths
        """
        note_files = []

        # Find all .md files
        for md_file in source_path.rglob("*.md"):
            if md_file.is_file():
                # Apply exclude patterns
                if exclude_patterns:
                    relative_path = str(md_file.relative_to(source_path))
                    if any(pattern in relative_path for pattern in exclude_patterns):
                        continue

                note_files.append(md_file)

        # Sort for consistent ordering
        note_files.sort()

        return note_files

    def _get_processed_paths(self) -> set[str]:
        """Get set of already processed note paths.

        Returns:
            Set of absolute paths to processed notes
        """
        try:
            return set()
        except Exception as e:
            logger.warning("failed_to_get_processed_paths", error=str(e))
            return set()

    def _parse_note_file(self, file_path: Path) -> Note | None:
        """Parse a note file into a domain Note entity.

        Args:
            file_path: Path to the note file

        Returns:
            Note entity if parsing successful, None otherwise
        """
        try:
            # Parse the note using the parser interface
            metadata, qa_pairs = self.note_parser.parse_note(file_path)

            # Create domain entity
            note = Note(
                id=metadata.id,
                title=metadata.title,
                content=self._read_file_content(file_path),
                file_path=file_path,
                metadata=NoteMetadata(
                    topic=metadata.topic,
                    language_tags=metadata.language_tags,
                    subtopics=metadata.subtopics or [],
                    difficulty=metadata.difficulty,
                    question_kind=metadata.question_kind,
                    tags=metadata.tags or [],
                    aliases=metadata.aliases or [],
                    status=metadata.status,
                    original_language=metadata.original_language,
                    source=metadata.source,
                    moc=metadata.moc,
                    related=metadata.related or [],
                    anki_note_type=metadata.anki_note_type,
                    anki_slugs=metadata.anki_slugs or [],
                ),
                created_at=metadata.created,
                updated_at=metadata.updated,
            )

            return note

        except Exception as e:
            logger.warning(
                "note_parsing_failed",
                file_path=str(file_path),
                error=str(e),
            )
            return None

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content safely.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(
                "failed_to_read_file",
                file_path=str(file_path),
                error=str(e),
            )
            return ""

    def get_note_statistics(self, notes: list[Note]) -> dict:
        """Calculate statistics about discovered notes.

        Args:
            notes: List of discovered notes

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_notes": len(notes),
            "by_topic": {},
            "by_language": {},
            "by_status": {},
            "by_difficulty": {},
        }

        for note in notes:
            # Count by topic
            topic = note.metadata.topic
            stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1

            # Count by primary language
            primary_lang = note.metadata.primary_language
            stats["by_language"][primary_lang] = (
                stats["by_language"].get(primary_lang, 0) + 1
            )

            # Count by status
            status = note.metadata.status or "unknown"
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by difficulty
            difficulty = note.metadata.difficulty or "unknown"
            stats["by_difficulty"][difficulty] = (
                stats["by_difficulty"].get(difficulty, 0) + 1
            )

        return stats
