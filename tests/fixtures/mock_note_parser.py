"""Mock implementation of INoteParser for testing."""

from pathlib import Path

from obsidian_anki_sync.domain.entities.note import NoteMetadata, QAPair
from obsidian_anki_sync.domain.interfaces.note_parser import INoteParser


class MockNoteParser(INoteParser):
    """Mock implementation of note parser for testing.

    Provides controllable parsing results for testing sync operations
    without requiring real file parsing.
    """

    def __init__(self):
        """Initialize mock parser."""
        self.parsed_notes = {}  # file_path -> (metadata, qa_pairs)
        self.should_fail = False
        self.fail_message = "Mock note parser failure"

    def parse_note(self, file_path: Path) -> tuple[NoteMetadata, list[QAPair]]:
        """Parse note file."""
        if self.should_fail:
            raise Exception(self.fail_message)

        # Check if we have mock data for this file
        file_key = str(file_path)
        if file_key in self.parsed_notes:
            return self.parsed_notes[file_key]

        # Generate default mock data
        metadata = NoteMetadata(
            topic="Mock Topic",
            language_tags=["en", "ru"],
            difficulty="medium",
            question_kind="concept",
            tags=["mock", "test"],
        )

        qa_pairs = [
            QAPair(
                card_index=1,
                question_en="What is a mock test?",
                question_ru="Что такое мок тест?",
                answer_en="A mock test simulates behavior for testing.",
                answer_ru="Мок тест симулирует поведение для тестирования.",
            ),
            QAPair(
                card_index=2,
                question_en="Why use mocks?",
                question_ru="Зачем использовать моки?",
                answer_en="Mocks isolate code under test from dependencies.",
                answer_ru="Моки изолируют тестируемый код от зависимостей.",
            ),
        ]

        return metadata, qa_pairs

    def parse_frontmatter(
        self, content: str, file_path: Path | None = None
    ) -> NoteMetadata:
        """Parse YAML frontmatter."""
        if self.should_fail:
            raise Exception(self.fail_message)

        # Simple mock parsing - extract from content if it looks like YAML
        if content.strip().startswith("---"):
            # Mock YAML parsing
            return NoteMetadata(
                topic="Parsed Topic",
                language_tags=["en"],
                tags=["parsed"],
            )

        # Default metadata
        return NoteMetadata(
            topic="Default Topic",
            language_tags=["en"],
        )

    def extract_qa_pairs(self, content: str) -> list[QAPair]:
        """Extract Q&A pairs from content."""
        if self.should_fail:
            raise Exception(self.fail_message)

        # Simple mock extraction - look for question patterns
        pairs = []
        lines = content.split("\n")
        current_pair = None

        for line in lines:
            line = line.strip()
            if line.startswith("# Question"):
                if current_pair:
                    pairs.append(current_pair)
                current_pair = QAPair(
                    card_index=len(pairs) + 1,
                    question_en="Mock question",
                    question_ru="Мок вопрос",
                    answer_en="Mock answer",
                    answer_ru="Мок ответ",
                )
            elif line.startswith("# Answer") and current_pair:
                # Could parse answer here
                pass

        if current_pair:
            pairs.append(current_pair)

        return pairs

    def validate_note_structure(
        self,
        metadata: NoteMetadata,
        content: str,
        enforce_bilingual: bool = True,
        check_code_fences: bool = True,
    ) -> list[str]:
        """Validate note structure."""
        errors = []

        if not metadata.topic:
            errors.append("Missing topic")

        if not metadata.language_tags:
            errors.append("Missing language tags")

        if enforce_bilingual and len(metadata.language_tags) < 2:
            errors.append("Bilingual content required")

        if check_code_fences:
            # Check for balanced code fences
            backticks = content.count("```")
            if backticks % 2 != 0:
                errors.append("Unbalanced code fences")

        return errors

    def supports_language(self, language: str) -> bool:
        """Check if language is supported."""
        return language in ["en", "ru", "es", "fr", "de"]

    def get_supported_languages(self) -> list[str]:
        """Get supported languages."""
        return ["en", "ru", "es", "fr", "de"]

    def extract_title_from_content(self, content: str) -> str | None:
        """Extract title from content."""
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# ") and not line.startswith("##"):
                return line[2:].strip()
        return None

    def get_note_creation_date(self, file_path: Path) -> float | None:
        """Get creation date."""
        try:
            stat = file_path.stat()
            return stat.st_ctime
        except OSError:
            return None

    def get_note_modification_date(self, file_path: Path) -> float | None:
        """Get modification date."""
        try:
            stat = file_path.stat()
            return stat.st_mtime
        except OSError:
            return None

    # Test helper methods

    def set_mock_note_data(
        self, file_path: str, metadata: NoteMetadata, qa_pairs: list[QAPair]
    ) -> None:
        """Set mock data for a file path."""
        self.parsed_notes[file_path] = (metadata, qa_pairs)

    def set_failure(self, message: str = "Mock note parser failure") -> None:
        """Make parser fail on next calls."""
        self.should_fail = True
        self.fail_message = message

    def clear_failure(self) -> None:
        """Clear failure state."""
        self.should_fail = False

    def get_parsed_files_count(self) -> int:
        """Get number of files with mock data."""
        return len(self.parsed_notes)

    def reset(self) -> None:
        """Reset mock state."""
        self.parsed_notes.clear()
        self.should_fail = False
        self.fail_message = "Mock note parser failure"
