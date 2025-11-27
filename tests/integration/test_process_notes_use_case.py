"""Integration tests for ProcessNotesUseCase."""

import pytest
from pathlib import Path

from obsidian_anki_sync.application.use_cases.process_notes import (
    ProcessNotesRequest,
    ProcessNotesUseCase,
)
from obsidian_anki_sync.domain.entities.note import NoteMetadata
from tests.fixtures import MockCardGenerator, MockNoteParser, MockStateRepository


class TestProcessNotesUseCase:
    """Test the ProcessNotesUseCase integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.note_parser = MockNoteParser()
        self.card_generator = MockCardGenerator()
        self.state_repository = MockStateRepository()

        # Create the use case with mocked dependencies
        self.use_case = ProcessNotesUseCase(
            note_discovery_service=None,  # We'll mock this differently
            card_generator=self.card_generator,
        )

        # Mock the note discovery service
        self.mock_notes = self._create_mock_notes()

    def _create_mock_notes(self):
        """Create mock notes for testing."""
        notes = []

        # Create a sample note
        metadata = NoteMetadata(
            topic="Integration Testing",
            language_tags=["en", "ru"],
            difficulty="medium",
            tags=["test", "integration"],
        )

        note = Note(
            id="integration-test-1",
            title="Integration Test Note",
            content="# Test Note\n\nThis is a test.",
            file_path=Path("/tmp/test.md"),
            metadata=metadata,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        notes.append(note)
        return notes

    def test_process_notes_success(self):
        """Test successful note processing."""
        # Setup mock data
        self.card_generator.set_card_template("integration-test-1", [
            {
                "question": "What is integration testing?",
                "answer": "Integration testing verifies that different parts of the system work together.",
                "slug_suffix": "int-test-1",
            }
        ])

        # Create request
        request = ProcessNotesRequest(
            sample_size=None,
            incremental=False,
        )

        # Note: This test would need a real NoteDiscoveryService or mock
        # For now, this demonstrates the structure
        # result = self.use_case.execute(request)

        # Assertions would go here:
        # assert result.success
        # assert len(result.notes) > 0
        # assert len(result.cards) > 0
        # assert result.stats["cards_generated"] > 0

    def test_process_notes_with_errors(self):
        """Test note processing with errors."""
        # Setup generator to fail
        self.card_generator.set_failure("Simulated card generation failure")

        request = ProcessNotesRequest()

        # result = self.use_case.execute(request)
        # assert not result.success
        # assert len(result.errors) > 0

    def test_process_notes_language_filtering(self):
        """Test processing with language filtering."""
        request = ProcessNotesRequest(
            languages=["en"],  # Only English cards
        )

        # This would test that only English cards are generated
        # when language filtering is applied

    def test_card_generator_integration(self):
        """Test that the use case properly integrates with card generator."""
        # Verify that the use case calls the card generator correctly
        assert self.card_generator.get_generation_count() == 0

        # After processing, count should increase
        # assert self.card_generator.get_generation_count() > 0

    def test_statistics_tracking(self):
        """Test that processing statistics are tracked correctly."""
        request = ProcessNotesRequest()

        # result = self.use_case.execute(request)
        # assert "notes_discovered" in result.stats
        # assert "cards_generated" in result.stats
        # assert "processing_time_seconds" in result.stats
