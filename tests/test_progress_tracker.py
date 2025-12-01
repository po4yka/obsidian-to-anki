"""Tests for progress tracking functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from obsidian_anki_sync.sync.progress import (
    NoteProgress,
    ProgressTracker,
    SyncPhase,
    SyncProgress,
)
from obsidian_anki_sync.sync.state_db import StateDB


class TestSyncProgress:
    """Test SyncProgress dataclass."""

    def test_is_complete(self) -> None:
        """Test is_complete property."""
        progress = SyncProgress(
            session_id="test",
            phase=SyncPhase.COMPLETED,
            started_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert progress.is_complete

        progress.phase = SyncPhase.FAILED
        assert progress.is_complete

        progress.phase = SyncPhase.SCANNING
        assert not progress.is_complete

    def test_progress_pct(self) -> None:
        """Test progress percentage calculation."""
        progress = SyncProgress(
            session_id="test",
            phase=SyncPhase.SCANNING,
            started_at=datetime.now(),
            updated_at=datetime.now(),
            total_notes=100,
            notes_processed=25,
        )
        assert progress.progress_pct == 25.0

        progress.notes_processed = 50
        assert progress.progress_pct == 50.0

        progress.total_notes = 0
        assert progress.progress_pct == 0.0


class TestProgressTracker:
    """Test ProgressTracker class."""

    @pytest.fixture()
    def temp_db(self) -> None:
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_create_new_session(self, temp_db) -> None:
        """Test creating new progress tracker session."""
        tracker = ProgressTracker(temp_db)

        assert tracker.progress.session_id is not None
        assert tracker.progress.phase == SyncPhase.INITIALIZING
        assert tracker.progress.total_notes == 0
        assert tracker.progress.notes_processed == 0

    def test_resume_session(self, temp_db) -> None:
        """Test resuming existing session."""
        # Create initial session
        tracker1 = ProgressTracker(temp_db)
        session_id = tracker1.progress.session_id
        tracker1.set_total_notes(10)
        tracker1.complete_note("test.md", 1, "en", 1)

        # Resume session
        tracker2 = ProgressTracker(temp_db, session_id=session_id)
        assert tracker2.progress.session_id == session_id
        assert tracker2.progress.total_notes == 10
        assert tracker2.progress.notes_processed == 1

    def test_resume_nonexistent_session(self, temp_db) -> None:
        """Test resuming non-existent session raises error."""
        with pytest.raises(ValueError, match="No progress found"):
            ProgressTracker(temp_db, session_id="nonexistent")

    def test_set_phase(self, temp_db) -> None:
        """Test setting sync phase."""
        tracker = ProgressTracker(temp_db)

        tracker.set_phase(SyncPhase.SCANNING)
        assert tracker.progress.phase == SyncPhase.SCANNING

        tracker.set_phase(SyncPhase.APPLYING_CHANGES)
        assert tracker.progress.phase == SyncPhase.APPLYING_CHANGES

    def test_set_total_notes(self, temp_db) -> None:
        """Test setting total notes."""
        tracker = ProgressTracker(temp_db)

        tracker.set_total_notes(100)
        assert tracker.progress.total_notes == 100

    def test_note_tracking(self, temp_db) -> None:
        """Test tracking note processing."""
        tracker = ProgressTracker(temp_db)
        tracker.set_total_notes(5)

        # Start note
        tracker.start_note("note1.md", 1, "en")
        key = "note1.md:1:en"
        assert key in tracker.progress.note_progress
        assert tracker.progress.note_progress[key].status == "processing"

        # Complete note
        tracker.complete_note("note1.md", 1, "en", 1)
        assert tracker.progress.note_progress[key].status == "completed"
        assert tracker.progress.notes_processed == 1
        assert tracker.progress.cards_generated == 1

    def test_note_failure(self, temp_db) -> None:
        """Test tracking note failure."""
        tracker = ProgressTracker(temp_db)

        tracker.start_note("note1.md", 1, "en")
        tracker.fail_note("note1.md", 1, "en", "Parse error")

        key = "note1.md:1:en"
        assert tracker.progress.note_progress[key].status == "failed"
        assert tracker.progress.note_progress[key].error == "Parse error"
        assert tracker.progress.errors == 1

    def test_is_note_completed(self, temp_db) -> None:
        """Test checking if note is completed."""
        tracker = ProgressTracker(temp_db)

        assert not tracker.is_note_completed("note1.md", 1, "en")

        tracker.start_note("note1.md", 1, "en")
        assert not tracker.is_note_completed("note1.md", 1, "en")

        tracker.complete_note("note1.md", 1, "en", 1)
        assert tracker.is_note_completed("note1.md", 1, "en")

    def test_increment_stat(self, temp_db) -> None:
        """Test incrementing statistics."""
        tracker = ProgressTracker(temp_db)

        tracker.increment_stat("created", 5)
        assert tracker.progress.cards_created == 5

        tracker.increment_stat("updated", 3)
        assert tracker.progress.cards_updated == 3

        tracker.increment_stat("deleted")
        assert tracker.progress.cards_deleted == 1

    def test_complete_sync(self, temp_db) -> None:
        """Test completing sync."""
        tracker = ProgressTracker(temp_db)

        tracker.complete(success=True)
        assert tracker.progress.phase == SyncPhase.COMPLETED
        assert tracker.progress.completed_at is not None

        tracker2 = ProgressTracker(temp_db)
        tracker2.complete(success=False)
        assert tracker2.progress.phase == SyncPhase.FAILED

    def test_get_stats(self, temp_db) -> None:
        """Test getting statistics."""
        tracker = ProgressTracker(temp_db)
        tracker.set_total_notes(10)
        tracker.complete_note("note1.md", 1, "en", 2)
        tracker.increment_stat("created", 2)
        tracker.increment_stat("errors", 1)

        stats = tracker.get_stats()
        assert stats["processed"] == 1
        assert stats["created"] == 2
        assert stats["errors"] == 1

    def test_get_pending_notes(self, temp_db) -> None:
        """Test getting pending notes."""
        tracker = ProgressTracker(temp_db)

        tracker.start_note("note1.md", 1, "en")
        tracker.start_note("note2.md", 1, "en")
        tracker.complete_note("note1.md", 1, "en", 1)

        pending = tracker.get_pending_notes()
        assert len(pending) == 1
        assert pending[0].source_path == "note2.md"

    def test_interruption_flag(self, temp_db) -> None:
        """Test interruption flag."""
        tracker = ProgressTracker(temp_db)

        assert not tracker.is_interrupted()

        tracker._interrupt_event.set()
        assert tracker.is_interrupted()

    def test_signal_handler_installation(self, temp_db) -> None:
        """Test signal handler installation."""
        tracker = ProgressTracker(temp_db)

        # Should not raise
        tracker.install_signal_handlers()

        # Installing again should be idempotent
        tracker.install_signal_handlers()


class TestNoteProgress:
    """Test NoteProgress dataclass."""

    def test_creation(self) -> None:
        """Test creating note progress."""
        progress = NoteProgress(
            source_path="test.md",
            card_index=1,
            lang="en",
            status="processing",
            started_at=datetime.now(),
        )

        assert progress.source_path == "test.md"
        assert progress.card_index == 1
        assert progress.lang == "en"
        assert progress.status == "processing"
        assert progress.error is None
        assert progress.completed_at is None


class TestProgressPersistence:
    """Test progress persistence to database."""

    @pytest.fixture()
    def temp_db(self) -> None:
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            with StateDB(db_path) as db:
                yield db

    def test_save_and_load_progress(self, temp_db) -> None:
        """Test saving and loading progress."""
        # Create and save progress
        tracker1 = ProgressTracker(temp_db)
        session_id = tracker1.progress.session_id
        tracker1.set_total_notes(10)
        tracker1.set_phase(SyncPhase.SCANNING)
        tracker1.start_note("note1.md", 1, "en")
        tracker1.complete_note("note1.md", 1, "en", 1)

        # Load progress in new tracker
        tracker2 = ProgressTracker(temp_db, session_id=session_id)

        assert tracker2.progress.total_notes == 10
        assert tracker2.progress.notes_processed == 1
        assert tracker2.progress.phase == SyncPhase.SCANNING
        assert len(tracker2.progress.note_progress) == 1

    def test_multiple_sessions(self, temp_db) -> None:
        """Test multiple concurrent sessions."""
        tracker1 = ProgressTracker(temp_db)
        tracker2 = ProgressTracker(temp_db)

        assert tracker1.progress.session_id != tracker2.progress.session_id

        tracker1.set_total_notes(10)
        tracker2.set_total_notes(20)

        # Each should have independent progress
        assert tracker1.progress.total_notes == 10
        assert tracker2.progress.total_notes == 20
