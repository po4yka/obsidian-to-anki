"""Progress tracking for resumable sync operations."""

import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SyncPhase(Enum):
    """Phases of sync operation."""

    INITIALIZING = "initializing"
    SCANNING = "scanning"
    GENERATING = "generating"
    DETERMINING_ACTIONS = "determining_actions"
    APPLYING_CHANGES = "applying_changes"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class NoteProgress:
    """Progress for a single note."""

    source_path: str
    card_index: int
    lang: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class SyncProgress:
    """Overall sync progress state."""

    session_id: str
    phase: SyncPhase
    started_at: datetime
    updated_at: datetime
    total_notes: int = 0
    notes_processed: int = 0
    cards_generated: int = 0
    cards_created: int = 0
    cards_updated: int = 0
    cards_deleted: int = 0
    cards_restored: int = 0
    cards_skipped: int = 0
    errors: int = 0
    completed_at: datetime | None = None
    note_progress: dict[str, NoteProgress] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if sync is complete."""
        return self.phase in (SyncPhase.COMPLETED, SyncPhase.FAILED)

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total_notes == 0:
            return 0.0
        return (self.notes_processed / self.total_notes) * 100.0


class ProgressTracker:
    """Track and persist sync progress for resumability."""

    def __init__(
        self,
        progress_db,
        session_id: str | None = None,
        on_interrupt: Callable[[], None] | None = None,
    ):
        """
        Initialize progress tracker.

        Args:
            progress_db: Progress database instance
            session_id: Optional session ID for resuming
            on_interrupt: Optional callback for interruption
        """
        self.db = progress_db
        self.on_interrupt_callback = on_interrupt
        self._interrupted = False
        self._signal_handlers_installed = False

        # Load existing progress or create new session
        if session_id:
            progress = self.db.get_progress(session_id)
            if progress:
                self.progress = progress
                logger.info(
                    "resuming_sync",
                    session_id=session_id,
                    phase=progress.phase.value,
                    notes_processed=progress.notes_processed,
                    total_notes=progress.total_notes,
                )
            else:
                raise ValueError(f"No progress found for session {session_id}")
        else:
            # Create new session
            import uuid

            session_id = str(uuid.uuid4())
            self.progress = SyncProgress(
                session_id=session_id,
                phase=SyncPhase.INITIALIZING,
                started_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.db.save_progress(self.progress)
            logger.info("new_sync_session", session_id=session_id)

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful interruption."""
        if self._signal_handlers_installed:
            return

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.warning(
                "sync_interrupted",
                signal=signal_name,
                session_id=self.progress.session_id,
            )
            self._interrupted = True
            self.progress.phase = SyncPhase.INTERRUPTED
            self.progress.updated_at = datetime.now()
            self.db.save_progress(self.progress)

            if self.on_interrupt_callback:
                self.on_interrupt_callback()

            print(
                f"\n\nSync interrupted! Progress has been saved (session: {self.progress.session_id})"
            )
            print(
                f"Resume with: obsidian-anki-sync sync --resume {self.progress.session_id}\n"
            )
            sys.exit(130)  # Standard exit code for SIGINT

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._signal_handlers_installed = True
        logger.debug("signal_handlers_installed")

    def is_interrupted(self) -> bool:
        """Check if sync was interrupted."""
        return self._interrupted

    def set_phase(self, phase: SyncPhase) -> None:
        """Update current sync phase."""
        logger.debug("sync_phase_changed", from_phase=self.progress.phase.value, to_phase=phase.value)
        self.progress.phase = phase
        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)

    def set_total_notes(self, total: int) -> None:
        """Set total number of notes to process."""
        self.progress.total_notes = total
        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)
        logger.info("total_notes_set", total=total)

    def start_note(self, source_path: str, card_index: int, lang: str) -> None:
        """Mark a note as being processed."""
        key = f"{source_path}:{card_index}:{lang}"
        note_progress = NoteProgress(
            source_path=source_path,
            card_index=card_index,
            lang=lang,
            status="processing",
            started_at=datetime.now(),
        )
        self.progress.note_progress[key] = note_progress
        self.progress.updated_at = datetime.now()
        logger.debug("note_processing_started", source_path=source_path)

    def complete_note(
        self, source_path: str, card_index: int, lang: str, cards_generated: int = 1
    ) -> None:
        """Mark a note as completed."""
        key = f"{source_path}:{card_index}:{lang}"
        if key in self.progress.note_progress:
            self.progress.note_progress[key].status = "completed"
            self.progress.note_progress[key].completed_at = datetime.now()

        self.progress.notes_processed += 1
        self.progress.cards_generated += cards_generated
        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)
        logger.debug(
            "note_completed",
            source_path=source_path,
            progress=f"{self.progress.notes_processed}/{self.progress.total_notes}",
        )

    def fail_note(
        self, source_path: str, card_index: int, lang: str, error: str
    ) -> None:
        """Mark a note as failed."""
        key = f"{source_path}:{card_index}:{lang}"
        if key in self.progress.note_progress:
            self.progress.note_progress[key].status = "failed"
            self.progress.note_progress[key].error = error
            self.progress.note_progress[key].completed_at = datetime.now()

        self.progress.errors += 1
        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)
        logger.error("note_failed", source_path=source_path, error=error)

    def is_note_completed(self, source_path: str, card_index: int, lang: str) -> bool:
        """Check if a note was already processed."""
        key = f"{source_path}:{card_index}:{lang}"
        if key in self.progress.note_progress:
            return self.progress.note_progress[key].status == "completed"
        return False

    def increment_stat(self, stat: str, count: int = 1) -> None:
        """Increment a statistic counter."""
        if stat == "created":
            self.progress.cards_created += count
        elif stat == "updated":
            self.progress.cards_updated += count
        elif stat == "deleted":
            self.progress.cards_deleted += count
        elif stat == "restored":
            self.progress.cards_restored += count
        elif stat == "skipped":
            self.progress.cards_skipped += count
        elif stat == "errors":
            self.progress.errors += count

        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)

    def complete(self, success: bool = True) -> None:
        """Mark sync as completed."""
        self.progress.phase = SyncPhase.COMPLETED if success else SyncPhase.FAILED
        self.progress.completed_at = datetime.now()
        self.progress.updated_at = datetime.now()
        self.db.save_progress(self.progress)
        logger.info(
            "sync_completed",
            session_id=self.progress.session_id,
            success=success,
            duration=(self.progress.completed_at - self.progress.started_at).total_seconds(),
        )

    def get_stats(self) -> dict:
        """Get statistics dictionary."""
        return {
            "processed": self.progress.notes_processed,
            "created": self.progress.cards_created,
            "updated": self.progress.cards_updated,
            "deleted": self.progress.cards_deleted,
            "restored": self.progress.cards_restored,
            "skipped": self.progress.cards_skipped,
            "errors": self.progress.errors,
        }

    def get_pending_notes(self) -> list[NoteProgress]:
        """Get list of notes that haven't been processed yet."""
        return [
            note
            for note in self.progress.note_progress.values()
            if note.status in ("pending", "processing")
        ]
