"""Progress tracking for resumable sync operations."""

import signal
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, computed_field, field_validator

if TYPE_CHECKING:
    from .state_db import StateDB

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


# Maximum age for a resumable session (24 hours)
MAX_SESSION_AGE_HOURS = 24


class ResumeValidationError(Exception):
    """Error when resume state validation fails.

    Raised when a session cannot be safely resumed due to inconsistent state,
    stale session, or missing files.
    """

    def __init__(self, message: str, session_id: str, reason: str):
        super().__init__(message)
        self.session_id = session_id
        self.reason = reason


class SyncPhase(str, Enum):
    """Phases of sync operation."""

    INITIALIZING = "initializing"
    INDEXING = "indexing"
    SCANNING = "scanning"
    GENERATING = "generating"
    DETERMINING_ACTIONS = "determining_actions"
    APPLYING_CHANGES = "applying_changes"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class NoteProgress(BaseModel):
    """Progress for a single note."""

    source_path: str
    card_index: int = Field(ge=0)
    lang: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SyncProgress(BaseModel):
    """Overall sync progress state."""

    session_id: str
    phase: SyncPhase
    started_at: datetime
    updated_at: datetime
    total_notes: int = Field(default=0, ge=0)
    notes_processed: int = Field(default=0, ge=0)
    cards_generated: int = Field(default=0, ge=0)
    cards_created: int = Field(default=0, ge=0)
    cards_updated: int = Field(default=0, ge=0)
    cards_deleted: int = Field(default=0, ge=0)
    cards_restored: int = Field(default=0, ge=0)
    cards_skipped: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    completed_at: datetime | None = None
    note_progress: dict[str, NoteProgress] = Field(default_factory=dict)

    @field_validator("phase", mode="before")
    @classmethod
    def _ensure_phase_is_enum(cls, value: Any) -> SyncPhase:
        """Ensure phase is converted to SyncPhase enum."""
        if isinstance(value, SyncPhase):
            return value
        if isinstance(value, str):
            try:
                return SyncPhase(value)
            except ValueError:
                # Default to INITIALIZING for unknown values
                return SyncPhase.INITIALIZING
        return SyncPhase.INITIALIZING

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        """Check if sync is complete."""
        return self.phase in (SyncPhase.COMPLETED, SyncPhase.FAILED)

    @computed_field  # type: ignore[prop-decorator]
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
        progress_db: "StateDB",
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
        self._interrupt_event = threading.Event()  # Thread-safe interrupt flag
        self._signal_handlers_installed = False
        self._progress_lock = threading.RLock()  # Thread-safe lock for progress state

        # Load existing progress or create new session
        if session_id:
            progress = self.db.get_progress(session_id)
            if progress:
                # Validate the resume state before accepting it
                self._validate_resume_state(progress, session_id)

                self.progress = progress
                progress_pct = progress.progress_pct
                logger.info(
                    "progress_resumed",
                    session_id=session_id,
                    phase=progress.phase.value,
                    progress_pct=round(progress_pct, 1),
                    items_processed=progress.notes_processed,
                    items_total=progress.total_notes,
                    items_remaining=progress.total_notes - progress.notes_processed,
                    cards_generated=progress.cards_generated,
                )
            else:
                msg = f"No progress found for session {session_id}"
                raise ValueError(msg)
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

    def _validate_resume_state(self, progress: "SyncProgress", session_id: str) -> None:
        """Validate that a session can be safely resumed.

        Checks:
        - Session is not too old (within MAX_SESSION_AGE_HOURS)
        - Session is in a resumable phase (not FAILED or COMPLETED)
        - Progress state is consistent (processed <= total)

        Args:
            progress: The loaded progress state
            session_id: Session identifier

        Raises:
            ResumeValidationError: If validation fails
        """
        # Check session age
        session_age = datetime.now() - progress.started_at
        max_age = timedelta(hours=MAX_SESSION_AGE_HOURS)
        if session_age > max_age:
            age_hours = session_age.total_seconds() / 3600
            msg = (
                f"Session {session_id} is too old ({age_hours:.1f} hours). "
                f"Maximum age is {MAX_SESSION_AGE_HOURS} hours. "
                "Start a new sync instead."
            )
            logger.warning(
                "resume_validation_failed",
                session_id=session_id,
                reason="session_too_old",
                age_hours=round(age_hours, 1),
                max_age_hours=MAX_SESSION_AGE_HOURS,
            )
            raise ResumeValidationError(msg, session_id, "session_too_old")

        # Check phase is resumable
        non_resumable_phases = {SyncPhase.COMPLETED, SyncPhase.FAILED}
        if progress.phase in non_resumable_phases:
            msg = (
                f"Session {session_id} is in phase '{progress.phase.value}' "
                "which cannot be resumed. Start a new sync instead."
            )
            logger.warning(
                "resume_validation_failed",
                session_id=session_id,
                reason="non_resumable_phase",
                phase=progress.phase.value,
            )
            raise ResumeValidationError(msg, session_id, "non_resumable_phase")

        # Check progress consistency
        if progress.total_notes > 0 and progress.notes_processed > progress.total_notes:
            msg = (
                f"Session {session_id} has inconsistent state: "
                f"processed ({progress.notes_processed}) > total ({progress.total_notes}). "
                "Start a new sync instead."
            )
            logger.warning(
                "resume_validation_failed",
                session_id=session_id,
                reason="inconsistent_progress",
                processed=progress.notes_processed,
                total=progress.total_notes,
            )
            raise ResumeValidationError(
                msg, session_id, "inconsistent_progress")

        # Check for stale note_progress (files that no longer exist)
        missing_files = []
        for note_progress in progress.note_progress.values():
            source_path = Path(note_progress.source_path)
            if not source_path.exists():
                missing_files.append(str(source_path))

        if missing_files:
            # Log warning but don't fail - just skip missing files during resume
            logger.warning(
                "resume_validation_missing_files",
                session_id=session_id,
                missing_count=len(missing_files),
                sample_files=missing_files[:5],  # Log first 5 as sample
            )

        logger.debug(
            "resume_validation_passed",
            session_id=session_id,
            phase=progress.phase.value,
            processed=progress.notes_processed,
            total=progress.total_notes,
        )

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful interruption."""
        if self._signal_handlers_installed:
            return

        def signal_handler(signum: int, frame: Any) -> None:
            signal_name = signal.Signals(signum).name
            progress_pct = self.progress.progress_pct
            logger.warning(
                "progress_interrupted",
                session_id=self.progress.session_id,
                signal=signal_name,
                phase=self.progress.phase.value,
                progress_pct=round(progress_pct, 1),
                items_processed=self.progress.notes_processed,
                items_total=self.progress.total_notes,
                cards_generated=self.progress.cards_generated,
            )
            self._interrupt_event.set()
            self.progress.phase = SyncPhase.INTERRUPTED
            self.progress.updated_at = datetime.now()
            self.db.save_progress(self.progress)

            if self.on_interrupt_callback:
                self.on_interrupt_callback()

            sys.exit(130)  # Standard exit code for SIGINT

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._signal_handlers_installed = True
        logger.debug("signal_handlers_installed")

    def is_interrupted(self) -> bool:
        """Check if sync was interrupted (thread-safe)."""
        return self._interrupt_event.is_set()

    def set_phase(self, phase: SyncPhase) -> None:
        """Update current sync phase."""
        with self._progress_lock:
            old_phase = self.progress.phase.value
            progress_pct = self.progress.progress_pct
            logger.info(
                "progress_phase_changed",
                session_id=self.progress.session_id,
                from_phase=old_phase,
                to_phase=phase.value,
                progress_pct=round(progress_pct, 1),
                notes_processed=self.progress.notes_processed,
                total_notes=self.progress.total_notes,
            )
            self.progress.phase = phase
            self.progress.updated_at = datetime.now()
            self.db.save_progress(self.progress)

    def set_total_notes(self, total: int) -> None:
        """Set total number of notes to process."""
        with self._progress_lock:
            self.progress.total_notes = total
            self.progress.updated_at = datetime.now()
            self.db.save_progress(self.progress)
            logger.info("total_notes_set", total=total)

    def start_note(self, source_path: str, card_index: int, lang: str) -> None:
        """Mark a note as being processed."""
        with self._progress_lock:
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
        with self._progress_lock:
            key = f"{source_path}:{card_index}:{lang}"
            if key in self.progress.note_progress:
                self.progress.note_progress[key].status = "completed"
                self.progress.note_progress[key].completed_at = datetime.now()

            self.progress.notes_processed += 1
            self.progress.cards_generated += cards_generated
            self.progress.updated_at = datetime.now()

            # Check for milestones (10%, 25%, 50%, 75%, 90%)
            progress_pct = self.progress.progress_pct
            milestone_thresholds = [10, 25, 50, 75, 90]
            current_milestone = None
            for threshold in milestone_thresholds:
                if progress_pct >= threshold and progress_pct < threshold + 1:
                    current_milestone = threshold
                    break

            if current_milestone:
                logger.info(
                    "progress_milestone",
                    session_id=self.progress.session_id,
                    milestone=f"{current_milestone}%",
                    items_processed=self.progress.notes_processed,
                    items_total=self.progress.total_notes,
                    cards_generated=self.progress.cards_generated,
                )

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
        with self._progress_lock:
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
        with self._progress_lock:
            key = f"{source_path}:{card_index}:{lang}"
            if key in self.progress.note_progress:
                status: str = self.progress.note_progress[key].status
                return status == "completed"
            return False

    def is_note_failed(self, source_path: str, card_index: int, lang: str) -> bool:
        """Check if a note previously failed (skip on resume)."""
        with self._progress_lock:
            key = f"{source_path}:{card_index}:{lang}"
            if key in self.progress.note_progress:
                status: str = self.progress.note_progress[key].status
                return status == "failed"
            return False

    def increment_stat(self, stat: str, count: int = 1) -> None:
        """Increment a statistic counter."""
        with self._progress_lock:
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
        with self._progress_lock:
            self.progress.phase = SyncPhase.COMPLETED if success else SyncPhase.FAILED
            self.progress.completed_at = datetime.now()
            self.progress.updated_at = datetime.now()
            self.db.save_progress(self.progress)
            logger.info(
                "sync_completed",
                session_id=self.progress.session_id,
                success=success,
                duration=(
                    self.progress.completed_at - self.progress.started_at
                ).total_seconds(),
            )

    def get_stats(self) -> dict:
        """Get statistics dictionary."""
        with self._progress_lock:
            return {
                "processed": self.progress.notes_processed,
                "created": self.progress.cards_created,
                "updated": self.progress.cards_updated,
                "deleted": self.progress.cards_deleted,
                "restored": self.progress.cards_restored,
                "skipped": self.progress.cards_skipped,
                "errors": self.progress.errors,
            }

    def get_snapshot(self) -> SyncProgress:
        """Return a deep copy of the current progress state."""
        with self._progress_lock:
            return self.progress.model_copy(deep=True)

    def get_pending_notes(self) -> list[NoteProgress]:
        """Get list of notes that haven't been processed yet."""
        with self._progress_lock:
            return [
                note
                for note in self.progress.note_progress.values()
                if note.status in ("pending", "processing")
            ]
