"""Archival service for problematic notes."""

import errno
import threading
from pathlib import Path
from typing import Any

from obsidian_anki_sync.domain.interfaces.note_scanner import IArchiver
from obsidian_anki_sync.sync.scanner_utils import wait_for_fd_headroom
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class NoteArchiver(IArchiver):
    """Service for archiving problematic notes with deferred processing."""

    def __init__(
        self,
        archiver: Any,
        batch_size: int = 64,
        fd_headroom: int = 32,
        fd_poll_interval: float = 0.05,
    ):
        """Initialize the note archiver.

        Args:
            archiver: ProblematicNotesArchiver instance
            batch_size: Size of archival batches
            fd_headroom: Minimum file descriptor headroom required
            fd_poll_interval: Interval for FD headroom polling
        """
        self.archiver = archiver
        self._archival_lock = threading.Lock()
        self._deferred_archives: list[dict[str, Any]] = []
        self._defer_archival = False
        self._archiver_batch_size = max(1, batch_size)
        self._archiver_fd_headroom = max(1, fd_headroom)
        self._archiver_fd_poll_interval = max(0.01, fd_poll_interval)

    def set_defer_archival(self, defer: bool) -> None:
        """Set whether to defer archival operations.

        Args:
            defer: If True, defer archival to prevent FD exhaustion
        """
        self._defer_archival = defer

    def archive_note_safely(
        self,
        file_path: Path,
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Safely archive a problematic note.

        When _defer_archival is True (during parallel scans), archival requests
        are queued and processed sequentially after the scan completes.
        This prevents "too many open files" errors from concurrent file operations.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred
            note_content: Optional note content
            card_index: Optional card index
            language: Optional language
        """
        # During parallel scans, defer archival to prevent file descriptor exhaustion
        if self._defer_archival:
            with self._archival_lock:
                self._deferred_archives.append(
                    {
                        "file_path": file_path,
                        "relative_path": relative_path,
                        "error": error,
                        "processing_stage": processing_stage,
                        "note_content": note_content,
                        "card_index": card_index,
                        "language": language,
                    }
                )
            return

        # Immediate archival (non-parallel mode)
        self._archive_note_immediate(
            file_path=file_path,
            relative_path=relative_path,
            error=error,
            processing_stage=processing_stage,
            note_content=note_content,
            card_index=card_index,
            language=language,
        )

    def _archive_note_immediate(
        self,
        file_path: Path,
        relative_path: str,
        error: Exception,
        processing_stage: str,
        note_content: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
    ) -> None:
        """Immediately archive a problematic note.

        Args:
            file_path: Absolute path to the note file
            relative_path: Relative path for logging
            error: The exception that caused the failure
            processing_stage: Stage where error occurred
            note_content: Optional note content
            card_index: Optional card index
            language: Optional language
        """
        try:
            if note_content is None:
                try:
                    note_content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as read_err:
                    logger.debug(
                        "unable_to_read_note_for_archiving",
                        file=relative_path,
                        error=str(read_err),
                    )
                    note_content = None

            # Retry logic for FD exhaustion
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    self.archiver.archive_note(
                        note_path=file_path,
                        error=error,
                        error_type=type(error).__name__,
                        processing_stage=processing_stage,
                        card_index=card_index,
                        language=language,
                        note_content=note_content if note_content is not None else None,
                        context={"relative_path": relative_path},
                    )
                    return  # Success
                except OSError as e:
                    # Check for "Too many open files" (EMFILE) or "File table overflow" (ENFILE)
                    if e.errno in (errno.EMFILE, errno.ENFILE):
                        if attempt < max_retries:
                            logger.warning(
                                "archival_fd_exhaustion_retry",
                                file=relative_path,
                                attempt=attempt + 1,
                                max_retries=max_retries,
                                error=str(e),
                            )
                            # Wait for headroom before retrying
                            wait_for_fd_headroom(
                                required_headroom=self._archiver_fd_headroom,
                                poll_interval=self._archiver_fd_poll_interval,
                            )
                            continue
                    raise  # Re-raise if not FD error or retries exhausted

        except Exception as archive_error:
            logger.warning(
                "failed_to_archive_problematic_note",
                note_path=str(file_path),
                archive_error=str(archive_error),
            )

    def process_deferred_archives(self) -> None:
        """Process all deferred archival requests sequentially.

        Called after parallel scan completes to archive failed notes
        without risking file descriptor exhaustion.
        """
        with self._archival_lock:
            deferred_count = len(self._deferred_archives)
            archives_to_process = self._deferred_archives.copy()
            self._deferred_archives.clear()

        if deferred_count == 0:
            return

        logger.info(
            "processing_deferred_archives",
            count=deferred_count,
            batch_size=self._archiver_batch_size,
        )

        archived_count = 0
        for batch_start in range(0, deferred_count, self._archiver_batch_size):
            # Proactively check for headroom before starting a batch
            wait_for_fd_headroom(
                required_headroom=self._archiver_fd_headroom,
                poll_interval=self._archiver_fd_poll_interval,
            )

            batch = archives_to_process[
                batch_start : batch_start + self._archiver_batch_size
            ]

            for archive_request in batch:
                file_path: Path = archive_request["file_path"]
                note_content = archive_request["note_content"]
                if not file_path.exists() and note_content is None:
                    logger.warning(
                        "skipping_deferred_archive_missing_source",
                        file=str(file_path),
                        relative_path=archive_request["relative_path"],
                    )
                    continue

                self._archive_note_immediate(
                    file_path=file_path,
                    relative_path=archive_request["relative_path"],
                    error=archive_request["error"],
                    processing_stage=archive_request["processing_stage"],
                    note_content=note_content,
                    card_index=archive_request["card_index"],
                    language=archive_request["language"],
                )
                archived_count += 1

                if archived_count % 100 == 0:
                    logger.info(
                        "deferred_archive_progress",
                        processed=archived_count,
                        total=deferred_count,
                    )

        logger.info(
            "deferred_archives_completed",
            archived=archived_count,
        )
