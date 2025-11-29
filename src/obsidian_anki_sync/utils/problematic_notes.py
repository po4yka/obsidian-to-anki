"""Utilities for archiving problematic notes for diagnosis and repair.

This module provides functionality to copy notes that fail processing
to a dedicated folder structure for easier analysis and repair.
"""

import hashlib
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from obsidian_anki_sync.utils.fs_monitor import get_fd_limits, get_open_file_count
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ProblematicNotesArchiver:
    """Archives problematic notes with metadata for diagnosis."""

    def __init__(
        self,
        archive_dir: Path,
        enabled: bool = True,
    ):
        """Initialize the archiver.

        Args:
            archive_dir: Base directory for archived notes
            enabled: Whether archival is enabled
        """
        self.archive_dir = Path(archive_dir)
        self.enabled = enabled
        self.index_file = self.archive_dir / "index.json"

        if self.enabled:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _load_index(self) -> None:
        """Load or initialize the index of archived notes."""
        if self.index_file.exists():
            try:
                with open(self.index_file, encoding="utf-8") as f:
                    self.index = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "failed_to_load_problematic_notes_index",
                    error=str(e),
                    action="creating_new_index",
                )
                self.index = {"notes": [], "last_updated": None}
        else:
            self.index = {"notes": [], "last_updated": None}

    def _save_index(self) -> None:
        """Save the index to disk."""
        if not self.enabled:
            return

        try:
            self.index["last_updated"] = datetime.now().isoformat()
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(
                "failed_to_save_problematic_notes_index",
                error=str(e),
            )

    def _get_error_category(self, error_type: str) -> str:
        """Map error type to category directory.

        Args:
            error_type: Name of the exception class

        Returns:
            Category directory name
        """
        error_type_lower = error_type.lower()

        if "parser" in error_type_lower:
            return "parser_errors"
        elif "validation" in error_type_lower or "validator" in error_type_lower:
            return "validation_errors"
        elif "llm" in error_type_lower or "provider" in error_type_lower:
            return "llm_errors"
        elif "generation" in error_type_lower or "generator" in error_type_lower:
            return "generation_errors"
        else:
            return "other_errors"

    def archive_note(
        self,
        note_path: Path,
        error: Exception,
        error_type: str | None = None,
        processing_stage: str | None = None,
        card_index: int | None = None,
        language: str | None = None,
        note_content: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Path | None:
        """Archive a problematic note with metadata.

        Args:
            note_path: Path to the original note file
            error: The exception that occurred
            error_type: Type of error (if None, uses type(error).__name__)
            processing_stage: Stage where error occurred (parsing, validation, etc.)
            card_index: Card index if applicable
            language: Language tag if applicable
            note_content: Content of the note (if None, will be read from file)
            context: Additional context dictionary

        Returns:
            Path to archived note, or None if archival failed or disabled
        """
        if not self.enabled:
            return None

        try:
            # Determine error type
            if error_type is None:
                error_type = type(error).__name__

            # Read note content if not provided
            if note_content is None:
                try:
                    with open(note_path, encoding="utf-8") as f:
                        note_content = f.read()
                except OSError as e:
                    logger.warning(
                        "failed_to_read_note_for_archival",
                        note_path=str(note_path),
                        error=str(e),
                        **self._fd_snapshot(),
                    )
                    note_content = None

            # Compute content hash
            content_hash = (
                hashlib.sha256(note_content.encode("utf-8")).hexdigest()
                if note_content is not None
                else ""
            )

            # Create date-based directory structure
            date_str = datetime.now().strftime("%Y-%m-%d")
            category = self._get_error_category(error_type)
            date_dir = self.archive_dir / date_str / category
            date_dir.mkdir(parents=True, exist_ok=True)

            # Generate safe filename
            note_name = note_path.name
            # Avoid overwriting existing files
            counter = 0
            base_name = note_path.stem
            while (date_dir / note_name).exists():
                counter += 1
                note_name = f"{base_name}_{counter}{note_path.suffix}"

            archived_note_path = date_dir / note_name

            # Copy note file
            if not self._write_archived_note(
                source_path=note_path,
                destination_path=archived_note_path,
                note_content=note_content,
            ):
                return None

            # Create metadata file
            metadata = {
                "original_path": str(note_path),
                "archived_path": str(archived_note_path),
                "error_type": error_type,
                "error_message": str(error),
                "timestamp": datetime.now().isoformat(),
                "card_index": card_index,
                "language": language,
                "processing_stage": processing_stage,
                "content_hash": content_hash,
                "traceback": "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__)
                ),
                "context": context or {},
            }

            metadata_path = archived_note_path.with_suffix(".meta.json")
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except OSError as e:
                logger.warning(
                    "failed_to_save_note_metadata",
                    metadata_path=str(metadata_path),
                    error=str(e),
                    **self._fd_snapshot(),
                )

            # Update index
            index_entry = {
                "original_path": str(note_path),
                "archived_path": str(archived_note_path),
                "error_type": error_type,
                "timestamp": metadata["timestamp"],
                "category": category,
                "processing_stage": processing_stage,
            }
            self.index["notes"].append(index_entry)
            self._save_index()

            logger.info(
                "note_archived",
                original_path=str(note_path),
                archived_path=str(archived_note_path),
                error_type=error_type,
                category=category,
            )

            return archived_note_path

        except Exception as e:
            logger.error(
                "failed_to_archive_note",
                note_path=str(note_path),
                error=str(e),
            )
            return None

    def get_archived_notes(
        self,
        error_type: str | None = None,
        category: str | None = None,
        date: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get list of archived notes matching criteria.

        Args:
            error_type: Filter by error type
            category: Filter by category (parser_errors, validation_errors, etc.)
            date: Filter by date (YYYY-MM-DD format)
            limit: Maximum number of results to return

        Returns:
            List of index entries matching criteria
        """
        if not self.enabled or not self.index_file.exists():
            return []

        notes = self.index.get("notes", [])

        # Apply filters
        if error_type:
            notes = [n for n in notes if n.get("error_type") == error_type]

        if category:
            notes = [n for n in notes if n.get("category") == category]

        if date:
            notes = [n for n in notes if n.get(
                "timestamp", "").startswith(date)]

        # Sort by timestamp (newest first)
        notes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if limit:
            notes = notes[:limit]

        return list(notes)  # type: ignore[return-value]

    def cleanup_old_archives(self, max_age_days: int = 90) -> int:
        """Clean up archived notes older than specified days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of notes cleaned up
        """
        if not self.enabled:
            return 0

        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cleaned = 0

        # Remove old entries from index
        original_count = len(self.index.get("notes", []))
        self.index["notes"] = [
            n
            for n in self.index.get("notes", [])
            if datetime.fromisoformat(n.get("timestamp", "")).timestamp() > cutoff_date
        ]
        cleaned = original_count - len(self.index["notes"])

        # Remove old directories
        for date_dir in self.archive_dir.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date.timestamp() < cutoff_date:
                    shutil.rmtree(date_dir)
                    logger.info(
                        "removed_old_archive_directory",
                        directory=str(date_dir),
                    )
            except (ValueError, OSError):
                # Skip directories that don't match date format or can't be removed
                continue

        if cleaned > 0:
            self._save_index()

        return cleaned

    def _write_archived_note(
        self,
        source_path: Path,
        destination_path: Path,
        note_content: str | None,
    ) -> bool:
        """Write the archived note without leaking file descriptors."""
        try:
            if note_content is not None:
                destination_path.write_text(note_content, encoding="utf-8")
            else:
                with source_path.open("rb") as src, destination_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

            try:
                shutil.copystat(source_path, destination_path,
                                follow_symlinks=True)
            except OSError as stat_error:
                logger.debug(
                    "failed_to_copy_note_metadata_stat",
                    source=str(source_path),
                    destination=str(destination_path),
                    error=str(stat_error),
                )
            return True
        except OSError as e:
            logger.error(
                "failed_to_copy_note_for_archival",
                source=str(source_path),
                destination=str(destination_path),
                error=str(e),
                **self._fd_snapshot(),
            )
            return False

    @staticmethod
    def _fd_snapshot() -> dict[str, int | None]:
        """Return current file descriptor stats for diagnostics."""
        soft_limit, hard_limit = get_fd_limits()
        return {
            "open_fd_count": get_open_file_count(),
            "soft_fd_limit": soft_limit,
            "hard_fd_limit": hard_limit,
        }
