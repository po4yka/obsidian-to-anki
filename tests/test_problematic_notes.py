"""Tests for problematic notes archival system."""

from __future__ import annotations

import errno
import json
from contextlib import suppress
from pathlib import Path

import pytest

from obsidian_anki_sync.exceptions import ParserError, ValidationError
from obsidian_anki_sync.utils.fs_monitor import get_open_file_count
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

try:  # pragma: no cover - platform specific
    import resource
except ImportError:  # pragma: no cover - windows
    resource = None  # type: ignore[assignment]


@pytest.fixture
def temp_dir(tmp_path):
    """Alias for pytest's tmp_path fixture."""
    return tmp_path


@pytest.fixture
def sample_note_content():
    """Sample note content for archiver tests."""
    return """---
id: test-001
title: Test Question
topic: Testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Question (EN)

What is unit testing?

# Answer (EN)

Unit testing is testing individual components.
"""


class TestProblematicNotesArchiver:
    """Test problematic notes archival functionality."""

    def test_archiver_initialization(self, temp_dir):
        """Test archiver initialization."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        assert archiver.archive_dir == archive_dir
        assert archiver.enabled is True
        assert archive_dir.exists()
        # Index file is created lazily when saving, not on initialization
        assert archiver.index_file == archive_dir / "index.json"

    def test_archiver_disabled(self, temp_dir):
        """Test archiver when disabled."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=False)

        assert archiver.enabled is False
        result = archiver.archive_note(
            note_path=Path("test.md"),
            error=ParserError("Test error"),
        )
        assert result is None

    def test_archive_note_parser_error(self, temp_dir, sample_note_content):
        """Test archiving a note with parser error."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        note_path = temp_dir / "test_note.md"
        note_path.write_text(sample_note_content, encoding="utf-8")

        error = ParserError("Missing required field: language_tags")
        archived_path = archiver.archive_note(
            note_path=note_path,
            error=error,
            processing_stage="parsing",
        )

        assert archived_path is not None
        assert archived_path.exists()
        assert archived_path.name == "test_note.md"

        # Check metadata file
        metadata_path = archived_path.with_suffix(".meta.json")
        assert metadata_path.exists()

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["error_type"] == "ParserError"
        assert metadata["error_message"] == "Missing required field: language_tags"
        assert metadata["processing_stage"] == "parsing"
        assert metadata["original_path"] == str(note_path)
        assert "content_hash" in metadata
        assert "traceback" in metadata

        # Check category directory
        assert "parser_errors" in str(archived_path)

    def test_archive_note_with_context(self, temp_dir, sample_note_content):
        """Test archiving with additional context."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        note_path = temp_dir / "test_note.md"
        note_path.write_text(sample_note_content, encoding="utf-8")

        error = ValidationError("Validation failed")
        archived_path = archiver.archive_note(
            note_path=note_path,
            error=error,
            processing_stage="validation",
            card_index=1,
            language="en",
            context={"custom_field": "custom_value"},
        )

        assert archived_path is not None

        metadata_path = archived_path.with_suffix(".meta.json")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["card_index"] == 1
        assert metadata["language"] == "en"
        assert metadata["context"]["custom_field"] == "custom_value"

    def test_get_archived_notes(self, temp_dir, sample_note_content):
        """Test retrieving archived notes."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        # Archive multiple notes
        for i in range(3):
            note_path = temp_dir / f"test_note_{i}.md"
            note_path.write_text(sample_note_content, encoding="utf-8")

            error = ParserError(f"Error {i}")
            archiver.archive_note(
                note_path=note_path,
                error=error,
                processing_stage="parsing",
            )

        # Get all archived notes
        notes = archiver.get_archived_notes()
        assert len(notes) == 3

        # Filter by error type
        parser_notes = archiver.get_archived_notes(error_type="ParserError")
        assert len(parser_notes) == 3

        # Filter by category
        parser_category = archiver.get_archived_notes(category="parser_errors")
        assert len(parser_category) == 3

        # Limit results
        limited = archiver.get_archived_notes(limit=2)
        assert len(limited) == 2

    def test_error_category_mapping(self, temp_dir):
        """Test error type to category mapping."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        note_path = temp_dir / "test.md"
        note_path.write_text("test", encoding="utf-8")

        # Test different error types
        test_cases = [
            (ParserError("test"), "parser_errors"),
            (ValidationError("test"), "validation_errors"),
        ]

        for error, expected_category in test_cases:
            archived_path = archiver.archive_note(
                note_path=note_path,
                error=error,
            )
            assert expected_category in str(archived_path)

    def test_cleanup_old_archives(self, temp_dir, sample_note_content):
        """Test cleanup of old archived notes."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)

        # Archive a note
        note_path = temp_dir / "test.md"
        note_path.write_text(sample_note_content, encoding="utf-8")
        archiver.archive_note(
            note_path=note_path,
            error=ParserError("test"),
        )

        # Cleanup with long retention (should keep recent notes)
        cleaned = archiver.cleanup_old_archives(max_age_days=30)
        assert cleaned == 0  # Recent notes should not be cleaned

    def test_archive_note_prefers_provided_content(
        self, temp_dir, sample_note_content, monkeypatch
    ):
        """Ensure we do not reopen the source file when content is already available."""
        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)
        note_path = temp_dir / "prefetched.md"
        note_path.write_text("stale content", encoding="utf-8")

        captured: dict[str, str | None] = {}

        def fake_write(self, source_path, destination_path, note_content):
            captured["note_content"] = note_content
            destination_path.write_text(note_content or "", encoding="utf-8")
            return True

        monkeypatch.setattr(
            ProblematicNotesArchiver,
            "_write_archived_note",
            fake_write,
            raising=False,
        )

        error = ParserError("prefetch test")
        archiver.archive_note(
            note_path=note_path,
            error=error,
            processing_stage="testing",
            note_content=sample_note_content,
        )

        assert captured["note_content"] == sample_note_content

    @pytest.mark.skipif(resource is None, reason="resource module unavailable")
    def test_archiver_handles_low_fd_headroom(
        self, temp_dir, sample_note_content
    ) -> None:
        """Ensure archival succeeds when most file descriptors are already in use."""
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit < 64:
            pytest.skip("soft limit too low for FD exhaustion test")

        target_soft = max(32, min(soft_limit - 1, 128))
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard_limit))

        archive_dir = temp_dir / "problematic_notes"
        archiver = ProblematicNotesArchiver(
            archive_dir=archive_dir, enabled=True)
        note_path = temp_dir / "fd_stress_note.md"
        note_path.write_text(sample_note_content, encoding="utf-8")

        open_handles: list[object] = []
        exhausted = False
        try:
            for i in range(target_soft * 2):
                try:
                    handle = (temp_dir / f"holder_{i}.tmp").open("wb")
                    open_handles.append(handle)
                except OSError as exc:
                    if exc.errno == errno.EMFILE:
                        exhausted = True
                        break
                    raise

            if not exhausted:
                pytest.skip(
                    "Unable to exhaust file descriptors on this platform")

            # Free minimal headroom for the archival copy operation.
            for _ in range(min(4, len(open_handles))):
                open_handles.pop().close()

            open_count = get_open_file_count()
            assert open_count is None or (target_soft - open_count) <= 8

            error = ParserError("fd test")
            archived_path = archiver.archive_note(
                note_path=note_path,
                error=error,
                processing_stage="testing",
                note_content=sample_note_content,
            )
            assert archived_path is not None
        finally:
            for handle in open_handles:
                with suppress(OSError):
                    handle.close()
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (soft_limit, hard_limit))
