
import errno
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import sys
import os

# Add src to path if not already present
sys.path.insert(0, os.path.abspath("src"))

from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

class TestFDExhaustionFix(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.archiver_batch_size = 10
        self.config.archiver_min_fd_headroom = 32
        self.config.archiver_fd_poll_interval = 0.01

        self.state_db = MagicMock()
        self.card_generator = MagicMock()
        self.archiver = MagicMock(spec=ProblematicNotesArchiver)

        self.scanner = NoteScanner(
            config=self.config,
            state_db=self.state_db,
            card_generator=self.card_generator,
            archiver=self.archiver
        )

    @patch("obsidian_anki_sync.sync.note_scanner.has_fd_headroom")
    @patch("time.sleep")
    def test_archive_note_immediate_retry_success(self, mock_sleep, mock_has_headroom):
        # Setup mocks
        # First 2 calls fail with EMFILE, 3rd succeeds
        error = OSError(errno.EMFILE, "Too many open files")
        self.archiver.archive_note.side_effect = [error, error, True]

        # has_fd_headroom returns True (we just want to verify the call)
        mock_has_headroom.return_value = (True, {})

        # Execute
        self.scanner._archive_note_immediate(
            file_path=Path("test.md"),
            relative_path="test.md",
            error=Exception("Original error"),
            processing_stage="test"
        )

        # Verify
        # Should have called archive_note 3 times
        self.assertEqual(self.archiver.archive_note.call_count, 3)
        # Should have called wait_for_fd_headroom (which calls has_fd_headroom)
        self.assertEqual(mock_has_headroom.call_count, 2)
        # Should have slept
        # Note: _wait_for_fd_headroom calls has_fd_headroom, if it returns True immediately, it doesn't sleep.
        # But the retry loop calls _wait_for_fd_headroom.
        # Wait, my implementation of _archive_note_immediate calls _wait_for_fd_headroom()
        # _wait_for_fd_headroom calls has_fd_headroom. If it returns True, it returns immediately.
        # So mock_sleep might not be called if has_fd_headroom returns True.

    @patch("obsidian_anki_sync.sync.note_scanner.has_fd_headroom")
    def test_archive_note_immediate_retry_exhausted(self, mock_has_headroom):
        # Setup mocks
        # All calls fail with EMFILE
        error = OSError(errno.EMFILE, "Too many open files")
        self.archiver.archive_note.side_effect = error
        mock_has_headroom.return_value = (True, {})

        # Execute & Verify
        with self.assertRaises(OSError):
            self.scanner._archive_note_immediate(
                file_path=Path("test.md"),
                relative_path="test.md",
                error=Exception("Original error"),
                processing_stage="test"
            )

        # Should have retried max_retries + 1 times (initial + 3 retries = 4)
        self.assertEqual(self.archiver.archive_note.call_count, 4)

    @patch("obsidian_anki_sync.sync.note_scanner.has_fd_headroom")
    def test_process_deferred_archives_proactive_check(self, mock_has_headroom):
        # Setup
        self.scanner._defer_archival = True
        # Add 2 batches worth of items
        for i in range(20):
            self.scanner._archive_note_safely(
                file_path=Path(f"note_{i}.md"),
                relative_path=f"note_{i}.md",
                error=Exception("Test"),
                processing_stage="test"
            )

        self.scanner._defer_archival = False
        mock_has_headroom.return_value = (True, {})

        # Mock _archive_note_immediate to avoid actual calls
        with patch.object(self.scanner, "_archive_note_immediate") as mock_immediate:
            self.scanner._process_deferred_archives()

            # Should have processed all 20
            self.assertEqual(mock_immediate.call_count, 20)

            # Should have checked headroom at least twice (once per batch)
            # 20 items, batch size 10 -> 2 batches.
            # Plus maybe inside _wait_for_fd_headroom logic?
            # My implementation calls _wait_for_fd_headroom() before each batch.
            # So at least 2 calls.
            self.assertGreaterEqual(mock_has_headroom.call_count, 2)

if __name__ == "__main__":
    unittest.main()
