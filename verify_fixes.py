import logging
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from obsidian_anki_sync.infrastructure.cache.cache_manager import CacheManager
from obsidian_anki_sync.sync.transactions import CardTransaction, RollbackAction

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)


class TestFixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_cache_fix")
        self.test_dir.mkdir(exist_ok=True)
        self.db_path = self.test_dir / "test.db"
        self.db_path.touch()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_cache_manager_os_error(self):
        """Verify CacheManager handles OSError properly (not silent)."""
        print("\nTesting CacheManager OSError handling...")

        cm = CacheManager(self.db_path)

        # Create a dummy file to trigger the loop
        (cm.cache_dir / "agent_cards").mkdir(parents=True, exist_ok=True)
        (cm.cache_dir / "agent_cards" / "test.txt").touch()

        # Mock rglob to return a path that raises OSError on stat
        mock_path = MagicMock()
        mock_path.is_file.return_value = True
        mock_path.stat.side_effect = OSError("Permission denied")

        with patch("pathlib.Path.rglob", return_value=[mock_path]):
            # This should NOT raise exception but SHOULD log warning
            with self.assertLogs(
                "obsidian_anki_sync.infrastructure.cache.cache_manager", level="WARNING"
            ) as cm_logs:
                info = cm.get_cache_size_info()

            print("Logs captured:", cm_logs.output)
            self.assertTrue(
                any("error_calculating_directory_size" in log for log in cm_logs.output)
            )
            self.assertEqual(info["agent_cache_size"], 0)

    def test_transaction_rollback_verification_failure(self):
        """Verify transaction rollback reports unverified on error."""
        print("\nTesting Transaction Rollback Verification failure...")

        anki_mock = MagicMock()
        db_mock = MagicMock()

        txn = CardTransaction(anki_mock, db_mock)

        # Simulate a delete_anki_note action
        note_id = 123
        txn.rollback_actions.append(("delete_anki_note", note_id))

        # Mock delete_notes to succeed
        anki_mock.delete_notes.return_value = None

        # Mock notes_info to RAISE exception (simulating network error during verification)
        anki_mock.notes_info.side_effect = Exception("Network error")

        # Perform rollback
        with self.assertLogs(
            "obsidian_anki_sync.sync.transactions", level="WARNING"
        ) as txn_logs:
            report = txn.rollback(verify=True)

        print("Logs captured:", txn_logs.output)

        # Check results
        self.assertTrue(report.all_succeeded)  # The action itself succeeded
        self.assertFalse(report.all_verified)  # But verification failed
        self.assertEqual(report.verified, 0)

        # Check logs
        self.assertTrue(
            any("rollback_verification_error" in log for log in txn_logs.output)
        )


if __name__ == "__main__":
    unittest.main()
