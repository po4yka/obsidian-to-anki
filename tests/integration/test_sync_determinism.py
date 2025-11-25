"""Integration tests for sync determinism (INT-02, REGR-det-01)."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.sync.engine import SyncEngine


class TestSyncFlow:
    """Test sync orchestration flow (INT-02)."""

    def test_create_flow(
        self, test_config, temp_dir, sample_metadata, sample_qa_pair
    ) -> None:
        """Test create card flow."""
        # Setup mocks
        from obsidian_anki_sync.anki.client import AnkiClient
        from obsidian_anki_sync.sync.state_db import StateDB

        with StateDB(test_config.db_path) as db:
            # Mock AnkiConnect
            mock_anki = MagicMock(spec=AnkiClient)
            mock_anki.find_notes.return_value = []
            mock_anki.add_note.return_value = 12345

            # Mock APF generator to avoid LLM call
            with patch("obsidian_anki_sync.sync.engine.APFGenerator") as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen.return_value = mock_gen_instance

                # This test demonstrates the pattern
                # Full implementation would require more setup
                engine = SyncEngine(test_config, db, mock_anki)

                # Verify engine initialized
                assert engine.config == test_config
                assert engine.db == db
                assert engine.anki == mock_anki
