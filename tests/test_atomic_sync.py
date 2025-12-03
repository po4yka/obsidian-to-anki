from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.models import Card, Manifest, SyncAction
from obsidian_anki_sync.sync.engine import SyncEngine


@pytest.fixture
def mock_components():
    config = MagicMock()
    config.vault_path = Path("/tmp/vault")
    config.source_dir = "notes"
    config.source_subdirs = []
    config.max_concurrent_generations = 1
    config.auto_adjust_workers = False
    config.use_langgraph = False
    config.use_pydantic_ai = False

    state_db = MagicMock()
    state_db.get_all_cards.return_value = []

    note_scanner = MagicMock()
    anki_state_manager = MagicMock()
    change_applier = MagicMock()

    return config, state_db, note_scanner, anki_state_manager, change_applier


def test_atomic_sync_callback(mock_components):
    config, state_db, _note_scanner, _anki_state_manager, _change_applier = (
        mock_components
    )

    # Patch the classes that SyncEngine instantiates internally
    with (
        patch("obsidian_anki_sync.sync.engine.NoteScanner") as MockNoteScanner,
        patch(
            "obsidian_anki_sync.sync.engine.AnkiStateManager"
        ) as MockAnkiStateManager,
        patch("obsidian_anki_sync.sync.engine.ChangeApplier") as MockChangeApplier,
        patch("obsidian_anki_sync.sync.engine.CardGenerator") as MockCardGenerator,
        patch("obsidian_anki_sync.sync.engine.APFGenerator") as MockAPFGenerator,
    ):
        # Setup mock instances
        note_scanner_instance = MockNoteScanner.return_value
        anki_state_manager_instance = MockAnkiStateManager.return_value
        change_applier_instance = MockChangeApplier.return_value

        # Mock fetch_state
        anki_state_manager_instance.fetch_state.return_value = {}

        # Mock scan_notes to invoke the callback
        def mock_scan_notes(**kwargs):
            on_batch_complete = kwargs.get("on_batch_complete")
            assert on_batch_complete is not None

            # Simulate card generation
            card = Card(
                slug="test-card",
                lang="en",
                apf_html="<div>Test</div>",
                manifest=Manifest(
                    slug="test-card",
                    slug_base="test-card",
                    lang="en",
                    source_path="test.md",
                    source_anchor="anchor",
                    note_id="note-id",
                    note_title="Test Note",
                    card_index=0,
                    guid="guid",
                ),
                content_hash="hash",
                note_type="APF::Simple",
                tags=[],
                guid="guid",
            )

            # Invoke callback
            on_batch_complete([card])

            return {"test-card": card}

        note_scanner_instance.scan_notes.side_effect = mock_scan_notes

        # Mock determine_actions to generate a create action
        def mock_determine_actions(
            obs_cards, anki_cards, changes, db_cards_override=None
        ):
            for card in obs_cards.values():
                changes.append(SyncAction(type="create", card=card, reason="Test"))

        anki_state_manager_instance.determine_actions.side_effect = (
            mock_determine_actions
        )

        # Initialize engine
        engine = SyncEngine(
            config=config,
            state_db=state_db,
            anki_client=MagicMock(),
        )

        # Run sync
        engine.sync()

        # Verify
        # 1. fetch_state called before scan_notes
        assert anki_state_manager_instance.fetch_state.called
        assert note_scanner_instance.scan_notes.called

        # 2. apply_changes called via callback
        assert change_applier_instance.apply_changes.called
        assert len(change_applier_instance.apply_changes.call_args_list) >= 1

        # Check arguments of apply_changes
        args, _ = change_applier_instance.apply_changes.call_args
        changes = args[0]
        assert len(changes) == 1
        assert changes[0].type == "create"
        assert changes[0].card.slug == "test-card"
