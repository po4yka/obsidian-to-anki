"""Tests for queue integration."""

import asyncio
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arq.connections import ArqRedis

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.worker import process_note_job


@pytest.fixture
def mock_config():
    config = Config()
    config.enable_queue = True
    config.redis_url = "redis://localhost:6379"
    return config


@pytest.fixture
def mock_pool():
    pool = AsyncMock(spec=ArqRedis)
    return pool


def test_scan_notes_with_queue(mock_config, mock_pool):
    """Test that scan_notes_with_queue submits jobs and collects results."""

    # Mock job returned by enqueue_job
    mock_job = MagicMock()
    mock_job.job_id = "test-job-id"

    # Mock Job class instance for status checking
    mock_job_instance = MagicMock()
    mock_job_instance.status = AsyncMock(return_value="complete")
    mock_job_instance.result = AsyncMock(return_value={
        "success": True,
        "cards": [
            {
                "slug": "test-card",
                "lang": "en",
                "apf_html": "<!-- test -->",
                "manifest": {
                    "slug": "test-card",
                    "slug_base": "test-card",
                    "lang": "en",
                    "source_path": "/tmp/test.md",
                    "source_anchor": "#card-1",
                    "note_id": "test",
                    "note_title": "Test",
                    "card_index": 0,
                    "guid": "guid",
                    "hash6": None
                },
                "content_hash": "hash123",
                "note_type": "APF::Simple",
                "tags": ["tag"],
                "guid": "guid"
            }
        ],
        "slugs": ["test-card"]
    })

    mock_pool.enqueue_job.return_value = mock_job

    with patch("obsidian_anki_sync.sync.note_scanner.create_pool", return_value=mock_pool), \
         patch("obsidian_anki_sync.sync.note_scanner.Job", return_value=mock_job_instance) as mock_job_class:
        from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver
        archiver = ProblematicNotesArchiver(Path("/tmp"), enabled=True)
        scanner = NoteScanner(
            config=mock_config,
            state_db=MagicMock(),
            card_generator=MagicMock(),
            archiver=archiver,
        )

        # Test data
        note_files = [(Path("/tmp/test.md"), "test.md")]
        obsidian_cards = {}
        existing_slugs = set()
        error_by_type = defaultdict(int)
        error_samples = defaultdict(list)

        # Run scan
        result = scanner.scan_notes_with_queue(
            note_files,
            obsidian_cards,
            existing_slugs,
            error_by_type,
            error_samples
        )

        # Verify
        assert len(result) == 1
        assert "test-card" in result
        assert mock_pool.enqueue_job.called
        # Verify Job class was instantiated with correct parameters
        mock_job_class.assert_called_with(job_id="test-job-id", redis=mock_pool)


@pytest.mark.asyncio
async def test_process_note_job():
    """Test worker job processing."""

    ctx = {
        "config": Config(),
        "orchestrator": AsyncMock()
    }

    # Mock orchestrator result
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.generation = MagicMock()
    mock_result.generation.cards = [MagicMock()]
    ctx["orchestrator"].process_note.return_value = mock_result

    # Mock convert_to_cards
    mock_card = MagicMock()
    mock_card.slug = "test-card"
    mock_card.model_dump.return_value = {"slug": "test-card"}
    # Make convert_to_cards synchronous (not async)
    ctx["orchestrator"].convert_to_cards = MagicMock(return_value=[mock_card])

    with patch("pathlib.Path.exists", return_value=True), \
            patch("pathlib.Path.is_file", return_value=True), \
            patch("pathlib.Path.read_text", return_value="content"):

        # Provide dummy metadata and qa_pairs to avoid file parsing
        from datetime import datetime
        now = datetime.now()
        metadata_dict = {
            "id": "test",
            "title": "Test",
            "topic": "Testing",
            "created": now.isoformat(),
            "updated": now.isoformat()
        }
        qa_pairs_dicts = [{
            "question_en": "Q",
            "question_ru": "Q",
            "answer_en": "A",
            "answer_ru": "A",
            "card_index": 1
        }]

        result = await process_note_job(
            ctx,
            "/tmp/test.md",
            "test.md",
            metadata_dict,
            qa_pairs_dicts
        )

        assert result["success"] is True
        assert len(result["cards"]) == 1
        assert result["cards"][0]["slug"] == "test-card"
