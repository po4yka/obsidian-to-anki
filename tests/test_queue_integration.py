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


@pytest.mark.asyncio
async def test_scan_notes_with_queue(mock_config, mock_pool):
    """Test that scan_notes_with_queue submits jobs and collects results."""

    with patch("obsidian_anki_sync.sync.note_scanner.create_pool", return_value=mock_pool) as mock_create_pool:
        scanner = NoteScanner(mock_config)

        # Mock job
        mock_job = AsyncMock()
        mock_job.job_id = "test-job-id"
        mock_job.status.return_value = "complete"
        mock_job.result.return_value = {
            "success": True,
            "cards": [
                {
                    "slug": "test-card",
                    "front": "Front",
                    "back": "Back",
                    "tags": ["tag"],
                    "deck": "Default",
                    "model": "Basic"
                }
            ],
            "slugs": ["test-card"]
        }

        mock_pool.enqueue_job.return_value = mock_job
        mock_pool.get_job.return_value = mock_job

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
        assert mock_pool.get_job.called


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
    mock_result.generation.cards = [MagicMock()]
    ctx["orchestrator"].process_note.return_value = mock_result

    # Mock convert_to_cards
    mock_card = MagicMock()
    mock_card.slug = "test-card"
    mock_card.model_dump.return_value = {"slug": "test-card"}
    ctx["orchestrator"].convert_to_cards.return_value = [mock_card]

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value="content"):

        result = await process_note_job(
            ctx,
            "/tmp/test.md",
            "test.md"
        )

        assert result["success"] is True
        assert len(result["cards"]) == 1
        assert result["cards"][0]["slug"] == "test-card"
