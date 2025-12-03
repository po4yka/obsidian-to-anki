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
    # Add ping method for health check
    pool.ping = AsyncMock(return_value=True)
    # Add close method
    pool.close = AsyncMock(return_value=None)
    # Add blpop method
    pool.blpop = AsyncMock()
    # Add rpush method
    pool.rpush = AsyncMock()
    # Add expire method
    pool.expire = AsyncMock()
    # Add delete method
    pool.delete = AsyncMock()
    return pool


def test_scan_notes_with_queue(mock_config, mock_pool):
    """Test that scan_notes_with_queue submits jobs and collects results via BLPOP."""

    # Mock job returned by enqueue_job
    mock_job = MagicMock()
    mock_job.job_id = "test-job-id"

    mock_pool.enqueue_job.return_value = mock_job

    # Mock BLPOP return value
    # BLPOP returns (key, value)
    import json

    result_payload = {
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
                    "hash6": None,
                },
                "content_hash": "hash123",
                "note_type": "APF::Simple",
                "tags": ["tag"],
                "guid": "guid",
            }
        ],
        "slugs": ["test-card"],
    }

    # First call returns result, second call returns None (timeout) to break loop if we were looping on timeout
    # But our loop breaks when pending_jobs is empty.
    # We have 1 job, so 1 result should clear it.
    mock_pool.blpop.side_effect = [
        (b"queue", json.dumps(result_payload).encode("utf-8")),
        None,
    ]

    # create_pool is async, so we need to return a coroutine that returns the mock_pool
    async def mock_create_pool(*args, **kwargs):
        return mock_pool

    with (
        patch(
            "obsidian_anki_sync.sync.note_scanner.create_pool",
            side_effect=mock_create_pool,
        ),
        # We don't need to patch Job class anymore as we don't use it for status checks
    ):
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
            note_files, obsidian_cards, existing_slugs, error_by_type, error_samples
        )

        # Verify
        assert len(result) == 1
        assert "test-card" in result
        assert mock_pool.enqueue_job.called
        assert mock_pool.blpop.called

        # Verify enqueue was called with result_queue_name
        call_kwargs = mock_pool.enqueue_job.call_args[1]
        assert "result_queue_name" in call_kwargs
        assert "obsidian_anki_sync:results:" in call_kwargs["result_queue_name"]


@pytest.mark.asyncio
async def test_process_note_job():
    """Test worker job processing pushes to Redis."""

    # Mock Redis in context
    mock_redis = AsyncMock()
    ctx = {"config": Config(), "orchestrator": AsyncMock(),
           "redis": mock_redis}

    # Mock orchestrator result
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.generation = MagicMock()
    mock_result.generation.cards = [MagicMock()]
    mock_result.stage_times = {"generation": 1.0, "validation": 0.5}
    ctx["orchestrator"].process_note.return_value = mock_result

    # Mock convert_to_cards
    mock_card = MagicMock()
    mock_card.slug = "test-card"
    mock_card.model_dump.return_value = {"slug": "test-card"}
    # Make convert_to_cards synchronous (not async)
    ctx["orchestrator"].convert_to_cards = MagicMock(return_value=[mock_card])

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.read_text", return_value="content"),
    ):
        # Provide dummy metadata and qa_pairs to avoid file parsing
        from datetime import datetime

        now = datetime.now()
        metadata_dict = {
            "id": "test",
            "title": "Test",
            "topic": "Testing",
            "created": now.isoformat(),
            "updated": now.isoformat(),
        }
        qa_pairs_dicts = [
            {
                "question_en": "Q",
                "question_ru": "Q",
                "answer_en": "A",
                "answer_ru": "A",
                "card_index": 1,
            }
        ]

        result_queue = "test-queue"
        result = await process_note_job(
            ctx,
            "/tmp/test.md",
            "test.md",
            metadata_dict,
            qa_pairs_dicts,
            result_queue_name=result_queue,
        )

        assert result["success"] is True
        assert len(result["cards"]) == 1
        assert result["cards"][0]["slug"] == "test-card"

        # Verify push to Redis
        assert mock_redis.rpush.called
        assert mock_redis.rpush.call_args[0][0] == result_queue
        import json

        payload = json.loads(mock_redis.rpush.call_args[0][1])
        assert payload["success"] is True
        assert payload["cards"][0]["slug"] == "test-card"


@pytest.mark.asyncio
async def test_process_note_job_fallbacks_when_redis_missing():
    """Worker should create a temporary Redis connection when ctx lacks one."""

    # Configure orchestrator stub
    ctx = {"config": Config(), "orchestrator": AsyncMock()}

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.generation.cards = [MagicMock()]
    mock_result.generation.total_cards = 1
    mock_result.stage_times = {"generation": 1.0, "validation": 0.5}
    mock_result.retry_count = 0
    mock_result.total_time = 1.5
    mock_result.last_error = None
    ctx["orchestrator"].process_note.return_value = mock_result

    mock_card = MagicMock()
    mock_card.slug = "test-card"
    mock_card.model_dump.return_value = {"slug": "test-card"}
    ctx["orchestrator"].convert_to_cards = MagicMock(return_value=[mock_card])

    metadata_dict = {
        "id": "fallback-test",
        "title": "Test",
        "topic": "Testing",
        "created": "2025-12-03T00:00:00Z",
        "updated": "2025-12-03T00:00:00Z",
    }
    qa_pairs_dicts = [
        {
            "question_en": "Q",
            "question_ru": "Q",
            "answer_en": "A",
            "answer_ru": "A",
            "card_index": 1,
        }
    ]

    fallback_redis = AsyncMock()
    fallback_redis.close = AsyncMock()

    with (
        patch(
            "obsidian_anki_sync.worker.create_pool",
            new_callable=AsyncMock,
            create=True,
        ) as create_pool_mock,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value="content"),
    ):
        create_pool_mock.return_value = fallback_redis

        result = await process_note_job(
            ctx,
            "/tmp/test.md",
            "test.md",
            metadata_dict=metadata_dict,
            qa_pairs_dicts=qa_pairs_dicts,
            result_queue_name="test-queue",
        )

    assert result["success"] is True
    create_pool_mock.assert_awaited_once()
    fallback_redis.rpush.assert_awaited_once()
    fallback_redis.expire.assert_awaited_once_with("test-queue", 3600)
    fallback_redis.close.assert_awaited_once_with(close_connection_pool=True)
