"""Tests for queue integration."""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arq.connections import ArqRedis

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import CircuitBreakerOpenError
from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.sync.queue_processor import QueueNoteProcessor
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
        "job_id": mock_job.job_id,
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

    with patch(
        "obsidian_anki_sync.sync.queue_processor.create_pool",
        side_effect=mock_create_pool,
    ):
            from obsidian_anki_sync.sync.queue_processor import QueueNoteProcessor

            queue_processor = QueueNoteProcessor(config=mock_config)

            # Test data
            note_files = [(Path("/tmp/test.md"), "test.md")]
            obsidian_cards = {}
            existing_slugs = set()
            error_by_type = defaultdict(int)
            error_samples = defaultdict(list)

            # Run scan
            result = queue_processor.scan_notes_with_queue(
                note_files,
                obsidian_cards,
                existing_slugs,
                error_by_type,
                error_samples,
            )

    # Verify
    assert len(result) == 1
    assert "test-card" in result
    assert mock_pool.enqueue_job.called
    assert mock_pool.blpop.called
    mock_pool.expire.assert_any_call(
        mock_pool.expire.call_args_list[0].args[0],
        mock_config.result_queue_ttl_seconds,
    )

    # Verify enqueue was called with result_queue_name
    call_kwargs = mock_pool.enqueue_job.call_args[1]
    assert "result_queue_name" in call_kwargs
    assert "obsidian_anki_sync:results:" in call_kwargs["result_queue_name"]


@pytest.mark.asyncio
async def test_process_note_job():
    """Test worker job processing pushes to Redis."""

    # Mock Redis in context
    mock_redis = AsyncMock()
    ctx = {"config": Config(), "orchestrator": AsyncMock(), "redis": mock_redis}

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
        job_id = "job-with-result"
        result = await process_note_job(
            ctx,
            "/tmp/test.md",
            "test.md",
            job_id=job_id,
            metadata_dict=metadata_dict,
            qa_pairs_dicts=qa_pairs_dicts,
            result_queue_name=result_queue,
        )

        assert result["success"] is True
        assert result["job_id"] == job_id
        assert len(result["cards"]) == 1
        assert result["cards"][0]["slug"] == "test-card"

        # Verify push to Redis
        assert mock_redis.rpush.called
        assert mock_redis.rpush.call_args[0][0] == result_queue
        import json

        payload = json.loads(mock_redis.rpush.call_args[0][1])
        assert payload["job_id"] == job_id
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
            job_id="fallback-job",
            metadata_dict=metadata_dict,
            qa_pairs_dicts=qa_pairs_dicts,
            result_queue_name="test-queue",
        )

    assert result["success"] is True
    assert result["job_id"] == "fallback-job"
    create_pool_mock.assert_awaited_once()
    fallback_redis.rpush.assert_awaited_once()
    fallback_redis.expire.assert_awaited_once_with("test-queue", 3600)
    fallback_redis.close.assert_awaited_once_with(close_connection_pool=True)


@pytest.mark.asyncio
async def test_push_result_returns_false_when_pool_creation_fails():
    """_push_result should return False when fallback pool cannot be created."""

    ctx: dict[str, Any] = {"config": Config(), "orchestrator": AsyncMock()}
    with patch(
        "obsidian_anki_sync.worker.create_pool",
        new_callable=AsyncMock,
        side_effect=Exception("boom"),
    ):
        from obsidian_anki_sync.worker import _push_result

        ok = await _push_result(
            ctx,
            "queue",
            {"success": True, "cards": []},
            job_id="job",
        )

    assert ok is False


@pytest.mark.asyncio
async def test_push_result_routes_to_dead_letter_on_failure():
    """_push_result should push to dead-letter queue when main push fails."""

    redis = AsyncMock()
    redis.rpush = AsyncMock(side_effect=[Exception("push failed"), None])
    redis.expire = AsyncMock()
    redis.ltrim = AsyncMock()

    ctx: dict[str, Any] = {"config": Config(), "orchestrator": AsyncMock(), "redis": redis}

    from obsidian_anki_sync.worker import _push_result

    ok = await _push_result(
        ctx,
        "queue",
        {"success": True, "cards": []},
        job_id="job",
        ttl_seconds=10,
        dead_letter_ttl_seconds=20,
        dead_letter_max_length=5,
    )

    assert ok is False
    assert redis.rpush.await_args_list[0].args[0] == "queue"
    assert redis.rpush.await_args_list[1].args[0] == "queue:dlq"
    redis.expire.assert_awaited_with("queue:dlq", 20)
    redis.ltrim.assert_awaited_with("queue:dlq", -5, -1)


@pytest.mark.asyncio
async def test_queue_processor_circuit_breaks_on_repeated_redis_errors():
    """Queue processor should raise CircuitBreakerOpenError on repeated Redis failures."""

    # Minimal config stub
    config = MagicMock()
    config.redis_url = "redis://localhost:6379"
    config.redis_socket_connect_timeout = 0.1
    config.queue_circuit_breaker_threshold = 2
    config.queue_max_retries = 0
    config.queue_max_wait_time_seconds = 5
    config.strict_mode = False

    pool = AsyncMock(spec=ArqRedis)
    pool.ping = AsyncMock(return_value=True)
    pool.expire = AsyncMock(return_value=True)
    pool.delete = AsyncMock(return_value=True)
    pool.close = AsyncMock(return_value=None)

    mock_job = MagicMock()
    mock_job.job_id = "job-1"
    pool.enqueue_job = AsyncMock(return_value=mock_job)
    pool.blpop = AsyncMock(side_effect=[Exception("conn drop"), Exception("conn drop")])

    async def mock_create_pool(*args, **kwargs):
        return pool

    processor = QueueNoteProcessor(config=config)
    with patch(
        "obsidian_anki_sync.sync.queue_processor.create_pool",
        side_effect=mock_create_pool,
    ), pytest.raises(CircuitBreakerOpenError):
        await processor._scan_notes_with_queue_async(
            note_files=[(Path("/tmp/a.md"), "a.md")],
            obsidian_cards={},
            existing_slugs=set(),
            error_by_type=defaultdict(int),
            error_samples=defaultdict(list),
        )
