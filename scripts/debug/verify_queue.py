import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.append("/Users/po4yka/GitRep/obsidian-to-anki/src")

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.worker import process_note_job


async def verify_scan_notes_with_queue():
    print("Verifying scan_notes_with_queue...")

    config = Config()
    config.enable_queue = True
    config.redis_url = "redis://localhost:6379"

    mock_pool = AsyncMock()
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
                "model": "Basic",
            }
        ],
        "slugs": ["test-card"],
    }

    mock_pool.enqueue_job.return_value = mock_job
    mock_pool.get_job.return_value = mock_job

    with patch(
        "obsidian_anki_sync.sync.note_scanner.create_pool", return_value=mock_pool
    ):
        scanner = NoteScanner(config)

        note_files = [(Path("/tmp/test.md"), "test.md")]
        obsidian_cards = {}
        existing_slugs = set()
        error_by_type = defaultdict(int)
        error_samples = defaultdict(list)

        result = scanner.scan_notes_with_queue(
            note_files, obsidian_cards, existing_slugs, error_by_type, error_samples
        )

        assert len(result) == 1
        assert "test-card" in result
        assert mock_pool.enqueue_job.called
        assert mock_pool.get_job.called
        print("scan_notes_with_queue verified successfully!")


async def verify_process_note_job():
    print("Verifying process_note_job...")

    ctx = {"config": Config(), "orchestrator": AsyncMock()}

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.generation.cards = [MagicMock()]
    ctx["orchestrator"].process_note.return_value = mock_result

    mock_card = MagicMock()
    mock_card.slug = "test-card"
    mock_card.model_dump.return_value = {"slug": "test-card"}
    ctx["orchestrator"].convert_to_cards.return_value = [mock_card]

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_text", return_value="content"),
    ):
        result = await process_note_job(ctx, "/tmp/test.md", "test.md")

        assert result["success"] is True
        assert len(result["cards"]) == 1
        assert result["cards"][0]["slug"] == "test-card"
        print("process_note_job verified successfully!")


async def main():
    await verify_scan_notes_with_queue()
    await verify_process_note_job()


if __name__ == "__main__":
    asyncio.run(main())
