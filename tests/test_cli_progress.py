"""Unit tests for CLI progress formatting."""

from datetime import datetime, timedelta

from obsidian_anki_sync.cli_commands.sync_progress import summarize_progress
from obsidian_anki_sync.sync.progress import SyncPhase, SyncProgress


def _snapshot(total: int, processed: int) -> SyncProgress:
    now = datetime(2025, 1, 1, 12, 5, 0)
    start = now - timedelta(minutes=5)
    return SyncProgress(
        session_id="1234-5678",
        phase=SyncPhase.SCANNING,
        started_at=start,
        updated_at=now,
        total_notes=total,
        notes_processed=processed,
    )


def test_summarize_progress_handles_zero_total() -> None:
    summary = summarize_progress(_snapshot(total=0, processed=0))

    assert summary["percent_complete"] == 0.0
    assert summary["total"] == 0
    assert summary["remaining"] == 0
    assert summary["elapsed_seconds"] == 300
    assert summary["session_id_short"] == "1234"


def test_summarize_progress_clamps_percentage() -> None:
    summary = summarize_progress(_snapshot(total=10, processed=15))

    assert summary["percent_complete"] == 100.0
    assert summary["remaining"] == 0
    assert summary["processed"] == 15
    assert summary["total"] == 10
