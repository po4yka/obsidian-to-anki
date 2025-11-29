"""Tests covering NoteScanner's deferred archival batching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver


@pytest.fixture
def fake_config(tmp_path):
    """Provide a minimal config namespace for NoteScanner tests."""
    return SimpleNamespace(
        vault_path=tmp_path,
        source_dir=tmp_path,
        data_dir=tmp_path,
        max_concurrent_generations=1,
        auto_adjust_workers=False,
        archiver_batch_size=2,
        archiver_min_fd_headroom=4,
        archiver_fd_poll_interval=0.0,
    )


def _build_note_request(note_path: Path, idx: int, content: str) -> dict:
    return {
        "file_path": note_path,
        "relative_path": f"note_{idx}.md",
        "error": ParserError("boom"),
        "processing_stage": "testing",
        "note_content": content,
        "card_index": idx,
        "language": "en",
    }


def test_process_deferred_archives_batches_and_waits(
    tmp_path, monkeypatch, fake_config
) -> None:
    """Batch flushing waits for FD headroom and skips missing sources."""
    archiver = ProblematicNotesArchiver(tmp_path / "problematic", enabled=True)
    scanner = NoteScanner(
        config=fake_config,
        state_db=MagicMock(),
        card_generator=MagicMock(),
        archiver=archiver,
    )

    # Prime deferred archives with three real entries.
    scanner._deferred_archives = []
    for idx in range(3):
        note_path = tmp_path / f"note_{idx}.md"
        note_path.write_text(f"cached content {idx}", encoding="utf-8")
        scanner._deferred_archives.append(
            _build_note_request(note_path, idx, f"cached content {idx}")
        )

    # Add a missing note with no cached content so it gets skipped.
    missing_path = tmp_path / "missing.md"
    scanner._deferred_archives.append(
        {
            "file_path": missing_path,
            "relative_path": "missing.md",
            "error": ParserError("missing"),
            "processing_stage": "testing",
            "note_content": None,
            "card_index": 99,
            "language": "en",
        }
    )

    calls = {"count": 0}

    def fake_headroom(min_headroom: int):
        calls["count"] += 1
        if calls["count"] == 1:
            return False, {
                "open_fd_count": 120,
                "soft_fd_limit": 128,
                "hard_fd_limit": 256,
            }
        return True, {
            "open_fd_count": 32,
            "soft_fd_limit": 128,
            "hard_fd_limit": 256,
        }

    monkeypatch.setattr(
        "obsidian_anki_sync.sync.note_scanner.has_fd_headroom",
        fake_headroom,
    )
    monkeypatch.setattr(
        "obsidian_anki_sync.sync.note_scanner.time.sleep", lambda _seconds: None
    )

    scanner._process_deferred_archives()

    meta_files = list(archiver.archive_dir.rglob("*.meta.json"))
    assert len(meta_files) == 3  # Missing note skipped
    assert calls["count"] >= 2
    assert scanner._deferred_archives == []
