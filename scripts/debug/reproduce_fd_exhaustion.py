import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import ANY, MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from obsidian_anki_sync.sync.note_scanner import NoteScanner
from obsidian_anki_sync.utils.fs_monitor import get_open_file_count
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver


def setup_environment():
    # Set a low FD limit for testing
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Original FD limits: soft={soft}, hard={hard}")

    # We need enough for python imports + some buffer, but low enough to hit easily
    # Python startup uses some FDs.
    new_soft = 200
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    print(f"New FD limits: soft={new_soft}, hard={hard}")
    return new_soft


def consume_fds(count):
    fds = []
    try:
        for i in range(count):
            f = tempfile.TemporaryFile()
            fds.append(f)
    except OSError as e:
        print(f"Stopped consuming FDs at {len(fds)} due to error: {e}")
    return fds


def test_deferred_archival_exhaustion():
    # Setup directories
    temp_dir = Path(tempfile.mkdtemp())
    vault_dir = temp_dir / "vault"
    vault_dir.mkdir()
    archive_dir = temp_dir / "archive"
    archive_dir.mkdir()

    # Create dummy notes
    note_count = 50
    for i in range(note_count):
        (vault_dir / f"note_{i}.md").write_text(f"Content {i}")

    # Mock dependencies
    config = MagicMock()
    config.vault_path = vault_dir
    config.source_subdirs = None
    config.source_dir = None
    config.max_concurrent_generations = 1
    config.archiver_batch_size = 64  # Default
    config.archiver_min_fd_headroom = 32  # Default
    config.archiver_fd_poll_interval = 0.1

    state_db = MagicMock()
    card_generator = MagicMock()

    # Real archiver
    archiver = ProblematicNotesArchiver(archive_dir=archive_dir)

    scanner = NoteScanner(
        config=config,
        state_db=state_db,
        card_generator=card_generator,
        archiver=archiver,
    )

    # Manually populate deferred archives to simulate a parallel run result
    scanner._defer_archival = True
    for i in range(note_count):
        scanner._archive_note_safely(
            file_path=vault_dir / f"note_{i}.md",
            relative_path=f"note_{i}.md",
            error=Exception("Test error"),
            processing_stage="test",
        )

    print(f"Deferred {len(scanner._deferred_archives)} notes.")

    # Consume FDs to leave very little headroom
    # We want (soft_limit - open_fds) < min_headroom (32)
    # But enough to run python.
    current_open = get_open_file_count()
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Target: leave 10 FDs available
    target_open = soft - 10
    to_open = target_open - current_open

    print(
        f"Current open: {current_open}, Target open: {target_open}, To open: {to_open}"
    )

    held_fds = consume_fds(max(0, to_open))
    print(f"Held {len(held_fds)} FDs. Current open: {get_open_file_count()}")

    # Now process deferred archives
    # This should fail if it doesn't check headroom BEFORE the first batch
    print("Processing deferred archives...")
    try:
        scanner._process_deferred_archives()
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Processing failed with: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        for f in held_fds:
            f.close()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    setup_environment()
    test_deferred_archival_exhaustion()
