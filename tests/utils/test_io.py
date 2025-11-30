import os
import threading
import time
from pathlib import Path

import pytest

from obsidian_anki_sync.utils.io import atomic_write


def test_atomic_write_creates_file(tmp_path):
    """Test that atomic_write creates a file with correct content."""
    target_file = tmp_path / "test_file.txt"
    content = "Hello, World!"

    with atomic_write(target_file) as f:
        f.write(content)

    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == content


def test_atomic_write_overwrites_file(tmp_path):
    """Test that atomic_write overwrites an existing file."""
    target_file = tmp_path / "test_file.txt"
    target_file.write_text("Old content", encoding="utf-8")

    new_content = "New content"
    with atomic_write(target_file) as f:
        f.write(new_content)

    assert target_file.read_text(encoding="utf-8") == new_content


def test_atomic_write_failure_cleanup(tmp_path):
    """Test that temp file is cleaned up on failure and target is unchanged."""
    target_file = tmp_path / "test_file.txt"
    original_content = "Original content"
    target_file.write_text(original_content, encoding="utf-8")

    try:
        with atomic_write(target_file) as f:
            f.write("New content")
            raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass

    # Target file should be unchanged
    assert target_file.read_text(encoding="utf-8") == original_content

    # No temp files should be left
    temp_files = list(tmp_path.glob(".tmp_*"))
    assert len(temp_files) == 0


def test_atomic_write_atomicity(tmp_path):
    """
    Test atomicity by checking that the file doesn't exist (or has old content)
    until the context manager exits.
    """
    target_file = tmp_path / "atomic_test.txt"
    content = "Final content"

    with atomic_write(target_file) as f:
        f.write(content)
        f.flush()
        # File should not exist at target path yet
        assert not target_file.exists()

    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == content


if __name__ == "__main__":
    # Manual run if pytest is not available
    import shutil
    import tempfile

    print("Running manual tests...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        test_atomic_write_creates_file(temp_dir)
        print("test_atomic_write_creates_file passed")

        test_atomic_write_overwrites_file(temp_dir)
        print("test_atomic_write_overwrites_file passed")

        test_atomic_write_failure_cleanup(temp_dir)
        print("test_atomic_write_failure_cleanup passed")

        test_atomic_write_atomicity(temp_dir)
        print("test_atomic_write_atomicity passed")

        print("All tests passed!")
    finally:
        shutil.rmtree(temp_dir)
