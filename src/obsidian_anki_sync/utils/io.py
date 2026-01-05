"""File I/O utilities for safe and atomic operations."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def atomic_write(
    path: str | Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
    **kwargs: Any,
) -> Generator[Any]:
    """
    Context manager for atomic file writing.

    Writes to a temporary file first, then renames it to the target path
    upon successful completion. This ensures that the target file is never
    in a partially written state.

    Args:
        path: Target file path
        mode: File open mode (default: "w")
        encoding: File encoding (default: "utf-8" for text modes)
        **kwargs: Additional arguments passed to open()

    Yields:
        File object opened for writing

    Example:
        with atomic_write("config.json") as f:
            json.dump(config, f)
    """
    path = Path(path)
    parent = path.parent

    # Ensure parent directory exists
    parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in the same directory to ensure atomic rename works
    # (rename across filesystems is not atomic)
    try:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=parent,
            prefix=f".tmp_{path.name}_",
            text="b" not in mode,
        )

        # Close the low-level fd immediately, we'll open it with python's open()
        os.close(temp_fd)
        temp_path_obj = Path(temp_path)

        try:
            with open(temp_path, mode, encoding=encoding, **kwargs) as f:
                yield f
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path_obj.replace(path)

        except OSError:
            # Cleanup temp file on error
            if temp_path_obj.exists():
                with suppress(OSError):
                    temp_path_obj.unlink()
            raise

    except Exception as e:
        logger.error(
            "atomic_write_failed",
            path=str(path),
            error=str(e),
        )
        raise
