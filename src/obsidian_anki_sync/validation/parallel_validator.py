"""Parallel validation support for Q&A notes.

Provides asyncio-based parallel validation for improved performance
on large vaults with many notes.
"""

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass
class ParallelConfig:
    """Configuration for parallel validation."""

    max_workers: int | None = None  # None = CPU count
    batch_size: int = 50
    show_progress: bool = True
    timeout_per_file: float = 60.0  # seconds

    def __post_init__(self) -> None:
        """Set default max_workers based on CPU count."""
        if self.max_workers is None:
            self.max_workers = min(os.cpu_count() or 4, 8)


class ParallelValidator:
    """Parallel validation runner using asyncio and thread pool.

    Uses a ThreadPoolExecutor to run synchronous validation in parallel,
    wrapped in an async interface for easy integration with async code.
    """

    def __init__(self, config: ParallelConfig | None = None) -> None:
        """Initialize parallel validator.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "ParallelValidator":
        """Enter context manager."""
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def __aenter__(self) -> "ParallelValidator":
        """Enter async context manager."""
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and shutdown executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def validate_files_async(
        self,
        files: list[Path],
        validate_fn: Callable[[Path], dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Validate files in parallel using asyncio.

        Args:
            files: List of file paths to validate
            validate_fn: Synchronous validation function (Path -> result dict)
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of validation result dicts in same order as input files
        """
        if not self._executor:
            msg = "ParallelValidator must be used as context manager"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        results: list[dict[str, Any]] = []
        completed = 0
        total = len(files)

        # Process in batches to avoid overwhelming the system
        for batch_start in range(0, total, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total)
            batch = files[batch_start:batch_end]

            # Create tasks for batch
            tasks = []
            for filepath in batch:
                task = loop.run_in_executor(self._executor, validate_fn, filepath)
                tasks.append(task)

            # Wait for batch with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout_per_file * len(batch),
                )
            except TimeoutError:
                # Handle timeout - return error results for remaining
                batch_results = []
                for filepath in batch:
                    batch_results.append(
                        {
                            "file": str(filepath),
                            "success": False,
                            "error": "Validation timeout",
                            "issues": {},
                            "passed": [],
                            "fixes": [],
                        }
                    )

            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, BaseException):
                    # Convert exception to error result
                    filepath = batch[i]
                    results.append(
                        {
                            "file": str(filepath),
                            "success": False,
                            "error": str(result),
                            "issues": {},
                            "passed": [],
                            "fixes": [],
                        }
                    )
                elif isinstance(result, dict):
                    results.append(result)
                else:
                    # Should not happen, but handle gracefully
                    filepath = batch[i]
                    results.append(
                        {
                            "file": str(filepath),
                            "success": False,
                            "error": f"Unexpected result type: {type(result)}",
                            "issues": {},
                            "passed": [],
                            "fixes": [],
                        }
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return results

    def validate_files_sync(
        self,
        files: list[Path],
        validate_fn: Callable[[Path], dict[str, Any]],
        show_progress: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Validate files in parallel synchronously.

        This is a convenience wrapper that runs the async validation
        in the default event loop.

        Args:
            files: List of file paths to validate
            validate_fn: Synchronous validation function (Path -> result dict)
            show_progress: Whether to show progress bar (overrides config)

        Returns:
            List of validation result dicts in same order as input files
        """
        show_prog = (
            show_progress if show_progress is not None else self.config.show_progress
        )

        if show_prog and len(files) > 1:
            pbar = tqdm(total=len(files), desc="Validating (parallel)", unit="file")

            def progress_callback(completed: int, total: int) -> None:
                pbar.update(1)

            try:
                return asyncio.run(
                    self._validate_with_context(files, validate_fn, progress_callback)
                )
            finally:
                pbar.close()
        else:
            return asyncio.run(self._validate_with_context(files, validate_fn, None))

    async def _validate_with_context(
        self,
        files: list[Path],
        validate_fn: Callable[[Path], dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[dict[str, Any]]:
        """Run validation within async context manager."""
        async with self:
            return await self.validate_files_async(
                files, validate_fn, progress_callback
            )


def validate_directory_parallel(
    directory: Path,
    validate_fn: Callable[[Path], dict[str, Any]],
    pattern: str = "q-*.md",
    max_workers: int | None = None,
    batch_size: int = 50,
    show_progress: bool = True,
    filter_fn: Callable[[list[Path]], tuple[list[Path], int]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate all matching files in a directory in parallel.

    This is a high-level convenience function for parallel directory validation.

    Args:
        directory: Directory to validate
        validate_fn: Function to validate a single file (Path -> result dict)
        pattern: Glob pattern for files to validate
        max_workers: Maximum parallel workers (default: CPU count, max 8)
        batch_size: Number of files per batch
        show_progress: Whether to show progress bar
        filter_fn: Optional function to filter files (returns (files, skipped_count))

    Returns:
        Tuple of (list of result dicts, skipped file count)
    """
    # Find all matching files
    files = [
        f
        for f in sorted(directory.rglob(pattern))
        if not any(part.startswith(".") for part in f.parts)
    ]

    skipped_count = 0
    if filter_fn:
        files, skipped_count = filter_fn(files)

    if not files:
        return [], skipped_count

    config = ParallelConfig(
        max_workers=max_workers,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    validator = ParallelValidator(config)
    results = validator.validate_files_sync(files, validate_fn, show_progress)

    return results, skipped_count
