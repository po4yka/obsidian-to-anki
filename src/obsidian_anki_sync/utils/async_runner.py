"""Utilities for reusing a background asyncio event loop.

This helper allows synchronous codepaths to execute coroutines without
incurring the overhead of spinning up a new event loop per call. A single
loop is created on-demand in a dedicated thread and shared across callers.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable


class AsyncioRunner:
    """Run coroutines on a shared background event loop."""

    _instance: AsyncioRunner | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="asyncio-runner", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    @classmethod
    def get_global(cls) -> AsyncioRunner:
        """Get or create the process-wide runner instance."""

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def run(self, coro: Awaitable[Any]) -> Any:
        """Execute a coroutine on the background loop and wait for result."""

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        """Shut down the background event loop."""

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
