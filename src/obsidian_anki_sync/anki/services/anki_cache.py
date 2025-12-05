"""Caching service for Anki metadata."""

import time
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AnkiCache:
    """Cache for Anki metadata to reduce redundant API calls.

    Provides TTL-based caching for deck names, model names, field names,
    and other metadata that changes infrequently.
    """

    def __init__(self, ttl_seconds: float = 300.0):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached entries in seconds (default 5 minutes)
        """
        self._cache: dict[str, Any] = {}
        self._cache_times: dict[str, float] = {}
        self._ttl = ttl_seconds
        logger.debug("anki_cache_initialized", ttl_seconds=ttl_seconds)

    def get(self, key: str) -> Any | None:
        """Get cached value if still valid.

        Args:
            key: Cache key

        Returns:
            Cached value if valid, None otherwise
        """
        if key not in self._cache:
            return None

        cache_time = self._cache_times.get(key, 0)
        if time.time() - cache_time > self._ttl:
            # Cache expired
            self.invalidate(key)
            return None

        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        self._cache_times[key] = time.time()

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._cache_times:
            del self._cache_times[key]

    def invalidate_all(self) -> None:
        """Invalidate all cached entries."""
        self._cache.clear()
        self._cache_times.clear()
        logger.debug("anki_cache_invalidated_all")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._cache)
        expired_entries = sum(
            1 for key in self._cache
            if time.time() - self._cache_times.get(key, 0) > self._ttl
        )

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "ttl_seconds": self._ttl,
        }
