"""Infrastructure service for managing persistent caches."""

import threading
from pathlib import Path
from typing import Any

try:
    import diskcache
except ImportError:
    diskcache = None

from ...utils.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages persistent disk caches for the sync engine.

    This service handles cache initialization, statistics tracking,
    and cache lifecycle management following the Single Responsibility Principle.
    """

    def __init__(self, db_path: Path):
        """Initialize cache manager.

        Args:
            db_path: Path to the database file (caches are stored alongside it)
        """
        self.db_path = db_path
        self.cache_dir = db_path.parent / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache instances
        self._agent_card_cache = None
        self._apf_card_cache = None

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_stats: dict[str, Any] = {
            "hits": 0, "misses": 0, "generation_times": []}

        # Thread safety
        self._lock = threading.Lock()

        self._initialize_caches()
        logger.info("cache_manager_initialized", cache_dir=str(self.cache_dir))

    def _initialize_caches(self) -> None:
        """Initialize disk cache instances."""
        if diskcache is None:
            logger.warning("diskcache_not_available")
            return

        # Agent card cache
        agent_cache_dir = self.cache_dir / "agent_cards"
        self._agent_card_cache = diskcache.Cache(
            directory=str(agent_cache_dir),
            size_limit=1 * 1024**3,  # 1GB limit
            eviction_policy="least-recently-used",
        )

        # APF card cache
        apf_cache_dir = self.cache_dir / "apf_cards"
        self._apf_card_cache = diskcache.Cache(
            directory=str(apf_cache_dir),
            size_limit=1 * 1024**3,  # 1GB limit
            eviction_policy="least-recently-used",
        )

    def get_agent_card_cache(self) -> "diskcache.Cache | None":
        """Get the agent card cache instance."""
        return self._agent_card_cache

    def get_apf_card_cache(self) -> "diskcache.Cache | None":
        """Get the APF card cache instance."""
        return self._apf_card_cache

    def get_cache_stats(self) -> dict[str, Any]:
        """Get current cache statistics."""
        with self._lock:
            stats: dict[str, Any] = dict(self._cache_stats.copy())
            stats.update(
                {
                    "total_requests": stats["hits"] + stats["misses"],
                    "hit_ratio": (
                        stats["hits"] / (stats["hits"] + stats["misses"])
                        if stats["hits"] + stats["misses"] > 0
                        else 0.0
                    ),
                    "avg_generation_time": (
                        sum(stats["generation_times"]) /
                        len(stats["generation_times"])
                        if stats["generation_times"]
                        else 0.0
                    ),
                }
            )
        return stats

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1
            self._cache_stats["hits"] += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1
            self._cache_stats["misses"] += 1

    def record_generation_time(self, time_seconds: float) -> None:
        """Record the time taken to generate a cached item.

        Args:
            time_seconds: Time in seconds
        """
        with self._lock:
            self._cache_stats["generation_times"].append(time_seconds)

            # Keep only last 1000 generation times to prevent unbounded growth
            if len(self._cache_stats["generation_times"]) > 1000:
                self._cache_stats["generation_times"] = self._cache_stats[
                    "generation_times"
                ][-1000:]

    def clear_cache_stats(self) -> None:
        """Clear cache statistics (for testing or reset)."""
        with self._lock:
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_stats = {"hits": 0,
                                 "misses": 0, "generation_times": []}

    def close_caches(self) -> None:
        """Close all cache instances."""
        if self._agent_card_cache:
            try:
                self._agent_card_cache.close()
                logger.debug("agent_card_cache_closed")
            except Exception as e:
                logger.warning("error_closing_agent_cache", error=str(e))

        if self._apf_card_cache:
            try:
                self._apf_card_cache.close()
                logger.debug("apf_card_cache_closed")
            except Exception as e:
                logger.warning("error_closing_apf_cache", error=str(e))

    def get_cache_size_info(self) -> dict[str, Any]:
        """Get information about cache sizes and disk usage."""
        info: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "cache_dir_exists": self.cache_dir.exists(),
            "agent_cache_size": 0,
            "apf_cache_size": 0,
            "total_size_mb": 0,
        }

        if not self.cache_dir.exists():
            return info

        # Calculate directory sizes
        try:
            agent_cache_dir = self.cache_dir / "agent_cards"
            apf_cache_dir = self.cache_dir / "apf_cards"

            if agent_cache_dir.exists():
                info["agent_cache_size"] = self._get_directory_size(
                    agent_cache_dir)

            if apf_cache_dir.exists():
                info["apf_cache_size"] = self._get_directory_size(
                    apf_cache_dir)

            info["total_size_mb"] = (
                info["agent_cache_size"] + info["apf_cache_size"]
            ) / (1024 * 1024)

        except Exception as e:
            logger.warning("error_calculating_cache_sizes", error=str(e))

        return info

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory in bytes.

        Args:
            directory: Directory to measure

        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size

    def cleanup_expired_cache_entries(self, max_age_seconds: int = 86400 * 30) -> int:
        """Clean up expired cache entries.

        Args:
            max_age_seconds: Maximum age in seconds (default: 30 days)

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0

        if self._agent_card_cache:
            try:
                # DiskCache handles expiration automatically, but we can clear if needed
                # For now, just log that cleanup would happen
                logger.debug(
                    "agent_cache_cleanup_requested", max_age_seconds=max_age_seconds
                )
            except Exception as e:
                logger.warning("error_cleaning_agent_cache", error=str(e))

        if self._apf_card_cache:
            try:
                logger.debug(
                    "apf_cache_cleanup_requested", max_age_seconds=max_age_seconds
                )
            except Exception as e:
                logger.warning("error_cleaning_apf_cache", error=str(e))

        return cleaned

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensure caches are closed."""
        self.close_caches()
