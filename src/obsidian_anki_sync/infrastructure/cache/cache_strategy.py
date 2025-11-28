"""Infrastructure service for agent-specific caching strategies."""

import hashlib
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AgentCacheKey:
    """Utility for generating cache keys for agent operations."""

    @staticmethod
    def for_note_processing(
        note_id: str,
        content_hash: str,
        operation: str,
        language: str | None = None,
    ) -> str:
        """Generate cache key for note processing operations.

        Args:
            note_id: Unique note identifier
            content_hash: Hash of note content
            operation: Operation type (e.g., 'pre_validation', 'generation')
            language: Optional language code

        Returns:
            Cache key string
        """
        key_parts = [note_id, content_hash, operation]
        if language:
            key_parts.append(language)

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    @staticmethod
    def for_card_generation(
        note_id: str,
        qa_pairs_hash: str,
        model: str,
        language: str,
    ) -> str:
        """Generate cache key for card generation.

        Args:
            note_id: Note identifier
            qa_pairs_hash: Hash of Q&A pairs
            model: LLM model used
            language: Target language

        Returns:
            Cache key string
        """
        key_string = f"{note_id}|{qa_pairs_hash}|{model}|{language}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


class AgentCacheStrategy:
    """Caching strategy for agent operations.

    This service handles caching logic specific to agent operations,
    including cache key generation, hit/miss tracking, and cache
    invalidation strategies.
    """

    def __init__(self, cache_manager: Any):
        """Initialize cache strategy.

        Args:
            cache_manager: Cache manager instance with get/set methods
        """
        self.cache_manager = cache_manager
        self._hit_count = 0
        self._miss_count = 0

        logger.debug("agent_cache_strategy_initialized")

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            # Try agent-specific cache first
            agent_cache = self.cache_manager.get_agent_card_cache()
            if agent_cache:
                value = agent_cache.get(key)
                if value is not None:
                    self._hit_count += 1
                    logger.debug("agent_cache_hit", key=key[:8])
                    return value

            # Fallback to APF cache
            apf_cache = self.cache_manager.get_apf_card_cache()
            if apf_cache:
                value = apf_cache.get(key)
                if value is not None:
                    self._hit_count += 1
                    logger.debug("apf_cache_hit", key=key[:8])
                    return value

            self._miss_count += 1
            logger.debug("cache_miss", key=key[:8])
            return None

        except Exception as e:
            logger.warning("cache_get_error", key=key[:8], error=str(e))
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        try:
            # Use agent cache for agent-specific data
            agent_cache = self.cache_manager.get_agent_card_cache()
            if agent_cache:
                agent_cache.set(key, value, expire=ttl)
                logger.debug("agent_cache_set", key=key[:8])
                return

            # Fallback to APF cache
            apf_cache = self.cache_manager.get_apf_card_cache()
            if apf_cache:
                apf_cache.set(key, value, expire=ttl)
                logger.debug("apf_cache_set", key=key[:8])

        except Exception as e:
            logger.warning("cache_set_error", key=key[:8], error=str(e))

    def invalidate_note_cache(self, note_id: str) -> int:
        """Invalidate all cache entries for a specific note.

        Args:
            note_id: Note identifier

        Returns:
            Number of entries invalidated
        """
        invalidated = 0

        try:
            agent_cache = self.cache_manager.get_agent_card_cache()
            if agent_cache:
                # Clear all agent cache entries (simplified)
                agent_cache.clear()
                invalidated += 1

            logger.info(
                "note_cache_invalidated",
                note_id=note_id,
                entries_invalidated=invalidated,
            )

        except Exception as e:
            logger.warning("cache_invalidation_error", note_id=note_id, error=str(e))

        return invalidated

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        base_stats = self.cache_manager.get_cache_stats()

        return {
            **base_stats,
            "agent_hits": self._hit_count,
            "agent_misses": self._miss_count,
            "agent_hit_ratio": (
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0
                else 0.0
            ),
        }

    def should_cache_result(
        self,
        operation: str,
        result_quality: str = "normal",
        force_cache: bool = False,
    ) -> bool:
        """Determine if a result should be cached.

        Args:
            operation: Type of operation
            result_quality: Quality of result ('normal', 'low', 'high')
            force_cache: Force caching regardless of quality

        Returns:
            True if result should be cached
        """
        if force_cache:
            return True

        # Don't cache low-quality results
        if result_quality == "low":
            return False

        # Cache based on operation type
        cacheable_operations = {
            "pre_validation",
            "generation",
            "post_validation",
        }

        return operation in cacheable_operations

    def get_cache_ttl(
        self,
        operation: str,
        result_quality: str = "normal",
    ) -> int | None:
        """Get recommended TTL for cached result.

        Args:
            operation: Type of operation
            result_quality: Quality of result

        Returns:
            TTL in seconds, or None for no expiration
        """
        # Base TTL by operation
        base_ttl = {
            "pre_validation": 3600,  # 1 hour
            "generation": 7200,  # 2 hours
            "post_validation": 1800,  # 30 minutes
        }.get(operation, 3600)  # Default 1 hour

        # Adjust based on quality
        if result_quality == "high":
            return int(base_ttl * 2)  # Longer TTL for high quality
        elif result_quality == "low":
            return int(base_ttl * 0.5)  # Shorter TTL for low quality

        return base_ttl
