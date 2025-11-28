"""Domain service for content hashing and change detection."""

import hashlib
from typing import Any


class ContentHashService:
    """Domain service handling content hashing for change detection.

    This service encapsulates all business logic related to content hashing,
    including normalization, caching, and comparison.
    """

    # Simple in-memory cache for performance
    _cache: dict[str, str] = {}
    _cache_max_size = 10000

    @staticmethod
    def compute_hash(content: str, algorithm: str = "sha256", length: int = 64) -> str:
        """Compute hash of content for change detection.

        Args:
            content: Content to hash
            algorithm: Hash algorithm ('sha256', 'md5', etc.)
            length: Length of hash to return (truncate if needed)

        Returns:
            Hexadecimal hash string
        """
        # Normalize content for consistent hashing
        normalized = ContentHashService._normalize_content(content)

        # Check cache first
        cache_key = f"{algorithm}:{hash(normalized)}"
        if cache_key in ContentHashService._cache:
            return ContentHashService._cache[cache_key]

        # Compute hash
        if algorithm == "sha256":
            hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        elif algorithm == "md5":
            hash_obj = hashlib.md5(normalized.encode("utf-8"))
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hash_value = hash_obj.hexdigest()[:length]

        # Cache result
        ContentHashService._add_to_cache(cache_key, hash_value)

        return hash_value

    @staticmethod
    def compute_structured_hash(data: dict[str, Any]) -> str:
        """Compute hash of structured data (dict).

        Args:
            data: Dictionary to hash

        Returns:
            Hash string representing the structured data
        """
        # Convert to normalized JSON-like string
        normalized = ContentHashService._normalize_dict(data)
        return ContentHashService.compute_hash(normalized)

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hash of file content
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return ContentHashService.compute_hash(content)
        except (OSError, UnicodeDecodeError):
            # Fallback to file metadata if content can't be read
            import os

            stat = os.stat(file_path)
            metadata = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            return ContentHashService.compute_hash(metadata)

    @staticmethod
    def hashes_equal(hash1: str, hash2: str) -> bool:
        """Compare two hashes for equality.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            True if hashes are equal
        """
        return hash1 == hash2

    @staticmethod
    def has_content_changed(
        current_content: str,
        previous_hash: str,
        algorithm: str = "sha256",
        length: int = 64,
    ) -> bool:
        """Check if content has changed compared to previous hash.

        Args:
            current_content: Current content
            previous_hash: Previous hash to compare against
            algorithm: Hash algorithm used
            length: Hash length

        Returns:
            True if content has changed
        """
        current_hash = ContentHashService.compute_hash(
            current_content, algorithm, length
        )
        return not ContentHashService.hashes_equal(current_hash, previous_hash)

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize content for consistent hashing.

        Args:
            content: Raw content

        Returns:
            Normalized content
        """
        if not content:
            return ""

        # Normalize line endings
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")

        # Remove leading and trailing whitespace from lines
        lines = [line.strip() for line in normalized.split("\n")]

        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)

        return "\n".join(lines)

    @staticmethod
    def _normalize_dict(data: dict[str, Any]) -> str:
        """Normalize dictionary for consistent hashing.

        Args:
            data: Dictionary to normalize

        Returns:
            Normalized string representation
        """

        def normalize_value(value: Any) -> str:
            if isinstance(value, dict):
                return ContentHashService._normalize_dict(value)
            elif isinstance(value, list):
                return ",".join(sorted(str(normalize_value(item)) for item in value))
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value)

        # Sort keys for consistent ordering
        sorted_items = sorted(data.items())
        normalized_parts = []

        for key, value in sorted_items:
            normalized_parts.append(f"{key}:{normalize_value(value)}")

        return "|".join(normalized_parts)

    @staticmethod
    def _add_to_cache(key: str, value: str) -> None:
        """Add entry to cache with size management.

        Args:
            key: Cache key
            value: Cache value
        """
        ContentHashService._cache[key] = value

        # Simple LRU-style cache size management
        if len(ContentHashService._cache) > ContentHashService._cache_max_size:
            # Remove oldest entries (simple implementation)
            items_to_remove = (
                len(ContentHashService._cache) -
                ContentHashService._cache_max_size
            )
            keys_to_remove = list(ContentHashService._cache.keys())[
                :items_to_remove]

            for k in keys_to_remove:
                del ContentHashService._cache[k]

    @staticmethod
    def clear_cache() -> None:
        """Clear the hash cache."""
        ContentHashService._cache.clear()

    @staticmethod
    def get_cache_size() -> int:
        """Get current cache size."""
        return len(ContentHashService._cache)

    @staticmethod
    def get_cache_stats() -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(ContentHashService._cache),
            "max_size": ContentHashService._cache_max_size,
            "utilization_percent": int(
                (len(ContentHashService._cache) /
                 ContentHashService._cache_max_size)
                * 100
            ),
        }
