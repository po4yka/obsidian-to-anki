"""Utilities for computing stable content hashes for cards."""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, Tuple

from ..models import NoteMetadata, QAPair

# Thread-safe LRU cache for content hashes
_hash_cache: Dict[Tuple[str, ...], str] = {}
_cache_lock = threading.Lock()
_MAX_CACHE_SIZE = 2048


def compute_content_hash(qa_pair: QAPair, metadata: NoteMetadata, lang: str) -> str:
    """Compute a content hash that captures all card-relevant sections.

    Args:
        qa_pair: Parsed Q/A pair.
        metadata: Note metadata (for tags/contextual fields).
        lang: Language code ("en" or "ru").

    Returns:
        SHA256 hash string that changes whenever any card component changes.
    """
    # Create a cache key from the relevant fields
    question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
    answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

    # Use a tuple of hashable values as the cache key
    cache_key = (
        lang,
        question.strip(),
        answer.strip(),
        qa_pair.followups.strip(),
        qa_pair.references.strip(),
        qa_pair.related.strip(),
        qa_pair.context.strip(),
        metadata.title,
        metadata.topic,
        tuple(sorted(metadata.subtopics)),
        tuple(sorted(metadata.tags)),
    )

    # Check cache first
    with _cache_lock:
        if cache_key in _hash_cache:
            return _hash_cache[cache_key]

    # Compute hash
    payload = "\n".join(str(component) for component in cache_key)
    hash_result = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Store in cache (with size limit)
    with _cache_lock:
        if len(_hash_cache) >= _MAX_CACHE_SIZE:
            # Remove oldest entries (simple FIFO eviction)
            items_to_remove = len(_hash_cache) - _MAX_CACHE_SIZE + 1
            for key in list(_hash_cache.keys())[:items_to_remove]:
                del _hash_cache[key]
        _hash_cache[cache_key] = hash_result

    return hash_result


def clear_content_hash_cache() -> None:
    """Clear the content hash cache to free memory."""
    with _cache_lock:
        _hash_cache.clear()
