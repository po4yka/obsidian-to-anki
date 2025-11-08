"""Helper utilities for agent slug generation."""

from __future__ import annotations

import hashlib
import re

from ..models import NoteMetadata

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]+")


def _slugify(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "-")
    normalized = _SLUG_CLEAN_RE.sub("-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def generate_agent_slug_base(metadata: NoteMetadata, max_length: int = 60) -> str:
    """Create a collision-resistant slug base for agent-generated cards."""

    topic_segment = _slugify(metadata.topic)
    title_segment = _slugify(metadata.title)

    # Mix in a stable hash of the note identifier to avoid cross-note collisions
    note_id = metadata.id or ""
    id_hash = hashlib.sha1(note_id.encode("utf-8")).hexdigest()[:8]

    prefix_segments = [segment for segment in (topic_segment, title_segment) if segment]
    prefix = "-".join(prefix_segments)

    if prefix:
        slug_base = f"{prefix}-{id_hash}"
    else:
        slug_base = id_hash

    if len(slug_base) > max_length:
        # Preserve the hash suffix and trim the prefix as needed.
        hash_suffix = id_hash
        separator = "-" if prefix else ""
        available = max_length - len(hash_suffix) - (1 if prefix else 0)

        if available <= 0:
            return hash_suffix

        trimmed_prefix = prefix[:available].rstrip("-")
        slug_base = (
            f"{trimmed_prefix}{separator}{hash_suffix}"
            if trimmed_prefix
            else hash_suffix
        )

    slug_base = slug_base.strip("-")
    return slug_base or id_hash
