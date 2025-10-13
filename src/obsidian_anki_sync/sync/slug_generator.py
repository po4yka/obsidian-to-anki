"""Slug generation with collision resolution."""

import hashlib
import re
import unicodedata
from pathlib import Path

from ..models import Manifest, NoteMetadata
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Constants
MAX_SLUG_LENGTH = 70  # Maximum length for slug base before language suffix
HASH_LENGTH = 6  # Length of collision resolution hash


def _normalize_segment(segment: str) -> str:
    """Normalize a single path segment into a slug-friendly form."""
    normalized = unicodedata.normalize("NFKD", segment)
    ascii_segment = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_segment = re.sub(r"[^a-z0-9-]", "-", ascii_segment.lower())
    ascii_segment = re.sub(r"-+", "-", ascii_segment).strip("-")
    return ascii_segment


def generate_slug(
    source_path: str, card_index: int, lang: str, existing_slugs: set[str]
) -> tuple[str, str, str | None]:
    """
    Generate a stable slug for a card.

    Args:
        source_path: Relative path to source note
        card_index: Card index (1-based)
        lang: Language code (en, ru)
        existing_slugs: Set of existing slugs to check for collisions

    Returns:
        Tuple of (slug, slug_base, hash6 or None)
    """
    # 1. Sanitize path (include directories so slugs stay unique per note path)
    path_parts = Path(source_path).with_suffix("").parts
    slug_parts = [_normalize_segment(part) for part in path_parts]
    slug_parts = [part for part in slug_parts if part]
    sanitized = "-".join(slug_parts) or "note"

    # 2. Form base without language suffix
    base_without_suffix = f"{sanitized}-p{card_index:02d}"
    slug_base = base_without_suffix[:MAX_SLUG_LENGTH]
    slug = f"{slug_base}-{lang}"
    hash6 = None

    # 3. Collision detection
    if slug in existing_slugs:
        logger.warning("slug_collision_detected", slug=slug, source=source_path)

        # Generate hash from stable components (path/index only to keep slug stable across edits)
        hash_input = f"{source_path}|{card_index}|{lang}"
        hash6 = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:HASH_LENGTH]

        available = MAX_SLUG_LENGTH - (HASH_LENGTH + 1)  # room for "-hash"
        prefix = base_without_suffix[: max(available, 0)].rstrip("-")
        slug_base = f"{prefix}-{hash6}" if prefix else hash6
        slug_base = slug_base[:MAX_SLUG_LENGTH]
        slug = f"{slug_base}-{lang}"

        # 4. If still collision (critical), add version counter
        version = 2
        while slug in existing_slugs:
            logger.error("excessive_slug_collision", slug=slug, version=version)
            version_suffix = f"-v{version}"
            available = MAX_SLUG_LENGTH - len(version_suffix)
            prefix = base_without_suffix[: max(available, 0)].rstrip("-")
            slug_base = (
                f"{prefix}{version_suffix}" if prefix else version_suffix.lstrip("-")
            )
            slug = f"{slug_base}-{lang}"
            version += 1
            if version > 10:
                raise ValueError(f"Excessive slug collision for {source_path}")

    logger.debug(
        "generated_slug",
        slug=slug,
        slug_base=slug_base,
        hash6=hash6,
        source=source_path,
        index=card_index,
        lang=lang,
    )

    return slug, slug_base, hash6


def create_manifest(
    slug: str,
    slug_base: str,
    lang: str,
    source_path: str,
    card_index: int,
    metadata: NoteMetadata,
    guid: str,
    hash6: str | None = None,
) -> Manifest:
    """
    Create a manifest for a card.

    Args:
        slug: Full slug with language suffix
        slug_base: Base slug without language
        lang: Language code
        source_path: Relative path to source note
        card_index: Card index (1-based)
        metadata: Note metadata
        hash6: Optional hash for collision resolution

    Returns:
        Manifest object
    """
    return Manifest(
        slug=slug,
        slug_base=slug_base,
        lang=lang,
        source_path=source_path,
        source_anchor=f"p{card_index:02d}",
        note_id=metadata.id,
        note_title=metadata.title,
        card_index=card_index,
        guid=guid,
        hash6=hash6,
    )
