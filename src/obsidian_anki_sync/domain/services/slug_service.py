"""Domain service for slug generation and management."""

import hashlib
import re
import threading
from pathlib import Path


class SlugService:
    """Domain service handling slug generation and collision resolution.

    This service encapsulates all business logic related to slug generation,
    including collision handling, normalization, and validation.
    """

    _lock = threading.Lock()
    _counters: dict[str, int] = {}

    @staticmethod
    def generate_slug_base(file_path: Path, source_dir: str, card_index: int) -> str:
        """Generate base slug from file path and card index.

        Args:
            file_path: Path to the note file
            source_dir: Source directory path
            card_index: Card index within the note

        Returns:
            Base slug without language suffix
        """
        # Remove source directory prefix
        try:
            relative_path = file_path.relative_to(Path(source_dir))
        except ValueError:
            # File not under source dir, use full path
            relative_path = file_path

        # Remove file extension
        stem = relative_path.stem

        # Remove common prefixes (q-, c-, moc-)
        stem = re.sub(r"^(q|c|moc)-", "", stem, flags=re.IGNORECASE)

        # Replace path separators with hyphens
        slug_base = str(stem).replace("/", "-").replace("\\", "-")

        # Normalize: lowercase, remove special chars, collapse hyphens
        slug_base = re.sub(r"[^a-z0-9\-]", "-", slug_base.lower())
        slug_base = re.sub(r"-+", "-", slug_base).strip("-")

        # Add card index (1-based)
        if card_index > 0:
            slug_base = f"{slug_base}-p{card_index}"

        return slug_base

    @staticmethod
    def generate_full_slug(
        base_slug: str, language: str, existing_slugs: set[str | None] | None = None
    ) -> str:
        """Generate full slug with language and collision resolution.

        Args:
            base_slug: Base slug without language
            language: Language code (e.g., 'en', 'ru')
            existing_slugs: Set of existing slugs to avoid collisions

        Returns:
            Unique slug with language suffix and collision counter if needed
        """
        if existing_slugs is None:
            existing_slugs = set()

        initial_slug = f"{base_slug}-{language}"

        # Check for collisions
        if initial_slug not in existing_slugs:
            return initial_slug

        # Resolve collision with counter
        counter = 1
        while f"{initial_slug}-{counter}" in existing_slugs:
            counter += 1

        return f"{initial_slug}-{counter}"

    @staticmethod
    def generate_thread_safe_slug(base_slug: str, language: str) -> str:
        """Generate slug with thread-safe collision resolution.

        Uses an in-memory counter to handle concurrent slug generation
        within the same process.

        Args:
            base_slug: Base slug without language
            language: Language code

        Returns:
            Unique slug with collision counter if needed
        """
        initial_slug = f"{base_slug}-{language}"

        with SlugService._lock:
            if initial_slug not in SlugService._counters:
                SlugService._counters[initial_slug] = 0
                return initial_slug
            else:
                SlugService._counters[initial_slug] += 1
                counter = SlugService._counters[initial_slug]
                return f"{initial_slug}-{counter}"

    @staticmethod
    def compute_hash(content: str, length: int = 6) -> str:
        """Compute content hash for change detection.

        Args:
            content: Content to hash
            length: Length of hash to return (default: 6)

        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return hash_obj.hexdigest()[:length]

    @staticmethod
    def normalize_slug(slug: str) -> str:
        """Normalize slug for consistent formatting.

        Args:
            slug: Slug to normalize

        Returns:
            Normalized slug
        """
        # Convert to lowercase
        normalized = slug.lower()

        # Replace spaces and underscores with hyphens
        normalized = re.sub(r"[_\s]+", "-", normalized)

        # Remove invalid characters (keep alphanumeric, hyphens)
        normalized = re.sub(r"[^a-z0-9\-]", "", normalized)

        # Collapse multiple hyphens
        normalized = re.sub(r"-+", "-", normalized)

        # Remove leading/trailing hyphens
        normalized = normalized.strip("-")

        return normalized

    @staticmethod
    def validate_slug(slug: str) -> bool:
        """Validate slug format.

        Args:
            slug: Slug to validate

        Returns:
            True if slug is valid
        """
        if not slug:
            return False

        # Must contain only lowercase letters, numbers, and hyphens
        if not re.match(r"^[a-z0-9\-]+$", slug):
            return False

        # Cannot start or end with hyphen
        if slug.startswith("-") or slug.endswith("-"):
            return False

        # Cannot have consecutive hyphens
        return "--" not in slug

    @staticmethod
    def extract_language_from_slug(slug: str) -> str | None:
        """Extract language code from slug.

        Args:
            slug: Full slug with language suffix

        Returns:
            Language code if found, None otherwise
        """
        # Look for language pattern at the end (e.g., -en, -ru)
        match = re.search(r"-([a-z]{2})(?:-\d+)?$", slug)
        return match.group(1) if match else None

    @staticmethod
    def get_base_slug(slug: str) -> str:
        """Get base slug without language suffix or collision counter.

        Args:
            slug: Full slug

        Returns:
            Base slug
        """
        # Remove language suffix and collision counter
        base = re.sub(r"-([a-z]{2})(?:-\d+)?$", "", slug)
        return base

    @staticmethod
    def clear_counters() -> None:
        """Clear collision counters (for testing)."""
        with SlugService._lock:
            SlugService._counters.clear()
