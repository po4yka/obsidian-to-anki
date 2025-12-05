"""Tag generation domain service.

This module implements the tag generation logic following
Clean Architecture principles.
"""

from obsidian_anki_sync.domain.interfaces.tag_generation import ITagGenerator
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.utils.code_detection import (
    detect_code_language_from_content,
    detect_code_language_from_metadata,
)


class TagGenerator(ITagGenerator):
    """Domain service for generating deterministic tags from metadata.

    This ensures tag taxonomy compliance and consistency across all card generation.
    """

    def generate_tags(self, metadata: NoteMetadata, lang: str) -> list[str]:
        """Generate deterministic tags from metadata.

        This ensures tag taxonomy compliance and consistency.

        Args:
            metadata: Note metadata
            lang: Language code

        Returns:
            List of 3-6 snake_case tags
        """
        tags = []

        # 1. Primary language/tech (required first)
        primary_tech = detect_code_language_from_metadata(metadata)
        if primary_tech != "plaintext":
            tags.append(primary_tech)

        # 2. Platform (if available)
        platforms = {
            "android",
            "ios",
            "kmp",
            "jvm",
            "nodejs",
            "browser",
            "linux",
            "macos",
            "windows",
        }
        for tag in metadata.tags:
            tag_lower = tag.lower().replace("-", "_")
            if tag_lower in platforms and tag_lower not in tags:
                tags.append(tag_lower)
                break

        # 3. Topic-based tag
        topic_tag = metadata.topic.lower().replace(" ", "_").replace("-", "_")
        if topic_tag not in tags:
            tags.append(topic_tag)

        # 4. Subtopic tags (up to 3 more)
        for subtopic in metadata.subtopics:
            if len(tags) >= 6:
                break
            tag = subtopic.lower().replace(" ", "_").replace("-", "_")
            if tag not in tags:
                tags.append(tag)

        # 5. Difficulty (if less than 6 tags and specified)
        if len(tags) < 6 and metadata.difficulty:
            difficulty_tag = f"difficulty_{metadata.difficulty.lower()}"
            if difficulty_tag not in tags:
                tags.append(difficulty_tag)

        # Ensure at least 3 tags
        while len(tags) < 3:
            if "programming" not in tags:
                tags.append("programming")
            elif "conceptual" not in tags:
                tags.append("conceptual")
            else:
                tags.append("general")

        return tags[:6]  # Max 6 tags


class CodeDetector:
    """Domain service for detecting programming languages from code."""

    def detect_code_language(self, code: str) -> str:
        """Detect programming language from code content.

        Uses syntax patterns to automatically detect the language.

        Args:
            code: Code snippet to analyze

        Returns:
            Detected language name or 'text' if unknown
        """
        if not code or not code.strip():
            return "text"

        detected = detect_code_language_from_content(code)
        return detected if detected else "text"

    def get_code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks from metadata."""
        return detect_code_language_from_metadata(metadata)
