"""Load controlled vocabularies from TAXONOMY.md."""

import re
from pathlib import Path


class TaxonomyLoader:
    """Loads and parses TAXONOMY.md for controlled vocabularies.

    The TAXONOMY.md file contains valid topic definitions used by
    the YAML validator to check topic field values.
    """

    # Fallback topics if TAXONOMY.md is not found or empty
    DEFAULT_TOPICS = [
        "algorithms",
        "data-structures",
        "system-design",
        "android",
        "kotlin",
        "programming-languages",
        "architecture-patterns",
        "concurrency",
        "distributed-systems",
        "databases",
        "networking",
        "operating-systems",
        "security",
        "performance",
        "cloud",
        "testing",
        "devops-ci-cd",
        "tools",
        "debugging",
        "ui-ux-accessibility",
        "behavioral",
        "cs",
    ]

    def __init__(self, taxonomy_path: Path) -> None:
        """Initialize taxonomy loader.

        Args:
            taxonomy_path: Path to TAXONOMY.md file
        """
        self.taxonomy_path = taxonomy_path
        self.valid_topics: list[str] = []
        self._load_topics()

    def _load_topics(self) -> None:
        """Extract valid topics from TAXONOMY.md."""
        if not self.taxonomy_path.exists():
            # Use default topics if file doesn't exist
            self.valid_topics = self.DEFAULT_TOPICS.copy()
            return

        content = self.taxonomy_path.read_text(encoding="utf-8")

        # Find the "Valid Topics" section
        # Look for the code block with topics
        # Pattern: lines that look like topic definitions
        topic_pattern = r"^([a-z-]+)\s+#"

        for line in content.split("\n"):
            match = re.match(topic_pattern, line)
            if match:
                topic = match.group(1)
                self.valid_topics.append(topic)

        # Fallback: if no topics found, use default list
        if not self.valid_topics:
            self.valid_topics = self.DEFAULT_TOPICS.copy()

    def get_valid_topics(self) -> list[str]:
        """Get list of valid topics.

        Returns:
            List of valid topic strings
        """
        return self.valid_topics

    @staticmethod
    def find_taxonomy_file(start_path: Path) -> Path | None:
        """Find TAXONOMY.md file in vault.

        Searches common locations first, then recursively.

        Args:
            start_path: Path to start searching from (usually vault root)

        Returns:
            Path to TAXONOMY.md if found, None otherwise
        """
        # Check common locations
        candidates = [
            start_path / "00-Administration" / "TAXONOMY.md",
            start_path / "TAXONOMY.md",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Search recursively
        for taxonomy_file in start_path.rglob("TAXONOMY.md"):
            return taxonomy_file

        return None
