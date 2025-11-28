#!/usr/bin/env python3
"""Analyze Russian content in System Design files."""

import re
from pathlib import Path

# Files to analyze
FILES = [
    "q-caching-strategies--system-design--medium.md",
    "q-cap-theorem-distributed-systems--system-design--hard.md",
    "q-database-sharding-partitioning--system-design--hard.md",
    "q-design-url-shortener--system-design--medium.md",
    "q-horizontal-vertical-scaling--system-design--medium.md",
    "q-load-balancing-strategies--system-design--medium.md",
    "q-message-queues-event-driven--system-design--medium.md",
    "q-rest-api-design-best-practices--system-design--medium.md",
    "q-sql-nosql-databases--system-design--medium.md",
]


def main() -> None:
    """Analyze Russian content length in specified files."""
    base_path = Path.cwd() / "30-System-Design"

    if not base_path.exists():
        return

    results: list[tuple[str, int]] = []

    for file in FILES:
        filepath = base_path / file
        if not filepath.exists():
            continue

        content = filepath.read_text(encoding="utf-8")

        # Find Russian section (## Ответ (RU))
        match = re.search(
            r"## Ответ \(RU\)(.*?)(?=## Follow-ups|$)", content, re.DOTALL
        )
        if match:
            russian_content = match.group(1).strip()
            char_count = len(russian_content)
            results.append((file, char_count))
        else:
            pass

    # Sort by character count (ascending)
    results.sort(key=lambda x: x[1])

    for i, (file, chars) in enumerate(results[:5], 1):
        pass


if __name__ == "__main__":
    main()
