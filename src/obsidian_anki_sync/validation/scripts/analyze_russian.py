#!/usr/bin/env python3
"""Analyze Russian content in System Design files."""

import re
from pathlib import Path
from typing import List

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
        print(f"Error: Directory not found: {base_path}")
        return

    results: List[tuple[str, int]] = []

    for file in FILES:
        filepath = base_path / file
        if not filepath.exists():
            print(f"{file}: FILE NOT FOUND")
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
            print(f"{file}: {char_count} chars")
        else:
            print(f"{file}: NO RUSSIAN SECTION FOUND")

    # Sort by character count (ascending)
    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 80)
    print("TOP 5 SHORTEST RUSSIAN TRANSLATIONS:")
    print("=" * 80)
    for i, (file, chars) in enumerate(results[:5], 1):
        print(f"{i}. {file}")
        print(f"   Russian chars: {chars}")
        print()


if __name__ == "__main__":
    main()
