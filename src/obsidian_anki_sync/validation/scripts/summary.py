#!/usr/bin/env python3
"""Final summary of Russian translation work."""

import re
from pathlib import Path

FILES = [
    "q-database-sharding-partitioning--system-design--hard.md",
    "q-message-queues-event-driven--system-design--medium.md",
    "q-rest-api-design-best-practices--system-design--medium.md",
    "q-sql-nosql-databases--system-design--medium.md",
    "q-caching-strategies--system-design--medium.md",
]

# Known BEFORE counts
BEFORE_COUNTS: dict[str, int] = {
    "q-database-sharding-partitioning--system-design--hard.md": 1452,
    "q-message-queues-event-driven--system-design--medium.md": 1523,
    "q-rest-api-design-best-practices--system-design--medium.md": 1620,
    "q-sql-nosql-databases--system-design--medium.md": 1927,
    "q-caching-strategies--system-design--medium.md": 2285,
}


def main() -> None:
    """Generate summary of Russian translation work."""
    base_path = Path.cwd() / "30-System-Design"

    if not base_path.exists():
        return

    total_before = 0
    total_after = 0
    results: list[dict] = []

    for filename in FILES:
        filepath = base_path / filename

        if not filepath.exists():
            continue

        content = filepath.read_text(encoding="utf-8")

        match = re.search(
            r"## Ответ \(RU\)(.*?)(?=## Follow-ups|$)", content, re.DOTALL
        )
        if match:
            russian_content = match.group(1).strip()
            after_chars = len(russian_content)
            before_chars = BEFORE_COUNTS.get(filename, 0)

            total_before += before_chars
            total_after += after_chars
            added = after_chars - before_chars

            results.append(
                {
                    "file": filename,
                    "before": before_chars,
                    "after": after_chars,
                    "added": added,
                }
            )

    for r in results:
        short_name = (
            r["file"]
            .replace("q-", "")
            .replace("--system-design--", " (")
            .replace("--", " ")
            + ")"
        )


if __name__ == "__main__":
    main()
