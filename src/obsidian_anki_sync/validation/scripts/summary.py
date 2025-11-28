#!/usr/bin/env python3
"""Final summary of Russian translation work."""

import re
from pathlib import Path
from typing import Dict, List

FILES = [
    "q-database-sharding-partitioning--system-design--hard.md",
    "q-message-queues-event-driven--system-design--medium.md",
    "q-rest-api-design-best-practices--system-design--medium.md",
    "q-sql-nosql-databases--system-design--medium.md",
    "q-caching-strategies--system-design--medium.md",
]

# Known BEFORE counts
BEFORE_COUNTS: Dict[str, int] = {
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
        print(f"Error: Directory not found: {base_path}")
        return

    print("=" * 80)
    print("RUSSIAN TRANSLATION SUMMARY")
    print("=" * 80)
    print()

    total_before = 0
    total_after = 0
    results: List[Dict] = []

    for filename in FILES:
        filepath = base_path / filename

        if not filepath.exists():
            print(f"Warning: File not found: {filename}")
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

    print(f"{'File':<60} {'Before':>10} {'After':>10} {'Added':>10}")
    print("-" * 95)

    for r in results:
        short_name = (
            r["file"]
            .replace("q-", "")
            .replace("--system-design--", " (")
            .replace("--", " ")
            + ")"
        )
        print(f"{short_name:<60} {r['before']:>10} {r['after']:>10} {r['added']:>10}")

    print("-" * 95)
    print(
        f"{'TOTAL':<60} {total_before:>10} {total_after:>10} {total_after - total_before:>10}"
    )
    print()
    print("=" * 80)
    print(f"Total Russian characters BEFORE: {total_before:,}")
    print(f"Total Russian characters AFTER:  {total_after:,}")
    print(f"Total characters ADDED:          {total_after - total_before:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
