#!/usr/bin/env python3
"""Script to complete Russian translations for System Design files.

Processes files with shortest Russian content.
"""

import re
from pathlib import Path

# Files to process (5 shortest Russian content)
FILES_TO_PROCESS = [
    "q-database-sharding-partitioning--system-design--hard.md",
    "q-message-queues-event-driven--system-design--medium.md",
    "q-rest-api-design-best-practices--system-design--medium.md",
    "q-sql-nosql-databases--system-design--medium.md",
    "q-caching-strategies--system-design--medium.md",
]


def extract_english_content(content: str) -> str:
    """Extract English answer section."""
    match = re.search(
        r"## Answer \(EN\)(.*?)(?=## Ответ \(RU\)|$)", content, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_russian_content(content: str) -> str:
    """Extract existing Russian answer section."""
    match = re.search(r"## Ответ \(RU\)(.*?)(?=## Follow-ups|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def count_chars_before_after() -> int:
    """Count Russian characters before translation."""
    base_path = Path.cwd() / "30-System-Design"

    if not base_path.exists():
        print(f"Error: Directory not found: {base_path}")
        return 0

    print("=" * 80)
    print("BEFORE TRANSLATION - Russian Character Counts:")
    print("=" * 80)

    total_before = 0
    for filename in FILES_TO_PROCESS:
        filepath = base_path / filename
        if not filepath.exists():
            print(f"{filename}: FILE NOT FOUND")
            continue

        content = filepath.read_text(encoding="utf-8")
        russian_content = extract_russian_content(content)
        char_count = len(russian_content)
        total_before += char_count
        print(f"{filename}: {char_count} chars")

    print(f"\nTotal BEFORE: {total_before} chars")
    print("=" * 80)
    return total_before


def main() -> None:
    """Main entry point for translation utilities."""
    count_chars_before_after()
    print("\nReady to translate...")


if __name__ == "__main__":
    main()
