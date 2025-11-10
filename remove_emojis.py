#!/usr/bin/env python3
"""Remove emojis from Python files while preserving UTF-8 encoding."""

import re
import sys
from pathlib import Path


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text while preserving other Unicode."""
    # Comprehensive emoji pattern covering all emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"  # dingbats
        "\U000024c2-\U0001f251"  # enclosed characters
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U0001fa00-\U0001fa6f"  # chess symbols
        "\U0001fa70-\U0001faff"  # symbols and pictographs extended-a
        "\U00002600-\U000026ff"  # miscellaneous symbols
        "\U00002700-\U000027bf"  # dingbats
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def process_file(file_path: Path) -> bool:
    """Process a single file to remove emojis."""
    try:
        # Read with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove emojis
        cleaned = remove_emojis(content)

        # Only write if changed
        if cleaned != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Cleaned: {file_path}")
            return True
        else:
            print(f"No emojis: {file_path}")
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    files = [
        "src/obsidian_anki_sync/agents/langchain/__init__.py",
        "src/obsidian_anki_sync/agents/langchain/adapter.py",
        "src/obsidian_anki_sync/agents/langchain/models.py",
        "src/obsidian_anki_sync/agents/langchain/tools/card_differ.py",
        "src/obsidian_anki_sync/agents/langchain/tools/enhanced_card_differ.py",
        "src/obsidian_anki_sync/agents/parser_repair.py",
        "src/obsidian_anki_sync/cli.py",
        "src/obsidian_anki_sync/exceptions.py",
        "src/obsidian_anki_sync/obsidian/parser.py",
    ]

    changed_count = 0
    for file_path_str in files:
        file_path = Path(file_path_str)
        if file_path.exists():
            if process_file(file_path):
                changed_count += 1
        else:
            print(f"Not found: {file_path}", file=sys.stderr)

    print(f"\nProcessed {len(files)} files, {changed_count} changed")
