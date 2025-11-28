#!/usr/bin/env python3
"""Update wikilinks after file renames.

After renaming files from --topic-- to --android--, we need to update
all wikilinks that reference the old filenames.
"""

import argparse
import re
from pathlib import Path
from typing import Dict


class WikilinkUpdater:
    """Updates wikilinks after file renames."""

    def __init__(self, vault_root: Path, dry_run: bool = False):
        self.vault_root = vault_root
        self.dry_run = dry_run
        self.rename_mapping = self._build_rename_mapping()
        self.stats: Dict[str, int] = {
            "files_scanned": 0,
            "files_updated": 0,
            "links_updated": 0,
        }

    def _build_rename_mapping(self) -> Dict[str, str]:
        """Build mapping of old filename patterns to new filenames."""
        # Common renames in 40-Android
        patterns = [
            ("jetpack-compose", "android"),
            ("custom-views", "android"),
            ("accessibility", "android"),
            ("devops", "android"),
            ("permissions", "android"),
            ("distribution", "android"),
            ("security", "android"),
            ("performance", "android"),
            ("testing", "android"),
            ("gradle", "android"),
            ("multiplatform", "android"),
            ("di", "android"),
        ]

        mapping: Dict[str, str] = {}

        # Scan actual files to build accurate mapping
        android_dir = self.vault_root / "40-Android"
        if android_dir.exists():
            for md_file in android_dir.glob("q-*--android--*.md"):
                stem = md_file.stem
                # Extract base name before --android--
                match = re.match(r"(q-.+?)--android--(easy|medium|hard)", stem)
                if match:
                    base = match.group(1)
                    difficulty = match.group(2)

                    # Try different old topic names
                    for old_topic, _ in patterns:
                        old_name = f"{base}--{old_topic}--{difficulty}"
                        mapping[old_name] = stem

        return mapping

    def update_file(self, filepath: Path) -> bool:
        """Update wikilinks in a file."""
        try:
            content = filepath.read_text(encoding="utf-8")
            updated = False

            # Find all wikilinks [[...]]
            wikilink_pattern = r"\[\[([^\]]+)\]\]"

            def replace_link(match: re.Match[str]) -> str:
                nonlocal updated
                link = match.group(1)

                # Check if this link matches any renamed file
                for old_name, new_name in self.rename_mapping.items():
                    if link == old_name or link.startswith(f"{old_name}|"):
                        # Replace the link
                        if "|" in link:
                            # Has display text [[link|display]]
                            display = link.split("|")[1]
                            new_link = f"[[{new_name}|{display}]]"
                        else:
                            new_link = f"[[{new_name}]]"

                        updated = True
                        self.stats["links_updated"] += 1
                        return new_link

                # No match, return original
                return match.group(0)

            content = re.sub(wikilink_pattern, replace_link, content)

            if updated:
                if not self.dry_run:
                    filepath.write_text(content, encoding="utf-8")

                print(f"  UPDATED: {filepath.name}")
                self.stats["files_updated"] += 1

            self.stats["files_scanned"] += 1
            return updated

        except Exception as e:
            print(f"  ERROR: {filepath.name}: {e}")
            return False

    def update_directory(self, directory: Path) -> None:
        """Update all markdown files in directory."""
        files = list(directory.rglob("*.md"))
        print(f"Scanning {len(files)} files in {directory.name}...")
        print()

        for filepath in files:
            # Skip hidden directories
            if any(part.startswith(".") for part in filepath.parts):
                continue

            self.update_file(filepath)

    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("WIKILINK UPDATE SUMMARY")
        print("=" * 80)
        print(f"Files scanned:      {self.stats['files_scanned']}")
        print(f"Files updated:      {self.stats['files_updated']}")
        print(f"Links updated:      {self.stats['links_updated']}")
        print("=" * 80)


def main() -> None:
    """Main entry point for wikilink updater."""
    parser = argparse.ArgumentParser(
        description="Update wikilinks after file renames")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )

    args = parser.parse_args()

    vault_root = Path.cwd()
    updater = WikilinkUpdater(vault_root, dry_run=args.dry_run)

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print()

    # Update entire vault
    updater.update_directory(vault_root)
    updater.print_summary()


if __name__ == "__main__":
    main()
