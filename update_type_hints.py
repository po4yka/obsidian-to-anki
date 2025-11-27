#!/usr/bin/env python3
"""Script to update old-style type hints to modern Python 3.13 syntax.

This script updates:
- `typing.List` → `list`
- `typing.Dict` → `dict`
- `typing.Optional[T]` → `T | None`
- `typing.Tuple` → `tuple`
- Removes unused typing imports

Usage:
    python update_type_hints.py [path/to/file.py | path/to/directory]
"""

import re
import sys
from pathlib import Path
from typing import Set


def update_file_type_hints(file_path: Path) -> bool:
    """Update type hints in a single file.

    Returns True if file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    original_content = content

    # Step 1: Update type annotations in the code
    # We need to handle nested types carefully. Process from most specific to least specific.
    # Be careful not to match inside complex type expressions like Callable[[X], Y]

    # Handle Optional - but only at the top level, not inside brackets
    # Use word boundaries and avoid matching inside nested brackets
    content = re.sub(
        r'\bOptional\[([^\[\]]*(?:\[[^\[\]]*\])*[^\[\]]*)\]', r'\1 | None', content)

    # Handle List, Dict, Tuple
    content = re.sub(r'\bList\[([^\]]+)\]', r'list[\1]', content)
    content = re.sub(r'\bDict\[([^\]]+)\]', r'dict[\1]', content)
    content = re.sub(r'\bTuple\[([^\]]+)\]', r'tuple[\1]', content)

    # Step 2: Update typing imports
    # Find all typing imports
    typing_import_match = re.search(
        r'^from typing import (.+)$', content, re.MULTILINE)
    if typing_import_match:
        imports_str = typing_import_match.group(1)

        # Parse individual imports
        imports = [imp.strip() for imp in imports_str.split(',')]
        imports = [imp.split(' as ')[0] for imp in imports]  # Remove aliases

        # Remove old-style types that are being replaced
        old_types = {'List', 'Dict', 'Optional', 'Tuple'}
        new_imports = [imp for imp in imports if imp not in old_types]

        if new_imports != imports:
            if new_imports:
                new_import_line = f"from typing import {', '.join(sorted(new_imports))}"
            else:
                new_import_line = ""

            # Replace the import line
            content = re.sub(
                r'^from typing import .+$',
                new_import_line,
                content,
                flags=re.MULTILINE
            )

            # Clean up empty import lines
            content = re.sub(r'^from typing import $\n?',
                             '', content, flags=re.MULTILINE)

    # Check if content changed
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False

    return False


def find_python_files_with_old_hints(root_path: Path) -> Set[Path]:
    """Find Python files that contain old-style type hints."""
    files_to_update = set()

    # Common old-style type patterns to look for
    old_patterns = [
        r'\bList\[',
        r'\bDict\[',
        r'\bOptional\[',
        r'\bTuple\[',
        r'from typing import.*List',
        r'from typing import.*Dict',
        r'from typing import.*Optional',
        r'from typing import.*Tuple',
    ]

    for py_file in root_path.rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
            if any(re.search(pattern, content) for pattern in old_patterns):
                files_to_update.add(py_file)
        except Exception:
            continue

    return files_to_update


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python update_type_hints.py [path/to/file.py | path/to/directory]")
        sys.exit(1)

    target_path = Path(sys.argv[1])

    if target_path.is_file():
        if target_path.suffix == '.py':
            updated = update_file_type_hints(target_path)
            if updated:
                print(f"Updated 1 file")
            else:
                print("No changes needed")
        else:
            print("Target must be a Python file")
            sys.exit(1)
    elif target_path.is_dir():
        files_to_update = find_python_files_with_old_hints(target_path)
        if not files_to_update:
            print("No files found with old-style type hints")
            return

        print(f"Found {len(files_to_update)} files with old-style type hints:")
        for f in sorted(files_to_update):
            print(f"  {f}")

        updated_count = 0
        for file_path in sorted(files_to_update):
            if update_file_type_hints(file_path):
                updated_count += 1

        print(f"\nUpdated {updated_count} out of {len(files_to_update)} files")
    else:
        print(f"Path does not exist: {target_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
