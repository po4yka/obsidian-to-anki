"""Write updates to Obsidian note frontmatter while preserving structure."""

import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from ..utils.logging import get_logger

logger = get_logger(__name__)


def update_frontmatter(file_path: Path, updates: dict[str, Any]) -> bool:
    """
    Update specific fields in a note's YAML frontmatter while preserving structure.

    Uses ruamel.yaml to maintain formatting, comments, and field order.

    Args:
        file_path: Path to the markdown file
        updates: Dict of field names to new values

    Returns:
        True if update was successful, False otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.error("failed_to_read_file", file=str(file_path), error=str(e))
        return False

    # Extract frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        logger.error("no_frontmatter_found", file=str(file_path))
        return False

    frontmatter_text = match.group(1)
    body = content[match.end():]

    # Parse with ruamel.yaml to preserve formatting
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    try:
        data = yaml.load(StringIO(frontmatter_text))
        if data is None:
            data = {}
    except Exception as e:
        logger.error("yaml_parse_failed", file=str(file_path), error=str(e))
        return False

    # Apply updates
    for key, value in updates.items():
        # Convert datetime to ISO string for YAML
        if isinstance(value, datetime):
            value = value.isoformat()
        data[key] = value

    # Serialize back to YAML
    try:
        output = StringIO()
        yaml.dump(data, output)
        new_frontmatter = output.getvalue()
    except Exception as e:
        logger.error("yaml_dump_failed", file=str(file_path), error=str(e))
        return False

    # Reconstruct file
    new_content = f"---\n{new_frontmatter}---\n{body}"

    # Write back
    try:
        file_path.write_text(new_content, encoding="utf-8")
        logger.info("frontmatter_updated", file=str(file_path), fields=list(updates.keys()))
        return True
    except OSError as e:
        logger.error("failed_to_write_file", file=str(file_path), error=str(e))
        return False


def update_anki_sync_status(
    file_path: Path,
    slugs: list[str],
    sync_time: datetime | None = None
) -> bool:
    """
    Update Anki sync status in note frontmatter.

    Sets:
    - anki_synced: true
    - anki_slugs: list of generated slugs
    - anki_last_sync: timestamp of sync

    Args:
        file_path: Path to the markdown file
        slugs: List of Anki card slugs generated from this note
        sync_time: Timestamp of sync (defaults to now)

    Returns:
        True if update was successful
    """
    if sync_time is None:
        sync_time = datetime.now()

    updates = {
        "anki_synced": True,
        "anki_slugs": slugs,
        "anki_last_sync": sync_time,
    }

    return update_frontmatter(file_path, updates)
