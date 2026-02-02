"""YAML frontmatter parsing for Obsidian notes."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import frontmatter as frontmatter_lib
from ruamel.yaml import YAML

from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)

# Configure ruamel.yaml for preserving comments and order
ruamel_yaml = YAML()
ruamel_yaml.preserve_quotes = True
ruamel_yaml.default_flow_style = False
ruamel_yaml.width = 4096  # Prevent line wrapping


def _preprocess_yaml_frontmatter(content: str) -> str:
    """
    Preprocess YAML frontmatter to fix common syntax errors.

    Fixes issues like:
    - Backticks in YAML arrays/strings (replaces with quotes or removes)
    - Orphaned list items after inline arrays (converts to block format)
    - Other common YAML syntax problems

    Args:
        content: Full note content with YAML frontmatter

    Returns:
        Preprocessed content with fixed YAML syntax
    """
    # Extract frontmatter section
    frontmatter_match = re.match(r"^(---\s*\n)(.*?)(\n---\s*\n)", content, re.DOTALL)
    if not frontmatter_match:
        return content

    frontmatter_start = frontmatter_match.group(1)
    frontmatter_body = frontmatter_match.group(2)
    frontmatter_end = frontmatter_match.group(3)
    rest_content = content[frontmatter_match.end() :]

    # Fix backticks in YAML arrays/strings
    def fix_backticks(text: str) -> str:
        # Replace backticks in array values: `word` -> word
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text

    def fix_orphaned_list_items(text: str) -> str:
        """Fix orphaned list items that follow inline arrays."""
        lines = text.split("\n")
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line is a field with an inline array
            inline_array_match = re.match(r"^(\w[\w\-]*)\s*:\s*\[(.*)\]\s*$", line)

            if inline_array_match:
                field_name = inline_array_match.group(1)
                array_content = inline_array_match.group(2).strip()

                # Look ahead for orphaned list items
                orphaned_items = []
                j = i + 1
                while j < len(lines) and re.match(r"^\s*-\s+", lines[j]):
                    item_match = re.match(r"^\s*-\s+(.+)$", lines[j])
                    if item_match:
                        orphaned_items.append(item_match.group(1))
                    j += 1

                if orphaned_items:
                    logger.debug(
                        "fixing_orphaned_list_items",
                        field=field_name,
                        orphaned_count=len(orphaned_items),
                    )

                    fixed_lines.append(f"{field_name}:")

                    if array_content:
                        inline_items = _parse_inline_array(array_content)
                        for item in inline_items:
                            fixed_lines.append(f"- {item}")

                    for item in orphaned_items:
                        fixed_lines.append(f"- {item}")

                    i = j
                    continue

            fixed_lines.append(line)
            i += 1

        return "\n".join(fixed_lines)

    fixed_frontmatter = fix_backticks(frontmatter_body)
    fixed_frontmatter = fix_orphaned_list_items(fixed_frontmatter)

    return frontmatter_start + fixed_frontmatter + frontmatter_end + rest_content


def _parse_inline_array(array_content: str) -> list[str]:
    """Parse inline YAML array content into list of items.

    Handles both quoted and unquoted items.

    Args:
        array_content: Content inside [...] brackets

    Returns:
        List of parsed items
    """
    items = []
    current = ""
    in_quotes = False
    quote_char = None

    for char in array_content:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == "," and not in_quotes:
            item = current.strip().strip("\"'")
            if item:
                items.append(item)
            current = ""
        else:
            current += char

    # Don't forget the last item
    item = current.strip().strip("\"'")
    if item:
        items.append(item)

    return items


def _detect_content_corruption(content: str, file_path: Path) -> None:
    """
    Detect actual content corruption patterns (binary garbage, encoding issues).

    This checks for genuine corruption indicators like control characters and
    encoding errors, while allowing legitimate markdown/code syntax.
    """
    # Pattern 1: Control characters in content (excluding legitimate whitespace)
    control_chars = []
    for char in content:
        if ord(char) < 32 and char not in "\n\r\t":
            control_chars.append(f"0x{ord(char):02x}")

    if control_chars:
        unique_controls = list(set(control_chars))
        logger.warning(
            "content_corruption_detected",
            file=str(file_path),
            pattern="control_characters",
            control_chars=unique_controls,
            message="File contains control characters that may indicate corruption",
        )

    # Pattern 2: Excessive Unicode replacement characters
    replacement_char_count = content.count("\ufffd")
    if replacement_char_count > 5:
        logger.warning(
            "content_corruption_detected",
            file=str(file_path),
            pattern="unicode_replacement_chars",
            count=replacement_char_count,
            message="File contains many Unicode replacement characters (\\ufffd) indicating encoding issues",
        )


def parse_frontmatter(content: str, file_path: Path) -> NoteMetadata:
    """
    Extract and parse YAML frontmatter from note content.

    Uses python-frontmatter with ruamel.yaml to preserve comments and order.
    Applies preprocessing to fix common YAML syntax errors before parsing.

    Args:
        content: Full note content
        file_path: Path to the file (for context)

    Returns:
        Parsed metadata

    Raises:
        ParserError: If frontmatter is missing or invalid
    """
    # Check for content corruption patterns
    _detect_content_corruption(content, file_path)

    # Preprocess content to fix common YAML syntax errors
    try:
        preprocessed_content = _preprocess_yaml_frontmatter(content)
    except Exception as e:
        logger.warning(
            "yaml_preprocessing_failed",
            file=str(file_path),
            error=str(e),
        )
        preprocessed_content = content

    # Parse frontmatter using python-frontmatter
    try:
        post = frontmatter_lib.loads(preprocessed_content)
    except Exception as e:
        error_msg = str(e)
        if "backtick" in error_msg.lower() or "`" in error_msg:
            msg = (
                f"Invalid YAML in {file_path}: {e}. "
                "Note: Backticks (`) are not valid YAML syntax. Use quotes or remove them."
            )
            raise ParserError(msg) from e
        msg = f"Invalid YAML in {file_path}: {e}"
        raise ParserError(msg) from e

    # Extract metadata dictionary
    data = post.metadata

    if not data:
        msg = f"No frontmatter found in {file_path}"
        raise ParserError(msg)

    # Validate required fields
    required_fields = ["id", "title", "topic", "language_tags", "created", "updated"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        msg = f"Missing required fields in {file_path}: {missing}"
        raise ParserError(msg)

    # Parse dates
    try:
        created = _parse_date(data["created"])
        updated = _parse_date(data["updated"])
    except (ValueError, TypeError) as e:
        msg = f"Invalid date format in {file_path}: {e}"
        raise ParserError(msg)

    # Build metadata object
    moc = _normalize_wikilink(data.get("moc"))
    related = _normalize_link_list(data.get("related"))
    tags = _normalize_string_list(data.get("tags"))
    aliases = _normalize_string_list(data.get("aliases"))
    subtopics = _normalize_string_list(data.get("subtopics"))
    sources = _normalize_sources(data.get("sources"))

    metadata = NoteMetadata(
        id=str(data["id"]),
        title=str(data["title"]),
        topic=str(data["topic"]),
        language_tags=_ensure_list(data["language_tags"]),
        created=created,
        updated=updated,
        aliases=aliases,
        subtopics=subtopics,
        question_kind=data.get("question_kind"),
        difficulty=data.get("difficulty"),
        original_language=data.get("original_language"),
        source=data.get("source"),
        source_note=data.get("source_note"),
        status=data.get("status"),
        moc=moc,
        related=related,
        tags=tags,
        sources=sources,
        anki_note_type=data.get("anki_note_type"),
        anki_slugs=_ensure_list(data.get("anki_slugs", [])),
    )

    # Validate original_language is in language_tags
    if (
        metadata.original_language
        and metadata.original_language not in metadata.language_tags
    ):
        logger.warning(
            "original_language_not_in_tags",
            file=str(file_path),
            original=metadata.original_language,
            tags=metadata.language_tags,
        )

    return metadata


def _parse_date(value: Any) -> datetime:
    """Parse date from various formats."""
    if isinstance(value, datetime):
        return value

    # Handle datetime.date objects from YAML parsing
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    if isinstance(value, str):
        # Try ISO format first
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        # Try common formats
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    msg = f"Cannot parse date: {value}"
    raise ValueError(msg)


def _ensure_list(value: Any) -> list[Any]:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _normalize_wikilink(value: Any) -> str | None:
    """Strip Obsidian wikilink syntax from a single value."""
    if value is None:
        return None

    if isinstance(value, list):
        for item in value:
            cleaned = _normalize_wikilink(item)
            if cleaned:
                return cleaned
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].strip()
    return text or None


def _normalize_link_list(value: Any) -> list[str]:
    """Normalize related/moc entries that may include wikilinks or bullet formatting."""
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[\n,]+", value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if item is None:
            continue

        # Handle nested lists (YAML interprets [[link]] as nested list structure)
        if isinstance(item, list):
            while isinstance(item, list) and len(item) > 0:
                item = item[0]
            if item is None:
                continue

        text = str(item).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text.startswith("[[") and text.endswith("]]"):
            text = text[2:-2].strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_sources(value: Any) -> list[dict[str, str]]:
    """Normalize sources into a list of dictionaries with string values."""
    if value is None:
        return []

    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[dict[str, str]] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            normalized.append({k: str(v) for k, v in item.items() if v is not None})
        else:
            text = str(item).strip()
            if text:
                normalized.append({"url": text})
    return normalized
