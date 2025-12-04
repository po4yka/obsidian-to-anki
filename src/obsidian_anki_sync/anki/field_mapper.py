"""Map APF content (HTML or Markdown) to Anki note type fields.

This module handles parsing APF card content and mapping it to Anki fields.
It supports both HTML and Markdown content, converting Markdown to HTML
at the Anki boundary.
"""

import json
import re
from typing import Any, cast

from obsidian_anki_sync.apf.markdown_converter import convert_apf_field_to_html
from obsidian_anki_sync.exceptions import FieldMappingError
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


def map_apf_to_anki_fields(apf_html: str, note_type: str) -> dict[str, str]:
    """
    Map APF HTML to Anki note type fields.

    Args:
        apf_html: APF card HTML
        note_type: Anki note type name

    Returns:
        Dict of field name -> field value

    Raises:
        FieldMappingError: If mapping fails
    """
    try:
        parsed = parse_apf_card(apf_html)
    except Exception as e:
        msg = f"Failed to parse APF card: {e}"
        raise FieldMappingError(msg)

    # Map based on note type
    if note_type == "APF::Simple":
        return _map_simple(parsed)
    elif note_type in ("APF::Missing (Cloze)", "APF::Missing"):
        return _map_missing(parsed)
    elif note_type == "APF::Draw":
        return _map_draw(parsed)
    else:
        # Try generic mapping
        logger.warning("unknown_note_type", note_type=note_type)
        return _map_simple(parsed)


def parse_apf_card(apf_html: str) -> dict:
    """
    Parse APF HTML into structured fields.

    Args:
        apf_html: APF card HTML

    Returns:
        Dict of parsed fields
    """
    logger.debug("parse_apf_card_input", apf_html_length=len(apf_html))

    # Extract card block (between BEGIN_CARDS and END_CARDS)
    match = re.search(
        r"<!-- BEGIN_CARDS -->(.*?)<!-- END_CARDS -->", apf_html, re.DOTALL
    )

    if not match:
        logger.error(
            "parse_apf_card_no_block",
            apf_html_preview=apf_html[:500] if apf_html else "empty",
        )
        msg = "No card block found"
        raise ValueError(msg)

    card_content = match.group(1).strip()
    logger.debug("parse_apf_card_block", card_content_length=len(card_content))

    # Parse header - more lenient regex to allow more slug characters
    header_match = re.match(
        r"<!-- Card \d+ \| slug: ([a-zA-Z0-9_-]+) \| CardType: (\w+) \| Tags: (.+?) -->",
        card_content,
    )

    if not header_match:
        # Log what the header actually looks like for debugging
        header_line = card_content.split("\n")[0] if card_content else ""
        logger.error(
            "parse_apf_card_invalid_header",
            header_line=header_line[:200],
            card_content_preview=card_content[:300],
        )
        msg = f"Invalid card header: {header_line[:100]}"
        raise ValueError(msg)

    slug, card_type, tags_str = header_match.groups()

    # Parse fields
    parsed = {
        "slug": slug,
        "card_type": card_type,
        "tags": tags_str.strip().split(),
        "title": _extract_field(card_content, "Title"),
        "subtitle": _extract_field(card_content, "Subtitle (optional)", ["Subtitle"]),
        "syntax": _extract_field(
            card_content, "Syntax (inline) (optional)", ["Syntax (inline)", "Syntax"]
        ),
        "sample_caption": _extract_field(
            card_content, "Sample (caption) (optional)", ["Sample (caption)"]
        ),
        "sample_code": _extract_field(
            card_content,
            "Sample (code block or image) (optional for Missing)",
            ["Sample (code block or image)", "Sample"],
        ),
        "key_point": _extract_field(
            card_content, "Key point (code block / image)", ["Key point"]
        ),
        "key_point_notes": _extract_field(card_content, "Key point notes"),
        "other_notes": _extract_field(
            card_content, "Other notes (optional)", ["Other notes"]
        ),
        "markdown": _extract_field(card_content, "Markdown (optional)", ["Markdown"]),
        "manifest": _extract_manifest(card_content),
    }

    return parsed


def _extract_field(
    content: str, field_name: str, alternative_names: list[str] | None = None
) -> str:
    """Extract field content from APF HTML.

    Args:
        content: The HTML content to search.
        field_name: The primary name of the field to extract.
        alternative_names: Optional list of alternative field names to try if primary fails.
    """

    # Helper to try a specific field name
    def try_extract(name: str) -> str | None:
        escaped_name = re.escape(name)
        # Regex allows for optional whitespace inside the comment: <!-- Name --> or <!--Name-->
        pattern = rf"<!--\s*{escaped_name}\s*-->\s*(.*?)(?=<!--|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    # Try primary name
    value = try_extract(field_name)
    if value is not None:
        if not value:
            logger.debug(f"Field extraction warning: '{field_name}' found but empty")
        return value

    # Try alternatives
    if alternative_names:
        for alt_name in alternative_names:
            value = try_extract(alt_name)
            if value is not None:
                logger.debug(
                    f"Field extraction: found '{field_name}' using alternative '{alt_name}'"
                )
                if not value:
                    logger.debug(
                        f"Field extraction warning: '{alt_name}' found but empty"
                    )
                return value

    logger.debug(
        f"Field extraction failed: '{field_name}' not found in content snippet (len={len(content)})"
    )
    return ""


def _extract_manifest(content: str) -> dict:
    """Extract and parse manifest JSON."""
    match = re.search(r"<!-- manifest: ({.*?}) -->", content)
    if not match:
        return {}

    try:
        return cast("dict[Any, Any]", json.loads(match.group(1)))
    except json.JSONDecodeError:
        return {}


def _convert_fields_to_html(fields: dict[str, str]) -> dict[str, str]:
    """Convert all field values from Markdown to HTML if needed.

    This is the boundary where Markdown content is converted to HTML
    before being sent to Anki.

    Args:
        fields: Dict of field name -> field value (may be Markdown or HTML)

    Returns:
        Dict with all values converted to HTML
    """
    converted = {}
    for field_name, value in fields.items():
        if value:
            converted[field_name] = convert_apf_field_to_html(value)
        else:
            converted[field_name] = value
    return converted


def _map_simple(parsed: dict) -> dict[str, str]:
    """Map to APF::Simple fields using APF 3.0.0 field names."""
    # Combine sample content (caption + code)
    sample = ""
    if parsed.get("sample_caption"):
        sample = parsed["sample_caption"]
    if parsed.get("sample_code"):
        if sample:
            sample += "\n\n"
        sample += parsed["sample_code"]

    fields = {
        "Primary Title": parsed.get("title", ""),
        "Secondary Subtitle": parsed.get("subtitle", ""),
        "Secondary Syntax (inline code)": parsed.get("syntax", ""),
        "Primary Sample (code block)": sample.strip(),
        "Primary Key point (code block)": parsed.get("key_point", ""),
        "Primary Key point notes": parsed.get("key_point_notes", ""),
        "Note Other notes": parsed.get("other_notes", ""),
        "Note Markdown": parsed.get("markdown", ""),
    }

    # Convert Markdown fields to HTML for Anki
    return _convert_fields_to_html(fields)


def _map_missing(parsed: dict) -> dict[str, str]:
    """Map to APF::Missing (Cloze) fields using APF 3.0.0 field names."""
    # Combine sample content (caption + code)
    sample = ""
    if parsed.get("sample_caption"):
        sample = parsed["sample_caption"]
    if parsed.get("sample_code"):
        if sample:
            sample += "\n\n"
        sample += parsed["sample_code"]

    fields = {
        "Primary Title": parsed.get("title", ""),
        "Secondary Subtitle": parsed.get("subtitle", ""),
        "Secondary Syntax (inline code)": parsed.get("syntax", ""),
        "Primary Sample (code block)": sample.strip(),
        "Primary Key point (code block)": parsed.get("key_point", ""),
        "Primary Key point notes": parsed.get("key_point_notes", ""),
        "Note Other notes": parsed.get("other_notes", ""),
        "Note Markdown": parsed.get("markdown", ""),
    }

    # Convert Markdown fields to HTML for Anki
    return _convert_fields_to_html(fields)


def _map_draw(parsed: dict) -> dict[str, str]:
    """Map to APF::Draw fields using APF 3.0.0 field names."""
    # Combine sample content (caption + code)
    sample = ""
    if parsed.get("sample_caption"):
        sample = parsed["sample_caption"]
    if parsed.get("sample_code"):
        if sample:
            sample += "\n\n"
        sample += parsed["sample_code"]

    fields = {
        "Primary Title": parsed.get("title", ""),
        "Secondary Subtitle": parsed.get("subtitle", ""),
        "Secondary Syntax (inline code)": parsed.get("syntax", ""),
        "Primary Sample (code block)": sample.strip(),
        "Primary Key point (code block)": parsed.get("key_point", ""),
        "Primary Key point notes": parsed.get("key_point_notes", ""),
        "Note Other notes": parsed.get("other_notes", ""),
        "Note Markdown": parsed.get("markdown", ""),
    }

    # Convert Markdown fields to HTML for Anki
    return _convert_fields_to_html(fields)
