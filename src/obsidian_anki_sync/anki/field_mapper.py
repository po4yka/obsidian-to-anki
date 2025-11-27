"""Map APF HTML to Anki note type fields."""

import json
import re
from typing import Any, cast

from ..exceptions import FieldMappingError
from ..utils.logging import get_logger

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
        raise FieldMappingError(f"Failed to parse APF card: {e}")

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
    # Extract card block (between BEGIN_CARDS and END_CARDS)
    match = re.search(
        r"<!-- BEGIN_CARDS -->(.*?)<!-- END_CARDS -->", apf_html, re.DOTALL
    )

    if not match:
        raise ValueError("No card block found")

    card_content = match.group(1).strip()

    # Parse header
    header_match = re.match(
        r"<!-- Card \d+ \| slug: ([a-z0-9-]+) \| CardType: (\w+) \| Tags: (.+?) -->",
        card_content,
    )

    if not header_match:
        raise ValueError("Invalid card header")

    slug, card_type, tags_str = header_match.groups()

    # Parse fields
    parsed = {
        "slug": slug,
        "card_type": card_type,
        "tags": tags_str.strip().split(),
        "title": _extract_field(card_content, "Title"),
        "subtitle": _extract_field(card_content, "Subtitle (optional)"),
        "syntax": _extract_field(card_content, "Syntax (inline) (optional)"),
        "sample_caption": _extract_field(card_content, "Sample (caption) (optional)"),
        "sample_code": _extract_field(
            card_content, "Sample (code block or image) (optional for Missing)"
        ),
        "key_point": _extract_field(card_content, "Key point (code block / image)"),
        "key_point_notes": _extract_field(card_content, "Key point notes"),
        "other_notes": _extract_field(card_content, "Other notes (optional)"),
        "markdown": _extract_field(card_content, "Markdown (optional)"),
        "manifest": _extract_manifest(card_content),
    }

    return parsed


def _extract_field(content: str, field_name: str) -> str:
    """Extract field content from APF HTML."""
    # Pattern: <!-- FieldName --> ... (content until next <!-- or end)
    escaped_name = re.escape(field_name)
    pattern = rf"<!-- {escaped_name} -->\s*(.*?)(?=<!--|\Z)"

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return ""

    return match.group(1).strip()


def _extract_manifest(content: str) -> dict:
    """Extract and parse manifest JSON."""
    match = re.search(r"<!-- manifest: ({.*?}) -->", content)
    if not match:
        return {}

    try:
        return cast(dict[Any, Any], json.loads(match.group(1)))
    except json.JSONDecodeError:
        return {}


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

    return {
        "★ Title": parsed.get("title", ""),
        "☆ Subtitle": parsed.get("subtitle", ""),
        "☆ Syntax (inline code)": parsed.get("syntax", ""),
        "★ Sample (code block)": sample.strip(),
        "★ Key point (code block)": parsed.get("key_point", ""),
        "★ Key point notes": parsed.get("key_point_notes", ""),
        "✎ Other notes": parsed.get("other_notes", ""),
        "✎ Markdown": parsed.get("markdown", ""),
    }


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

    return {
        "★ Title": parsed.get("title", ""),
        "☆ Subtitle": parsed.get("subtitle", ""),
        "☆ Syntax (inline code)": parsed.get("syntax", ""),
        "★ Sample (code block)": sample.strip(),
        "★ Key point (code block)": parsed.get("key_point", ""),
        "★ Key point notes": parsed.get("key_point_notes", ""),
        "✎ Other notes": parsed.get("other_notes", ""),
        "✎ Markdown": parsed.get("markdown", ""),
    }


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

    return {
        "★ Title": parsed.get("title", ""),
        "☆ Subtitle": parsed.get("subtitle", ""),
        "☆ Syntax (inline code)": parsed.get("syntax", ""),
        "★ Sample (code block)": sample.strip(),
        "★ Key point (code block)": parsed.get("key_point", ""),
        "★ Key point notes": parsed.get("key_point_notes", ""),
        "✎ Other notes": parsed.get("other_notes", ""),
        "✎ Markdown": parsed.get("markdown", ""),
    }
