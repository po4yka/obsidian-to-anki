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


class EmptyNoteError(FieldMappingError):
    """Raised when attempting to create an Anki note with empty required fields."""


def validate_anki_note_fields(
    fields: dict[str, str],
    note_type: str,
    slug: str | None = None,
) -> None:
    """Validate that Anki note fields are not empty before sending to AnkiConnect.

    This prevents the "cannot create note because it is empty" error from AnkiConnect.

    Args:
        fields: Dictionary of field name -> field value
        note_type: The Anki note type (e.g., "APF::Simple")
        slug: Optional slug for logging context

    Raises:
        EmptyNoteError: If required fields are empty or the note would be rejected
    """
    # Required fields that must have content for APF note types
    required_fields_by_type = {
        "APF::Simple": ["Primary Title", "Primary Key point (code block)"],
        "APF::Missing (Cloze)": ["Primary Title", "Primary Key point (code block)"],
        "APF::Missing": ["Primary Title", "Primary Key point (code block)"],
        "APF::Draw": ["Primary Title", "Primary Key point (code block)"],
    }

    # Get required fields for this note type (default to Simple's requirements)
    required_fields = required_fields_by_type.get(
        note_type, required_fields_by_type["APF::Simple"]
    )

    # Check each required field
    empty_fields = []
    for field_name in required_fields:
        value = fields.get(field_name, "")
        # Check if field is empty or contains only whitespace/HTML tags
        stripped = _strip_html_and_whitespace(value)
        if not stripped:
            empty_fields.append(field_name)

    if empty_fields:
        logger.error(
            "anki_note_validation_failed",
            slug=slug,
            note_type=note_type,
            empty_fields=empty_fields,
            fields_preview={
                k: (v[:50] + "..." if len(v) > 50 else v) if v else "EMPTY"
                for k, v in fields.items()
            },
        )
        msg = (
            f"Cannot create Anki note: required fields are empty: {empty_fields}. "
            f"Note type: {note_type}"
        )
        if slug:
            msg = f"[{slug}] {msg}"
        raise EmptyNoteError(msg)

    logger.debug(
        "anki_note_validation_passed",
        slug=slug,
        note_type=note_type,
        field_count=len(fields),
    )


def _strip_html_and_whitespace(value: str) -> str:
    """Strip HTML tags and whitespace from a value for validation purposes."""
    if not value:
        return ""
    # Remove HTML tags
    stripped = re.sub(r"<[^>]+>", "", value)
    # Remove whitespace
    stripped = stripped.strip()
    # Remove common empty placeholders
    if stripped in ("", "&nbsp;", " "):
        return ""
    return stripped


def validate_field_names_match_anki(
    fields: dict[str, str],
    model_name: str,
    anki_field_names: list[str],
    slug: str | None = None,
) -> None:
    """Validate that field names match what Anki expects for the note type.

    This prevents the "cannot create note because it is empty" error that occurs
    when field names don't match the Anki model's expected field names.

    Args:
        fields: Dictionary of field name -> field value to be sent to Anki
        model_name: Name of the Anki note type (model)
        anki_field_names: List of field names from Anki's modelFieldNames API
        slug: Optional slug for logging context

    Raises:
        FieldMappingError: If field names don't match what Anki expects
    """
    sent_field_names = set(fields.keys())
    expected_field_names = set(anki_field_names)

    missing = expected_field_names - sent_field_names
    extra = sent_field_names - expected_field_names

    if missing or extra:
        logger.error(
            "anki_field_name_mismatch",
            slug=slug,
            model_name=model_name,
            missing_fields=list(missing),
            extra_fields=list(extra),
            sent_fields=list(sent_field_names),
            anki_expects=anki_field_names,
        )
        msg = (
            f"Field name mismatch for model '{model_name}'. "
            f"Missing fields: {sorted(missing) if missing else 'none'}. "
            f"Extra fields: {sorted(extra) if extra else 'none'}. "
            f"Anki expects: {anki_field_names}"
        )
        if slug:
            msg = f"[{slug}] {msg}"
        raise FieldMappingError(msg)

    logger.debug(
        "anki_field_names_validated",
        slug=slug,
        model_name=model_name,
        field_count=len(fields),
    )


def map_apf_to_anki_fields(apf_html: str, note_type: str) -> dict[str, str]:
    """
    Map APF HTML to Anki note type fields.

    Args:
        apf_html: APF card HTML
        note_type: Anki note type name

    Returns:
        Dict of field name -> field value

    Raises:
        FieldMappingError: If mapping fails or APF content is invalid
    """
    try:
        parsed = parse_apf_card(apf_html)
    except ValueError as e:
        # ValueError indicates structural issues with APF content
        # (empty content, missing markers, sentinel values, invalid header)
        logger.error(
            "apf_parse_failed_structural",
            error=str(e),
            apf_html_length=len(apf_html) if apf_html else 0,
            apf_html_preview=apf_html[:200] if apf_html else "empty",
        )
        msg = f"APF content is invalid or malformed: {e}"
        raise FieldMappingError(msg) from e
    except Exception as e:
        # Unexpected parsing errors
        logger.error(
            "apf_parse_failed_unexpected",
            error=str(e),
            error_type=type(e).__name__,
            apf_html_length=len(apf_html) if apf_html else 0,
        )
        msg = f"Failed to parse APF card: {e}"
        raise FieldMappingError(msg) from e

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

    Raises:
        ValueError: If APF content is empty, contains only whitespace,
            has no valid card block, or is missing required structure.
    """
    # Validate input is not empty or whitespace-only
    if not apf_html or not apf_html.strip():
        logger.error(
            "parse_apf_card_empty_input",
            apf_html_length=len(apf_html) if apf_html else 0,
        )
        msg = "APF content is empty or contains only whitespace"
        raise ValueError(msg)

    # Check for sentinel values that indicate parsing failure upstream
    sentinel_patterns = [
        "<!-- EMPTY_APF_CONTENT -->",
        "<!-- APF_GENERATION_FAILED -->",
        "<!-- NO_CARD_GENERATED -->",
    ]
    for sentinel in sentinel_patterns:
        if sentinel in apf_html:
            logger.error(
                "parse_apf_card_sentinel_detected",
                sentinel=sentinel,
                apf_html_preview=apf_html[:200],
            )
            msg = f"APF content contains failure sentinel: {sentinel}"
            raise ValueError(msg)

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
        msg = "No card block found (missing BEGIN_CARDS/END_CARDS markers)"
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
