"""Wrapper for map_apf_to_anki_fields to check for required sentinels."""

from obsidian_anki_sync.anki.field_mapper import map_apf_to_anki_fields as _map_apf_to_anki_fields
from obsidian_anki_sync.exceptions import FieldMappingError

def map_apf_to_anki_fields(apf_html: str, note_type: str) -> dict[str, str]:
    """
    Wrapper for map_apf_to_anki_fields that checks for required sentinels.
    
    This prevents parsing crashes when the LLM omits the card block entirely.
    
    Args:
        apf_html: APF card HTML
        note_type: Anki note type name

    Returns:
        Dict of field name -> field value

    Raises:
        FieldMappingError: If mapping fails or sentinels are missing
    """
    if "<!-- BEGIN_CARDS -->" not in apf_html:
        raise FieldMappingError("Missing <!-- BEGIN_CARDS --> sentinel")
        
    return _map_apf_to_anki_fields(apf_html, note_type)
