"""Import cards from YAML or CSV files into Anki."""

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import AnkiError, DeckImportError
from ..utils.logging import get_logger
from .client import AnkiClient

logger = get_logger(__name__)


def import_cards_from_yaml(
    client: AnkiClient,
    input_path: str | Path,
    deck_name: str,
    note_type: str | None = None,
    key_field: str | None = None,
) -> dict[str, int]:
    """
    Import cards from YAML file into Anki deck.

    Args:
        client: AnkiClient instance
        input_path: Path to YAML file
        deck_name: Target deck name
        note_type: Note type to use (auto-detected if not provided)
        key_field: Field to use for identifying existing notes (auto-detected if not provided)

    Returns:
        Dict with counts: {'created': int, 'updated': int, 'errors': int}

    Raises:
        DeckImportError: If import fails
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise DeckImportError(f"Input file not found: {input_path}")

        # Load YAML data
        with input_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        if not isinstance(yaml_data, list):
            raise DeckImportError("YAML file must contain a list of cards")

        return _import_cards(
            client=client,
            cards_data=yaml_data,
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
        )

    except yaml.YAMLError as e:
        raise DeckImportError(f"Failed to parse YAML: {e}") from e
    except Exception as e:
        raise DeckImportError(f"Failed to import from YAML: {e}") from e


def import_cards_from_csv(
    client: AnkiClient,
    input_path: str | Path,
    deck_name: str,
    note_type: str | None = None,
    key_field: str | None = None,
) -> dict[str, int]:
    """
    Import cards from CSV file into Anki deck.

    Args:
        client: AnkiClient instance
        input_path: Path to CSV file
        deck_name: Target deck name
        note_type: Note type to use (auto-detected if not provided)
        key_field: Field to use for identifying existing notes (auto-detected if not provided)

    Returns:
        Dict with counts: {'created': int, 'updated': int, 'errors': int}

    Raises:
        DeckImportError: If import fails
    """
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise DeckImportError(f"Input file not found: {input_path}")

        # Load CSV data
        cards_data = []
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert CSV row to dict format similar to YAML
                card_data: dict[str, Any] = dict(row)

                # Parse tags if they're space-separated
                if "tags" in card_data and isinstance(card_data["tags"], str):
                    card_data["tags"] = (
                        card_data["tags"].split() if card_data["tags"] else []
                    )

                # Parse manifest if it's a JSON string
                if "manifest" in card_data and isinstance(card_data["manifest"], str):
                    try:
                        card_data["manifest"] = json.loads(card_data["manifest"])
                    except json.JSONDecodeError:
                        card_data["manifest"] = {}

                cards_data.append(card_data)

        return _import_cards(
            client=client,
            cards_data=cards_data,
            deck_name=deck_name,
            note_type=note_type,
            key_field=key_field,
        )

    except Exception as e:
        raise DeckImportError(f"Failed to import from CSV: {e}") from e


def _import_cards(
    client: AnkiClient,
    cards_data: list[dict[str, Any]],
    deck_name: str,
    note_type: str | None = None,
    key_field: str | None = None,
) -> dict[str, int]:
    """
    Import cards from data structure into Anki.

    Args:
        client: AnkiClient instance
        cards_data: List of card dictionaries
        deck_name: Target deck name
        note_type: Note type to use (auto-detected if not provided)
        key_field: Field to use for identifying existing notes (auto-detected if not provided)

    Returns:
        Dict with counts: {'created': int, 'updated': int, 'errors': int}
    """
    if not cards_data:
        logger.warning("no_cards_to_import")
        return {"created": 0, "updated": 0, "errors": 0}

    # Auto-detect note type if not provided
    if not note_type:
        # Check first card for noteType
        note_type = cards_data[0].get("noteType")
        if not note_type:
            # Try to infer from existing notes in deck
            note_ids = client.find_notes(f'deck:"{deck_name}"')
            if note_ids:
                notes_info = client.notes_info(note_ids[:1])
                if notes_info:
                    note_type = notes_info[0].get("modelName")
            if not note_type:
                note_type = "APF::Simple"  # Default

    # Auto-detect key field if not provided
    if not key_field:
        # Priority: noteId > first field of note type > slug
        if any("noteId" in card for card in cards_data):
            key_field = "noteId"
        else:
            # Get first field from note type
            field_names = client.get_model_field_names(note_type)
            if field_names:
                key_field = field_names[0]
            else:
                key_field = "slug"

    logger.info(
        "importing_cards",
        deck_name=deck_name,
        note_type=note_type,
        key_field=key_field,
        card_count=len(cards_data),
    )

    # Find existing notes if using noteId
    existing_notes: dict[str, int] = {}
    if key_field == "noteId":
        note_ids = client.find_notes(f'deck:"{deck_name}"')
        if note_ids:
            notes_info = client.notes_info(note_ids)
            for note_info in notes_info:
                note_id_str = str(note_info.get("noteId", ""))
                if note_id_str:
                    note_id_val = note_info.get("noteId")
                    if isinstance(note_id_val, int):
                        existing_notes[note_id_str] = note_id_val

    # Process cards
    created = 0
    updated = 0
    errors = 0

    for card_data in cards_data:
        try:
            # Extract fields (exclude metadata fields)
            metadata_fields = {"noteId", "slug", "noteType", "tags", "manifest"}
            fields = {
                k: v for k, v in card_data.items() if k not in metadata_fields and v
            }

            tags = card_data.get("tags", [])
            if isinstance(tags, str):
                tags = tags.split()

            # Determine if this is an update or create
            existing_note_id: int | None = None
            is_update = False

            if key_field == "noteId":
                note_id_str = str(card_data.get("noteId", ""))
                if note_id_str and note_id_str in existing_notes:
                    note_id_val = existing_notes[note_id_str]
                    if isinstance(note_id_val, int):
                        existing_note_id = note_id_val
                        is_update = True
            elif key_field in card_data:
                # Search for existing note by key field value
                key_value = str(card_data[key_field])
                query = f'deck:"{deck_name}" {key_field}:"{key_value}"'
                existing_note_ids = client.find_notes(query)
                if existing_note_ids:
                    existing_note_id = existing_note_ids[0]
                    is_update = True

            if is_update and existing_note_id is not None:
                # Update existing note
                client.update_note_fields(existing_note_id, fields)
                if tags:
                    client.add_tags([existing_note_id], " ".join(tags))
                updated += 1
                logger.debug(
                    "note_updated", note_id=existing_note_id, slug=card_data.get("slug")
                )
            else:
                # Create new note
                # Note: GUID support would need to be added to the interface if needed
                new_note_id = client.add_note(
                    deck_name=deck_name,
                    model_name=note_type or "APF::Simple",
                    fields=fields,
                    tags=tags,
                )
                created += 1
                logger.debug(
                    "note_created", note_id=new_note_id, slug=card_data.get("slug")
                )

        except (AnkiError, KeyError, ValueError) as e:
            errors += 1
            logger.error(
                "import_error",
                error=str(e),
                slug=card_data.get("slug", "unknown"),
            )

    result = {"created": created, "updated": updated, "errors": errors}
    logger.info("import_complete", **result)

    return result
