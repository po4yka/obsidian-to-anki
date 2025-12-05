"""Export cards to Anki deck files (.apkg) using genanki."""

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import genanki
import yaml

from obsidian_anki_sync.exceptions import DeckExportError
from obsidian_anki_sync.models import Card
from obsidian_anki_sync.utils.io import atomic_write
from obsidian_anki_sync.utils.logging import get_logger

from .field_mapper import map_apf_to_anki_fields
from .safe_exporter import safe_export_context

logger = get_logger(__name__)


# Define APF note types as genanki Models
# Model IDs are deterministic based on name
def _generate_model_id(model_name: str) -> int:
    """Generate deterministic model ID from name."""
    return int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)


# APF::Simple Model
APF_SIMPLE_MODEL = genanki.Model(
    model_id=_generate_model_id("APF::Simple"),
    name="APF::Simple",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Additional"},
        {"name": "Manifest"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
<div class="apf-card">
    <div class="front">
        {{Front}}
    </div>
</div>
""",
            "afmt": """
<div class="apf-card">
    <div class="front">
        {{Front}}
    </div>
    <hr id="answer">
    <div class="back">
        {{Back}}
    </div>
    {{#Additional}}
    <div class="additional">
        <hr>
        <h4>Additional Notes</h4>
        {{Additional}}
    </div>
    {{/Additional}}
</div>
""",
        }
    ],
    css="""
.apf-card {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.front {
    margin-bottom: 20px;
}

.back {
    margin-top: 20px;
}

.additional {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Monaco", "Consolas", monospace;
    font-size: 14px;
}

pre {
    background-color: #f4f4f4;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}

pre code {
    background-color: transparent;
    padding: 0;
}
""",
)


# APF::Missing (Cloze) Model
APF_MISSING_MODEL = genanki.Model(
    model_id=_generate_model_id("APF::Missing"),
    name="APF::Missing",
    fields=[
        {"name": "Text"},
        {"name": "Extra"},
        {"name": "Manifest"},
    ],
    templates=[
        {
            "name": "Cloze",
            "qfmt": "{{cloze:Text}}",
            "afmt": """
{{cloze:Text}}
{{#Extra}}
<hr>
<div class="extra">
    {{Extra}}
</div>
{{/Extra}}
""",
        }
    ],
    model_type=genanki.Model.CLOZE,
    css="""
.cloze {
    font-weight: bold;
    color: blue;
}

.extra {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Monaco", "Consolas", monospace;
    font-size: 14px;
}

pre {
    background-color: #f4f4f4;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}
""",
)


# APF::Draw Model
APF_DRAW_MODEL = genanki.Model(
    model_id=_generate_model_id("APF::Draw"),
    name="APF::Draw",
    fields=[
        {"name": "Prompt"},
        {"name": "Drawing"},
        {"name": "Notes"},
        {"name": "Manifest"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
<div class="apf-draw">
    <div class="prompt">
        {{Prompt}}
    </div>
</div>
""",
            "afmt": """
<div class="apf-draw">
    <div class="prompt">
        {{Prompt}}
    </div>
    <hr id="answer">
    <div class="drawing">
        {{Drawing}}
    </div>
    {{#Notes}}
    <div class="notes">
        <hr>
        {{Notes}}
    </div>
    {{/Notes}}
</div>
""",
        }
    ],
    css="""
.apf-draw {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.prompt {
    margin-bottom: 20px;
}

.drawing {
    margin-top: 20px;
}

.drawing img {
    max-width: 100%;
    height: auto;
}

.notes {
    margin-top: 20px;
    padding-top: 20px;
}
""",
)


# Model registry
MODEL_REGISTRY = {
    "APF::Simple": APF_SIMPLE_MODEL,
    "APF::Missing": APF_MISSING_MODEL,
    "APF::Missing (Cloze)": APF_MISSING_MODEL,
    "APF::Draw": APF_DRAW_MODEL,
}


def _generate_deck_id(deck_name: str) -> int:
    """Generate deterministic deck ID from name."""
    return int(hashlib.md5(deck_name.encode()).hexdigest()[:8], 16)


def create_deck(
    cards: list[Card],
    deck_name: str,
    deck_description: str = "",
) -> genanki.Deck:
    """
    Create a genanki Deck from cards.

    Args:
        cards: List of Card objects
        deck_name: Name for the deck
        deck_description: Optional description

    Returns:
        genanki.Deck instance

    Raises:
        DeckExportError: If deck creation fails
    """
    try:
        deck_id = _generate_deck_id(deck_name)
        deck = genanki.Deck(deck_id, deck_name, deck_description)

        for card in cards:
            note = _card_to_note(card)
            deck.add_note(note)

        logger.info(
            "deck_created",
            deck_name=deck_name,
            deck_id=deck_id,
            card_count=len(cards),
        )

        return deck

    except Exception as e:
        msg = f"Failed to create deck: {e}"
        raise DeckExportError(msg) from e


def _card_to_note(card: Card) -> genanki.Note:
    """
    Convert a Card to a genanki Note.

    Args:
        card: Card object

    Returns:
        genanki.Note instance

    Raises:
        DeckExportError: If conversion fails
    """
    try:
        # Get the appropriate model
        model = MODEL_REGISTRY.get(card.note_type)
        if not model:
            logger.warning(
                "unknown_note_type",
                note_type=card.note_type,
                fallback="APF::Simple",
            )
            model = APF_SIMPLE_MODEL

        # Map APF HTML to fields
        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

        # Create note with fields in the correct order
        field_names = [f["name"] for f in model.fields]
        field_values = [fields.get(name, "") for name in field_names]

        # Use card GUID if available
        guid = card.guid if card.guid else None

        note = genanki.Note(
            model=model,
            fields=field_values,
            tags=card.tags,
            guid=guid,
        )

        return note

    except Exception as e:
        msg = f"Failed to convert card {card.slug}: {e}"
        raise DeckExportError(msg) from e


def export_deck(
    deck: genanki.Deck,
    output_path: str | Path,
    media_files: list[str] | None = None,
) -> None:
    """
    Export a deck to an .apkg file.

    Args:
        deck: genanki.Deck to export
        output_path: Path to output .apkg file
        media_files: Optional list of media file paths to include

    Raises:
        DeckExportError: If export fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        package = genanki.Package(deck)

        if media_files:
            package.media_files = media_files

        package.write_to_file(str(output_path))

        logger.info(
            "deck_exported",
            output_path=str(output_path),
            card_count=len(deck.notes),
            media_count=len(media_files) if media_files else 0,
        )

    except Exception as e:
        msg = f"Failed to export deck: {e}"
        raise DeckExportError(msg) from e


def export_cards_to_apkg(
    cards: list[Card],
    output_path: str | Path,
    deck_name: str,
    deck_description: str = "",
    media_files: list[str] | None = None,
) -> None:
    """
    Export cards directly to an .apkg file.

    Convenience function that combines deck creation and export.

    Args:
        cards: List of Card objects
        output_path: Path to output .apkg file
        deck_name: Name for the deck
        deck_description: Optional description
        media_files: Optional list of media file paths to include

    Raises:
        DeckExportError: If export fails
    """
    deck = create_deck(cards, deck_name, deck_description)
    export_deck(deck, output_path, media_files)

    logger.info(
        "cards_exported_to_apkg",
        output_path=str(output_path),
        deck_name=deck_name,
        card_count=len(cards),
    )


def export_cards_to_yaml(
    cards: list[Card],
    output_path: str | Path,
    include_note_id: bool = True,
) -> None:
    """
    Export cards to YAML format.

    Args:
        cards: List of Card objects
        output_path: Path to output YAML file
        include_note_id: Whether to include noteId field for updates

    Raises:
        DeckExportError: If export fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert cards to YAML-serializable format
        yaml_data = []
        for card in cards:
            # Map APF HTML to Anki fields
            fields = map_apf_to_anki_fields(card.apf_html, card.note_type)

            card_data: dict[str, Any] = {
                "slug": card.slug,
                "noteType": card.note_type,
                "tags": card.tags,
            }

            # Add noteId if available and requested
            if include_note_id and card.manifest.guid:
                card_data["noteId"] = card.manifest.guid

            # Add all fields
            card_data.update(fields)

            # Add manifest data
            card_data["manifest"] = {
                "slug": card.manifest.slug,
                "slug_base": card.manifest.slug_base,
                "lang": card.manifest.lang,
                "source_path": card.manifest.source_path,
                "source_anchor": card.manifest.source_anchor,
                "note_id": card.manifest.note_id,
                "note_title": card.manifest.note_title,
                "card_index": card.manifest.card_index,
                "guid": card.manifest.guid,
            }

            if card.manifest.hash6:
                card_data["manifest"]["hash6"] = card.manifest.hash6

            yaml_data.append(card_data)

        # Write YAML file
        with atomic_write(output_path) as f:
            yaml.dump(
                yaml_data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(
            "cards_exported_to_yaml",
            output_path=str(output_path),
            card_count=len(cards),
        )

    except Exception as e:
        msg = f"Failed to export cards to YAML: {e}"
        raise DeckExportError(msg) from e


def export_cards_to_csv(
    cards: list[Card],
    output_path: str | Path,
    include_note_id: bool = True,
) -> None:
    """
    Export cards to CSV format.

    Args:
        cards: List of Card objects
        output_path: Path to output CSV file
        include_note_id: Whether to include noteId field for updates

    Raises:
        DeckExportError: If export fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cards:
            # Create empty CSV with headers
            with atomic_write(output_path, newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["noteId", "slug", "noteType", "tags", "fields"])
            logger.info("empty_csv_created", output_path=str(output_path))
            return

        # Collect all unique field names across all cards
        all_field_names: set[str] = set()
        card_data_list = []

        for card in cards:
            fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
            all_field_names.update(fields.keys())

            card_data = {
                "slug": card.slug,
                "noteType": card.note_type,
                "tags": " ".join(card.tags) if card.tags else "",
                **fields,
            }

            if include_note_id and card.manifest.guid:
                card_data["noteId"] = card.manifest.guid

            card_data_list.append(card_data)

        # Sort field names for consistent column order
        # Put common fields first
        common_fields = ["noteId", "slug", "noteType", "tags"]
        field_names = [
            f
            for f in common_fields
            if f in all_field_names or any(f in d for d in card_data_list)
        ]
        field_names.extend(sorted(all_field_names - set(common_fields)))

        # Write CSV file
        with atomic_write(output_path, newline="") as f:
            dict_writer: csv.DictWriter[str] = csv.DictWriter(
                f, fieldnames=field_names, extrasaction="ignore"
            )
            dict_writer.writeheader()
            dict_writer.writerows(card_data_list)

        logger.info(
            "cards_exported_to_csv",
            output_path=str(output_path),
            card_count=len(cards),
        )

    except Exception as e:
        msg = f"Failed to export cards to CSV: {e}"
        raise DeckExportError(msg) from e


def export_deck_from_anki(
    client: Any,
    deck_name: str,
    output_path: str | Path,
    format: str = "yaml",
    include_note_id: bool = True,
) -> None:
    """
    Export a deck from Anki to YAML or CSV file.

    Args:
        client: AnkiClient instance
        deck_name: Name of the deck to export
        output_path: Path to output file
        format: Output format ('yaml' or 'csv')
        include_note_id: Whether to include noteId field

    Raises:
        DeckExportError: If export fails
    """
    try:
        # Find all notes in the deck
        note_ids = client.find_notes(f'deck:"{deck_name}"')

        if not note_ids:
            logger.warning("no_notes_found", deck_name=deck_name)
            # Create empty file
            if format.lower() == "csv":
                export_cards_to_csv([], output_path, include_note_id)
            else:
                export_cards_to_yaml([], output_path, include_note_id)
            return

        notes_info = client.notes_info(note_ids)

        cards = []
        for note_info in notes_info:
            # Extract fields
            fields = dict(note_info.get("fields", {}).items())

            # Try to extract manifest from Manifest field
            manifest_data = {}
            if "Manifest" in fields:
                try:
                    # Manifest might be in HTML comment or JSON
                    manifest_text = fields["Manifest"]
                    # Try to parse as JSON first
                    manifest_data = json.loads(manifest_text)
                except (json.JSONDecodeError, KeyError):
                    pass

            from obsidian_anki_sync.models import Manifest

            manifest = Manifest(
                slug=manifest_data.get("slug", f"note-{note_info['noteId']}"),
                slug_base=manifest_data.get("slug_base", ""),
                lang=manifest_data.get("lang", "en"),
                source_path=manifest_data.get("source_path", ""),
                source_anchor=manifest_data.get("source_anchor", ""),
                note_id=manifest_data.get("note_id", ""),
                note_title=manifest_data.get("note_title", ""),
                card_index=manifest_data.get("card_index", 0),
                guid=str(note_info.get("noteId", "")),
                hash6=manifest_data.get("hash6"),
            )

            apf_html = _reconstruct_apf_from_fields(
                fields, note_info.get("modelName", "APF::Simple")
            )

            card = Card(
                slug=manifest.slug,
                lang=manifest.lang,
                apf_html=apf_html,
                manifest=manifest,
                content_hash="",  # We don't have this from Anki
                note_type=note_info.get("modelName", "APF::Simple"),
                tags=note_info.get("tags", []),
                guid=str(note_info.get("noteId", "")),
            )

            cards.append(card)

        # Export based on format
        if format.lower() == "csv":
            export_cards_to_csv(cards, output_path, include_note_id)
        else:
            export_cards_to_yaml(cards, output_path, include_note_id)

        logger.info(
            "deck_exported_from_anki",
            deck_name=deck_name,
            output_path=str(output_path),
            format=format,
            card_count=len(cards),
        )

    except Exception as e:
        msg = f"Failed to export deck from Anki: {e}"
        raise DeckExportError(msg) from e


def _reconstruct_apf_from_fields(fields: dict[str, str], note_type: str) -> str:
    """
    Reconstruct APF HTML from Anki fields.

    This is a simplified reconstruction. Full APF format reconstruction
    would require more complex logic.

    Args:
        fields: Dictionary of field names to values
        note_type: The note type name

    Returns:
        Reconstructed APF HTML string
    """
    if note_type.startswith("APF::Missing"):
        # Cloze deletion card
        text = fields.get("Text", "")
        extra = fields.get("Extra", "")
        return f"<!-- APF::Missing -->\n{text}\n{extra}"
    elif note_type == "APF::Draw":
        prompt = fields.get("Prompt", "")
        drawing = fields.get("Drawing", "")
        notes = fields.get("Notes", "")
        return f"<!-- APF::Draw -->\n{prompt}\n{drawing}\n{notes}"
    else:
        # Simple card
        front = fields.get("Front", "")
        back = fields.get("Back", "")
        additional = fields.get("Additional", "")
        return f"<!-- APF::Simple -->\n<!-- BEGIN_CARDS -->\n<div class='card'><div class='front'>{front}</div><div class='back'>{back}</div></div>\n<!-- END_CARDS -->\n{additional}"


# Safe export functions using new implementation

def export_cards_to_yaml_safe(
    cards: list[Card],
    output_path: str | Path,
    deck_name: str,
    deck_description: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    include_note_id: bool = True,
    create_backup: bool = True,
) -> dict[str, Any]:
    """
    Safely export cards to YAML file with validation, rollback, and progress tracking.

    This is the recommended function for new code. It provides:
    - Comprehensive input and output validation
    - Transaction-like operations with rollback on failure
    - Progress tracking and detailed error reporting
    - Automatic backup creation and cleanup
    - Resource management and memory safety

    Args:
        cards: List of Card objects to export
        output_path: Path to output YAML file
        deck_name: Deck name for metadata
        deck_description: Optional deck description
        progress_callback: Optional callback for progress updates
        include_note_id: Whether to include noteId field for updates
        create_backup: Whether to create backup of existing file

    Returns:
        Dictionary with operation results and metadata

    Raises:
        DeckExportError: If export fails (with automatic rollback)
    """
    with safe_export_context() as exporter:
        return exporter.export_to_yaml(
            cards=cards,
            output_path=output_path,
            deck_name=deck_name,
            deck_description=deck_description,
            progress_callback=progress_callback,
            include_note_id=include_note_id,
            create_backup=create_backup,
        )


def export_cards_to_csv_safe(
    cards: list[Card],
    output_path: str | Path,
    deck_name: str,
    deck_description: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    include_note_id: bool = True,
    create_backup: bool = True,
) -> dict[str, Any]:
    """
    Safely export cards to CSV file with validation, rollback, and progress tracking.

    This is the recommended function for new code. It provides:
    - Comprehensive input and output validation
    - Transaction-like operations with rollback on failure
    - Progress tracking and detailed error reporting
    - Automatic backup creation and cleanup
    - Resource management and memory safety

    Args:
        cards: List of Card objects to export
        output_path: Path to output CSV file
        deck_name: Deck name for metadata
        deck_description: Optional deck description
        progress_callback: Optional callback for progress updates
        include_note_id: Whether to include noteId field for updates
        create_backup: Whether to create backup of existing file

    Returns:
        Dictionary with operation results and metadata

    Raises:
        DeckExportError: If export fails (with automatic rollback)
    """
    with safe_export_context() as exporter:
        return exporter.export_to_csv(
            cards=cards,
            output_path=output_path,
            deck_name=deck_name,
            deck_description=deck_description,
            progress_callback=progress_callback,
            include_note_id=include_note_id,
            create_backup=create_backup,
        )


def export_cards_to_apkg_safe(
    cards: list[Card],
    output_path: str | Path,
    deck_name: str,
    deck_description: str = "",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    media_files: list[str] | None = None,
) -> dict[str, Any]:
    """
    Safely export cards to APKG file with validation, rollback, and progress tracking.

    This is the recommended function for new code. It provides:
    - Comprehensive input and output validation
    - Transaction-like operations with rollback on failure
    - Progress tracking and detailed error reporting
    - Automatic backup creation and cleanup
    - Resource management and memory safety

    Args:
        cards: List of Card objects to export
        output_path: Path to output APKG file
        deck_name: Deck name
        deck_description: Optional deck description
        progress_callback: Optional callback for progress updates
        media_files: Optional list of media file paths to include

    Returns:
        Dictionary with operation results and metadata

    Raises:
        DeckExportError: If export fails (with automatic rollback)
    """
    with safe_export_context() as exporter:
        return exporter.export_to_apkg(
            cards=cards,
            output_path=output_path,
            deck_name=deck_name,
            deck_description=deck_description,
            progress_callback=progress_callback,
            media_files=media_files,
        )
