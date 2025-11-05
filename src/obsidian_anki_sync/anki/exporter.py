"""Export cards to Anki deck files (.apkg) using genanki."""

import hashlib
from pathlib import Path

import genanki

from ..models import Card
from ..utils.logging import get_logger
from .field_mapper import map_apf_to_anki_fields

logger = get_logger(__name__)


class DeckExportError(Exception):
    """Error during deck export."""

    pass


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
        raise DeckExportError(f"Failed to create deck: {e}") from e


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
        raise DeckExportError(f"Failed to convert card {card.slug}: {e}") from e


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
        raise DeckExportError(f"Failed to export deck: {e}") from e


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
