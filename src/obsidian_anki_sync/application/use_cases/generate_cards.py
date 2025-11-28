"""Use case for generating cards from notes."""

from dataclasses import dataclass

from ...domain.entities.card import Card
from ...domain.entities.note import Note
from ...domain.interfaces.card_generator import ICardGenerator


@dataclass
class GenerateCardsRequest:
    """Request data for generate cards use case."""

    note: Note
    languages: list[str | None] = None
    force_regeneration: bool = False


@dataclass
class GenerateCardsResponse:
    """Response data from generate cards use case."""

    cards: list[Card]
    success: bool
    errors: list[str]
    stats: dict


class GenerateCardsUseCase:
    """Use case for generating Anki cards from Obsidian notes.

    This use case handles the card generation process,
    including validation and error handling.
    """

    def __init__(self, card_generator: ICardGenerator):
        """Initialize use case with card generator.

        Args:
            card_generator: Card generator implementation
        """
        self.card_generator = card_generator

    def execute(self, request: GenerateCardsRequest) -> GenerateCardsResponse:
        """Execute card generation for a note.

        Args:
            request: Card generation request

        Returns:
            Card generation response
        """
        errors = []
        cards = []
        stats = {
            "cards_generated": 0,
            "validation_errors": 0,
            "languages_processed": 0,
        }

        try:
            # Validate note
            if not request.note.is_valid:
                errors.append(
                    f"Note {request.note.id} is not valid for card generation"
                )
                return GenerateCardsResponse(
                    cards=[],
                    success=False,
                    errors=errors,
                    stats=stats,
                )

            # Determine languages to process
            languages = request.languages or request.note.metadata.language_tags
            if not languages:
                errors.append("No languages specified for card generation")
                return GenerateCardsResponse(
                    cards=[],
                    success=False,
                    errors=errors,
                    stats=stats,
                )

            stats["languages_processed"] = len(languages)

            # Generate cards for each language
            for language in languages:
                try:
                    note_cards = self.card_generator.generate_cards_from_note(
                        request.note
                    )
                    # Filter cards by language if needed
                    language_cards = [c for c in note_cards if c.language == language]

                    # Validate generated cards
                    for card in language_cards:
                        validation_errors = self.card_generator.validate_card_content(
                            card
                        )
                        if validation_errors:
                            stats["validation_errors"] += 1
                            errors.extend(validation_errors)
                        else:
                            cards.append(card)
                            stats["cards_generated"] += 1

                except Exception as e:
                    errors.append(
                        f"Failed to generate cards for language {language}: {e}"
                    )

            return GenerateCardsResponse(
                cards=cards,
                success=len(errors) == 0,
                errors=errors,
                stats=stats,
            )

        except Exception as e:
            errors.append(f"Card generation failed: {e}")
            return GenerateCardsResponse(
                cards=cards,
                success=False,
                errors=errors,
                stats=stats,
            )
