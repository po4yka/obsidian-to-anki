"""Use case for processing discovered notes into cards."""

from dataclasses import dataclass
from typing import Any

from ...domain.entities.card import Card
from ...domain.entities.note import Note
from ...domain.interfaces.card_generator import ICardGenerator
from ...utils.logging import get_logger
from ..services.note_discovery_service import NoteDiscoveryService

logger = get_logger(__name__)


@dataclass
class ProcessNotesRequest:
    """Request data for process notes use case."""

    sample_size: int | None = None
    incremental: bool = False
    exclude_patterns: list[str | None] = None
    languages: list[str | None] = None


@dataclass
class ProcessNotesResponse:
    """Response data from process notes use case."""

    notes: list[Note]
    cards: list[Card]
    success: bool
    errors: list[str]
    stats: dict[str, Any]


class ProcessNotesUseCase:
    """Use case for processing discovered notes into cards.

    This use case coordinates the discovery of notes and their
    conversion to cards, replacing the card generation logic
    that was previously in NoteScanner.
    """

    def __init__(
        self,
        note_discovery_service: NoteDiscoveryService,
        card_generator: ICardGenerator,
    ):
        """Initialize use case with dependencies.

        Args:
            note_discovery_service: Service for discovering notes
            card_generator: Generator for creating cards from notes
        """
        self.note_discovery_service = note_discovery_service
        self.card_generator = card_generator

    def execute(self, request: ProcessNotesRequest) -> ProcessNotesResponse:
        """Execute note processing workflow.

        Args:
            request: Processing request parameters

        Returns:
            Processing response with results
        """
        logger.info(
            "processing_notes_started",
            sample_size=request.sample_size,
            incremental=request.incremental,
        )

        errors = []
        notes = []
        cards = []
        stats = {
            "notes_discovered": 0,
            "cards_generated": 0,
            "notes_with_errors": 0,
            "processing_time_seconds": 0,
        }

        import time

        start_time = time.time()

        try:
            # Step 1: Discover notes
            notes = self.note_discovery_service.discover_notes(
                sample_size=request.sample_size,
                incremental=request.incremental,
                exclude_patterns=request.exclude_patterns,
            )

            stats["notes_discovered"] = len(notes)
            logger.debug("notes_discovered", count=len(notes))

            # Step 2: Generate cards from notes
            for note in notes:
                try:
                    note_cards = self.card_generator.generate_cards_from_note(
                        note)

                    # Filter by requested languages if specified
                    if request.languages:
                        note_cards = [
                            card
                            for card in note_cards
                            if card.language in request.languages
                        ]

                    cards.extend(note_cards)
                    stats["cards_generated"] += len(note_cards)

                    logger.debug(
                        "cards_generated_for_note",
                        note_id=note.id,
                        card_count=len(note_cards),
                    )

                except Exception as e:
                    error_msg = f"Failed to generate cards for note {note.id}: {e}"
                    logger.warning(
                        "card_generation_failed", note_id=note.id, error=str(e)
                    )
                    errors.append(error_msg)
                    stats["notes_with_errors"] += 1

            # Step 3: Calculate statistics
            processing_time = time.time() - start_time
            stats["processing_time_seconds"] = round(processing_time, 2)

            # Add discovery statistics
            if notes:
                discovery_stats = self.note_discovery_service.get_note_statistics(
                    notes)
                stats["discovery_stats"] = discovery_stats

            success = len(errors) == 0 or (
                len(cards) > 0 and len(errors) < len(
                    notes) // 2  # Allow some errors
            )

            logger.info(
                "processing_notes_completed",
                success=success,
                notes_processed=len(notes),
                cards_generated=len(cards),
                errors=len(errors),
                processing_time_seconds=stats["processing_time_seconds"],
            )

            return ProcessNotesResponse(
                notes=notes,
                cards=cards,
                success=success,
                errors=errors,
                stats=stats,
            )

        except Exception as e:
            error_msg = f"Note processing failed: {e}"
            logger.error("processing_notes_failed", error=str(e))
            errors.append(error_msg)

            return ProcessNotesResponse(
                notes=notes,
                cards=cards,
                success=False,
                errors=errors,
                stats=stats,
            )
