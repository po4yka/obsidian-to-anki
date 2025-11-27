"""Use case for synchronizing notes between Obsidian and Anki."""

from dataclasses import dataclass


from ...domain.entities.card import Card
from ...domain.entities.note import Note
from ...domain.interfaces.anki_client import IAnkiClient
from ...domain.interfaces.card_generator import ICardGenerator
from ...domain.interfaces.note_parser import INoteParser
from ...domain.interfaces.state_repository import IStateRepository


@dataclass
class SyncNotesRequest:
    """Request data for sync notes use case."""

    dry_run: bool = False
    sample_size: int | None = None
    incremental: bool = False
    build_index: bool = True


@dataclass
class SyncNotesResponse:
    """Response data from sync notes use case."""

    obsidian_cards: list[Card]
    anki_cards: list[Card]
    sync_actions: list[dict[str, any]]
    stats: dict[str, int]
    success: bool
    errors: list[str]


class SyncNotesUseCase:
    """Use case for synchronizing notes between Obsidian and Anki.

    This use case orchestrates the complete synchronization process:
    1. Discover and parse Obsidian notes
    2. Generate cards from notes
    3. Compare with Anki state
    4. Determine sync actions
    5. Apply changes (unless dry run)
    """

    def __init__(
        self,
        note_parser: INoteParser,
        card_generator: ICardGenerator,
        state_repository: IStateRepository,
        anki_client: IAnkiClient,
    ):
        """Initialize use case with dependencies.

        Args:
            note_parser: Parser for Obsidian notes
            card_generator: Generator for Anki cards
            state_repository: Repository for state persistence
            anki_client: Client for Anki communication
        """
        self.note_parser = note_parser
        self.card_generator = card_generator
        self.state_repository = state_repository
        self.anki_client = anki_client

    def execute(self, request: SyncNotesRequest) -> SyncNotesResponse:
        """Execute the sync notes use case.

        Args:
            request: Sync request parameters

        Returns:
            Sync response with results
        """
        errors = []
        stats = {
            "notes_processed": 0,
            "cards_generated": 0,
            "cards_created": 0,
            "cards_updated": 0,
            "cards_deleted": 0,
            "errors": 0,
        }

        try:
            # Step 1: Discover and parse Obsidian notes
            obsidian_notes = self._discover_notes(request)
            stats["notes_processed"] = len(obsidian_notes)

            # Step 2: Generate cards from notes
            obsidian_cards = []
            for note in obsidian_notes:
                try:
                    cards = self.card_generator.generate_cards_from_note(note)
                    obsidian_cards.extend(cards)
                except Exception as e:
                    errors.append(
                        f"Failed to generate cards for note {note.id}: {e}")
                    stats["errors"] += 1

            stats["cards_generated"] = len(obsidian_cards)

            # Step 3: Get current Anki state
            anki_cards = self._get_anki_cards()

            # Step 4: Determine sync actions
            sync_actions = self._determine_sync_actions(
                obsidian_cards, anki_cards)

            # Step 5: Apply changes (unless dry run)
            if not request.dry_run:
                apply_result = self._apply_sync_actions(sync_actions)
                stats.update(apply_result)

            # Step 6: Update statistics
            stats["cards_created"] = len(
                [a for a in sync_actions if a.get("type") == "create"])
            stats["cards_updated"] = len(
                [a for a in sync_actions if a.get("type") == "update"])
            stats["cards_deleted"] = len(
                [a for a in sync_actions if a.get("type") == "delete"])

            return SyncNotesResponse(
                obsidian_cards=obsidian_cards,
                anki_cards=anki_cards,
                sync_actions=sync_actions,
                stats=stats,
                success=len(errors) == 0,
                errors=errors,
            )

        except Exception as e:
            errors.append(f"Sync failed: {e}")
            return SyncNotesResponse(
                obsidian_cards=[],
                anki_cards=[],
                sync_actions=[],
                stats=stats,
                success=False,
                errors=errors,
            )

    def _discover_notes(self, request: SyncNotesRequest) -> list[Note]:
        """Discover and parse notes from Obsidian vault.

        Args:
            request: Sync request with parameters

        Returns:
            List of parsed notes
        """
        # This is a placeholder - actual implementation would use
        # the note parser to discover and parse notes from the vault
        # For now, return empty list
        return []

    def _get_anki_cards(self) -> list[Card]:
        """Get current cards from Anki.

        Returns:
            List of cards currently in Anki
        """
        # This is a placeholder - actual implementation would query
        # the Anki client and state repository to get current cards
        return []

    def _determine_sync_actions(
        self,
        obsidian_cards: list[Card],
        anki_cards: list[Card]
    ) -> list[dict[str, any]]:
        """Determine what sync actions need to be performed.

        Args:
            obsidian_cards: Cards from Obsidian
            anki_cards: Cards currently in Anki

        Returns:
            List of sync actions to perform
        """
        # This is a placeholder - actual implementation would compare
        # cards and determine what needs to be created, updated, or deleted
        actions = []

        # Simple logic: assume all obsidian cards need to be created
        for card in obsidian_cards:
            actions.append({
                "type": "create",
                "card": card,
                "reason": "New card from Obsidian",
            })

        return actions

    def _apply_sync_actions(self, actions: list[dict[str, any]]) -> dict[str, int]:
        """Apply sync actions to Anki.

        Args:
            actions: List of sync actions to apply

        Returns:
            Statistics about applied changes
        """
        # This is a placeholder - actual implementation would use
        # the Anki client to apply the sync actions
        return {
            "applied": len(actions),
            "failed": 0,
        }
