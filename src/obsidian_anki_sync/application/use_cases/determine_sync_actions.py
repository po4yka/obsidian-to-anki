"""Use case for determining sync actions between Obsidian and Anki."""

from dataclasses import dataclass


from ...domain.entities.card import Card, SyncAction, SyncActionType
from ...domain.services.content_hash_service import ContentHashService


@dataclass
class DetermineSyncActionsRequest:
    """Request data for determine sync actions use case."""

    obsidian_cards: list[Card]
    anki_cards: list[Card]
    # "prefer_obsidian" | "prefer_anki" | "manual"
    conflict_resolution: str = "prefer_obsidian"


@dataclass
class DetermineSyncActionsResponse:
    """Response data from determine sync actions use case."""

    sync_actions: list[SyncAction]
    conflicts: list[dict[str, any]]
    stats: dict[str, int]


class DetermineSyncActionsUseCase:
    """Use case for determining what sync actions need to be performed.

    This use case compares the state of cards in Obsidian vs Anki
    and determines what create, update, or delete operations are needed.
    """

    def __init__(self):
        """Initialize use case."""
        pass

    def execute(self, request: DetermineSyncActionsRequest) -> DetermineSyncActionsResponse:
        """Determine sync actions by comparing Obsidian and Anki states.

        Args:
            request: Sync action determination request

        Returns:
            Sync action determination response
        """
        sync_actions = []
        conflicts = []
        stats = {
            "obsidian_cards": len(request.obsidian_cards),
            "anki_cards": len(request.anki_cards),
            "actions_create": 0,
            "actions_update": 0,
            "actions_delete": 0,
            "conflicts": 0,
        }

        # Create lookup dictionaries for efficient comparison
        obsidian_by_slug = {card.slug: card for card in request.obsidian_cards}
        anki_by_slug = {card.slug: card for card in request.anki_cards}

        # Find cards that exist in both systems
        common_slugs = set(obsidian_by_slug.keys()) & set(anki_by_slug.keys())

        # Find cards that only exist in Obsidian (need to be created)
        obsidian_only_slugs = set(
            obsidian_by_slug.keys()) - set(anki_by_slug.keys())

        # Find cards that only exist in Anki (may need to be deleted)
        anki_only_slugs = set(anki_by_slug.keys()) - \
            set(obsidian_by_slug.keys())

        # Process cards that exist in both systems
        for slug in common_slugs:
            obsidian_card = obsidian_by_slug[slug]
            anki_card = anki_by_slug[slug]

            action = self._determine_action_for_existing_card(
                obsidian_card, anki_card, request.conflict_resolution
            )

            if action:
                sync_actions.append(action)
                if action.action_type == SyncActionType.UPDATE:
                    stats["actions_update"] += 1
            elif self._is_conflict(obsidian_card, anki_card):
                conflicts.append({
                    "slug": slug,
                    "obsidian_card": obsidian_card,
                    "anki_card": anki_card,
                    "reason": "Content differs between systems",
                })
                stats["conflicts"] += 1

        # Process cards that only exist in Obsidian (create actions)
        for slug in obsidian_only_slugs:
            obsidian_card = obsidian_by_slug[slug]
            action = SyncAction(
                action_type=SyncActionType.CREATE,
                card=obsidian_card,
                reason="New card from Obsidian",
            )
            sync_actions.append(action)
            stats["actions_create"] += 1

        # Process cards that only exist in Anki (delete actions)
        for slug in anki_only_slugs:
            anki_card = anki_by_slug[slug]
            action = SyncAction(
                action_type=SyncActionType.DELETE,
                card=anki_card,
                anki_guid=anki_card.anki_guid,
                reason="Card no longer exists in Obsidian",
            )
            sync_actions.append(action)
            stats["actions_delete"] += 1

        return DetermineSyncActionsResponse(
            sync_actions=sync_actions,
            conflicts=conflicts,
            stats=stats,
        )

    def _determine_action_for_existing_card(
        self,
        obsidian_card: Card,
        anki_card: Card,
        conflict_resolution: str
    ) -> SyncAction | None:
        """Determine action for a card that exists in both systems.

        Args:
            obsidian_card: Card from Obsidian
            anki_card: Card from Anki
            conflict_resolution: How to resolve conflicts

        Returns:
            SyncAction if one is needed, None if no action needed
        """
        # Compare content hashes to see if update is needed
        obsidian_hash = obsidian_card.content_hash
        anki_content_hash = self._get_anki_card_hash(anki_card)

        if obsidian_hash != anki_content_hash:
            # Content differs
            if conflict_resolution == "prefer_obsidian":
                return SyncAction(
                    action_type=SyncActionType.UPDATE,
                    card=obsidian_card,
                    anki_guid=anki_card.anki_guid,
                    reason="Obsidian version is newer",
                )
            elif conflict_resolution == "prefer_anki":
                # No action needed - keep Anki version
                return None
            else:  # manual resolution
                # Would need manual intervention - for now, prefer Obsidian
                return SyncAction(
                    action_type=SyncActionType.UPDATE,
                    card=obsidian_card,
                    anki_guid=anki_card.anki_guid,
                    reason="Manual resolution needed - using Obsidian version",
                )

        # Content is the same - no action needed
        return None

    def _is_conflict(self, obsidian_card: Card, anki_card: Card) -> bool:
        """Check if there is a genuine conflict between cards.

        Args:
            obsidian_card: Card from Obsidian
            anki_card: Card from Anki

        Returns:
            True if there is a conflict requiring manual resolution
        """
        # For now, consider it a conflict if content hashes differ
        # In a more sophisticated implementation, this could check
        # for different types of conflicts (structural vs content)
        obsidian_hash = obsidian_card.content_hash
        anki_hash = self._get_anki_card_hash(anki_card)
        return obsidian_hash != anki_hash

    def _get_anki_card_hash(self, anki_card: Card) -> str:
        """Get content hash for an Anki card.

        Args:
            anki_card: Card from Anki

        Returns:
            Content hash
        """
        content = f"{anki_card.apf_html}{anki_card.note_type}{','.join(sorted(anki_card.tags))}"
        return ContentHashService.compute_hash(content)
