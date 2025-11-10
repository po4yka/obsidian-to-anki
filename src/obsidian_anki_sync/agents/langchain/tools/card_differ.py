"""Card Diff Agent - Compares existing and proposed cards for safe updates.

This tool performs field-by-field comparison and determines whether
an update should be performed based on change severity and policy.
"""

from typing import Any, Optional

from obsidian_anki_sync.agents.langchain.models import (
    CardDiffResult,
    CardFieldChange,
    ChangeSeverity,
    ExistingAnkiNote,
    ProposedCard,
    RiskLevel,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class CardDifferTool:
    """Tool for comparing cards and determining update actions.

    This is primarily a rule-based tool, but can optionally use LLM
    for semantic similarity comparison.
    """

    def __init__(
        self,
        allow_content_updates: bool = True,
        allow_structural_updates: bool = False,
        llm: Optional[Any] = None,
    ):
        """Initialize Card Differ Tool.

        Args:
            allow_content_updates: Allow updates that change meaning
            allow_structural_updates: Allow model/deck changes
            llm: Optional LLM for semantic comparison (not implemented yet)
        """
        self.allow_content_updates = allow_content_updates
        self.allow_structural_updates = allow_structural_updates
        self.llm = llm

        logger.info(
            "card_differ_initialized",
            allow_content=allow_content_updates,
            allow_structural=allow_structural_updates,
        )

    def compare(
        self,
        existing: ExistingAnkiNote,
        proposed: ProposedCard,
    ) -> CardDiffResult:
        """Compare existing and proposed cards.

        Args:
            existing: Current card in Anki
            proposed: Proposed new card

        Returns:
            CardDiffResult with changes and update recommendation
        """
        logger.debug("card_diff_start", note_id=existing.note_id, slug=proposed.slug)

        changes: list[CardFieldChange] = []

        # Compare model (structural change)
        if existing.model_name != proposed.model_name:
            changes.append(
                CardFieldChange(
                    field="model_name",
                    old_value=existing.model_name,
                    new_value=proposed.model_name,
                    severity=ChangeSeverity.STRUCTURAL,
                    message=f"Model change: {existing.model_name} → {proposed.model_name}",
                )
            )

        # Compare deck (structural change)
        if existing.deck_name != proposed.deck_name:
            changes.append(
                CardFieldChange(
                    field="deck_name",
                    old_value=existing.deck_name,
                    new_value=proposed.deck_name,
                    severity=ChangeSeverity.STRUCTURAL,
                    message=f"Deck change: {existing.deck_name} → {proposed.deck_name}",
                )
            )

        # Compare fields
        for field_name, new_value in proposed.fields.items():
            old_value = existing.fields.get(field_name, "")

            if old_value != new_value:
                severity = self._classify_field_change_severity(
                    field_name, old_value, new_value
                )
                changes.append(
                    CardFieldChange(
                        field=field_name,
                        old_value=old_value,
                        new_value=new_value,
                        severity=severity,
                        message=f"Field '{field_name}' changed",
                    )
                )

        # Compare tags
        existing_tags = set(existing.tags)
        proposed_tags = set(proposed.tags)

        if existing_tags != proposed_tags:
            added = proposed_tags - existing_tags
            removed = existing_tags - proposed_tags

            changes.append(
                CardFieldChange(
                    field="tags",
                    old_value=list(existing_tags),
                    new_value=list(proposed_tags),
                    severity=ChangeSeverity.COSMETIC,
                    message=f"Tags: +{len(added)}, -{len(removed)}",
                )
            )

        # Determine if update should proceed
        should_update, reason, risk_level = self._should_update(changes)

        logger.info(
            "card_diff_complete",
            note_id=existing.note_id,
            changes=len(changes),
            should_update=should_update,
            risk=risk_level,
        )

        return CardDiffResult(
            changes=changes,
            should_update=should_update,
            reason=reason,
            risk_level=risk_level,
        )

    def _classify_field_change_severity(
        self, field_name: str, old_value: str, new_value: str
    ) -> ChangeSeverity:
        """Classify the severity of a field change.

        Args:
            field_name: Name of the field
            old_value: Original value
            new_value: New value

        Returns:
            ChangeSeverity classification
        """
        # Whitespace-only changes are cosmetic
        if old_value.strip() == new_value.strip():
            return ChangeSeverity.COSMETIC

        # Small edits (< 20% change) might be cosmetic
        if self._is_minor_edit(old_value, new_value):
            return ChangeSeverity.COSMETIC

        # Otherwise, it's a content change
        return ChangeSeverity.CONTENT

    def _is_minor_edit(self, old: str, new: str, threshold: float = 0.2) -> bool:
        """Check if edit is minor (small character difference).

        Args:
            old: Old text
            new: New text
            threshold: Maximum relative difference to be considered minor

        Returns:
            True if minor edit
        """
        if not old or not new:
            return False

        # Simple character-based similarity
        max_len = max(len(old), len(new))
        diff = abs(len(old) - len(new))

        return (diff / max_len) < threshold

    def _should_update(
        self, changes: list[CardFieldChange]
    ) -> tuple[bool, str, RiskLevel]:
        """Determine if card should be updated based on changes.

        Args:
            changes: List of detected changes

        Returns:
            Tuple of (should_update, reason, risk_level)
        """
        if not changes:
            return False, "No changes detected", RiskLevel.LOW

        # Check for structural changes
        has_structural = any(c.severity == ChangeSeverity.STRUCTURAL for c in changes)
        if has_structural and not self.allow_structural_updates:
            return (
                False,
                "Structural changes (model/deck) not allowed by policy",
                RiskLevel.HIGH,
            )

        # Check for content changes
        has_content = any(c.severity == ChangeSeverity.CONTENT for c in changes)
        if has_content and not self.allow_content_updates:
            return (
                False,
                "Content changes not allowed by policy",
                RiskLevel.MEDIUM,
            )

        # Determine risk level
        if has_structural:
            risk = RiskLevel.HIGH
        elif has_content:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        # Allow update
        return True, f"Update approved ({len(changes)} changes)", risk
