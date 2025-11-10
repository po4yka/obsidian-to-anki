"""Enhanced Card Diff Agent with semantic analysis and intelligent change detection.

This enhanced version uses LLM to understand the semantic meaning of changes,
classify change types, and provide intelligent recommendations for updates.
"""

import difflib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional, cast

from obsidian_anki_sync.agents.langchain.models import (
    CardDiffResult,
    CardFieldChange,
    ChangeSeverity,
    ExistingAnkiNote,
    ProposedCard,
    RiskLevel,
)
from obsidian_anki_sync.utils.logging import get_logger

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = get_logger(__name__)


class ChangeType(str, Enum):
    """Classification of change types."""

    TYPO_FIX = "typo_fix"  # Spelling/grammar corrections
    CLARIFICATION = "clarification"  # Added details without changing meaning
    CONTENT_ADDITION = "content_addition"  # New information added
    CONTENT_REMOVAL = "content_removal"  # Information removed
    REPHRASING = "rephrasing"  # Same meaning, different words
    MEANING_CHANGE = "meaning_change"  # Actual change in content/meaning
    FORMATTING = "formatting"  # Whitespace, markdown, etc.
    COMPLETE_REWRITE = "complete_rewrite"  # Entirely different card


class ConflictResolution(str, Enum):
    """Strategies for resolving update conflicts."""

    OBSIDIAN_WINS = "obsidian_wins"  # Always use Obsidian version
    ANKI_WINS = "anki_wins"  # Keep Anki version (skip update)
    MERGE = "merge"  # Attempt to merge changes
    MANUAL = "manual"  # Require user review
    NEWEST_WINS = "newest_wins"  # Use most recently modified


SEMANTIC_DIFF_SYSTEM_PROMPT = """You are an expert at analyzing changes between two versions of flashcard content.

Your task is to analyze the differences between an existing Anki card and a proposed update,
and provide detailed insights about the nature of the changes.

For each field that changed, determine:

1. **Change Type**: What kind of change is this?
   - typo_fix: Spelling/grammar corrections only
   - clarification: Added details that clarify without changing core meaning
   - content_addition: Significant new information added
   - content_removal: Information was removed
   - rephrasing: Same meaning expressed differently
   - meaning_change: The actual meaning/answer has changed
   - formatting: Only formatting/whitespace changes
   - complete_rewrite: Entirely different content

2. **Severity**: How significant is this change?
   - cosmetic: Safe to apply, no impact on learning
   - content: Changes content but not fundamentally
   - structural: Changes the nature of the card

3. **Recommendation**: Should this update be applied?
   - approve: Safe to update
   - review: Needs human review
   - reject: Should not be updated

4. **Reasoning**: Brief explanation of your assessment

Respond with JSON only:
{
  "field_analyses": {
    "Front": {
      "change_type": "typo_fix",
      "severity": "cosmetic",
      "recommendation": "approve",
      "reasoning": "Fixed spelling: 'recieve' → 'receive'",
      "semantic_similarity": 0.99,
      "preserves_learning": true
    },
    "Back": { ... }
  },
  "overall_assessment": {
    "should_update": true,
    "risk_level": "low",
    "update_reason": "Minor typo fixes, safe to apply",
    "conflict_detected": false,
    "learning_impact": "none"
  }
}

IMPORTANT: Consider the impact on spaced repetition learning. Changes that alter
the fundamental question/answer relationship may disrupt learning progress."""


class EnhancedCardDiffer:
    """Enhanced card differ with semantic analysis.

    This tool uses LLM to understand the semantic nature of changes between
    existing and proposed cards, providing intelligent update recommendations.
    """

    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        allow_content_updates: bool = True,
        allow_structural_updates: bool = False,
        conflict_resolution: ConflictResolution = ConflictResolution.MANUAL,
        min_semantic_similarity: float = 0.85,
        track_history: bool = True,
    ):
        """Initialize Enhanced Card Differ.

        Args:
            llm: Optional LangChain LLM for semantic analysis
            allow_content_updates: Allow updates that change content
            allow_structural_updates: Allow model/deck changes
            conflict_resolution: Strategy for resolving conflicts
            min_semantic_similarity: Minimum similarity to auto-approve (0-1)
            track_history: Track change history for rollback
        """
        self.llm = llm
        self.allow_content_updates = allow_content_updates
        self.allow_structural_updates = allow_structural_updates
        self.conflict_resolution = conflict_resolution
        self.min_semantic_similarity = min_semantic_similarity
        self.track_history = track_history

        # Change history (in-memory for now, could be persisted)
        self._change_history: list[dict[str, Any]] = []

        logger.info(
            "enhanced_card_differ_initialized",
            has_llm=llm is not None,
            allow_content=allow_content_updates,
            allow_structural=allow_structural_updates,
            conflict_strategy=conflict_resolution.value,
        )

    def compare(
        self,
        existing: ExistingAnkiNote,
        proposed: ProposedCard,
        last_obsidian_sync: Optional[datetime] = None,
        last_anki_edit: Optional[datetime] = None,
    ) -> CardDiffResult:
        """Compare existing and proposed cards with semantic analysis.

        Args:
            existing: Current card in Anki
            proposed: Proposed new card
            last_obsidian_sync: When card was last synced from Obsidian
            last_anki_edit: When card was last edited in Anki

        Returns:
            Enhanced CardDiffResult with semantic analysis
        """
        logger.debug(
            "enhanced_diff_start",
            note_id=existing.note_id,
            slug=proposed.slug,
            has_llm=self.llm is not None,
        )

        # Detect conflicts (edits in both Obsidian and Anki)
        conflict_detected = self._detect_conflict(last_obsidian_sync, last_anki_edit)

        # Perform basic diff
        changes = self._compute_field_changes(existing, proposed)

        # If LLM available, perform semantic analysis
        if self.llm and changes:
            semantic_analysis = self._perform_semantic_analysis(
                existing, proposed, changes
            )
        else:
            semantic_analysis = None

        # Classify changes
        classified_changes = self._classify_changes(changes, semantic_analysis)

        # Determine update decision
        should_update, reason, risk_level = self._determine_update_decision(
            classified_changes,
            semantic_analysis,
            conflict_detected,
        )

        # Track in history if enabled
        if self.track_history:
            self._record_change(existing, proposed, classified_changes, should_update)

        # Build result
        result = CardDiffResult(
            changes=classified_changes,
            should_update=should_update,
            reason=reason,
            risk_level=risk_level,
        )

        logger.info(
            "enhanced_diff_complete",
            note_id=existing.note_id,
            changes=len(classified_changes),
            should_update=should_update,
            risk=risk_level.value,
            conflict=conflict_detected,
        )

        return result

    def _detect_conflict(
        self,
        last_obsidian_sync: Optional[datetime],
        last_anki_edit: Optional[datetime],
    ) -> bool:
        """Detect if card was edited in both Obsidian and Anki since last sync.

        Args:
            last_obsidian_sync: Last sync timestamp
            last_anki_edit: Last edit in Anki timestamp

        Returns:
            True if conflict detected
        """
        if not last_obsidian_sync or not last_anki_edit:
            return False

        # Conflict if Anki was edited after last sync
        conflict = last_anki_edit > last_obsidian_sync

        if conflict:
            logger.warning(
                "conflict_detected",
                last_sync=last_obsidian_sync.isoformat(),
                last_edit=last_anki_edit.isoformat(),
            )

        return conflict

    def _compute_field_changes(
        self, existing: ExistingAnkiNote, proposed: ProposedCard
    ) -> list[CardFieldChange]:
        """Compute basic field-level changes.

        Args:
            existing: Existing card
            proposed: Proposed card

        Returns:
            List of field changes
        """
        changes: list[CardFieldChange] = []

        # Compare model (structural)
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

        # Compare deck (structural)
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
        all_fields = set(existing.fields.keys()) | set(proposed.fields.keys())
        for field_name in all_fields:
            old_value = existing.fields.get(field_name, "")
            new_value = proposed.fields.get(field_name, "")

            if old_value != new_value:
                # Compute text similarity
                similarity = self._compute_text_similarity(old_value, new_value)

                # Initial severity classification (will be refined by LLM)
                if similarity > 0.95:
                    severity = ChangeSeverity.COSMETIC
                elif similarity > 0.70:
                    severity = ChangeSeverity.CONTENT
                else:
                    severity = ChangeSeverity.CONTENT

                changes.append(
                    CardFieldChange(
                        field=field_name,
                        old_value=old_value,
                        new_value=new_value,
                        severity=severity,
                        message=f"Field '{field_name}' changed (similarity: {similarity:.2f})",
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

        return changes

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using sequence matcher.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0-1
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _perform_semantic_analysis(
        self,
        existing: ExistingAnkiNote,
        proposed: ProposedCard,
        changes: list[CardFieldChange],
    ) -> Optional[dict[str, Any]]:
        """Perform LLM-based semantic analysis of changes.

        Args:
            existing: Existing card
            proposed: Proposed card
            changes: List of detected changes

        Returns:
            Semantic analysis results or None
        """
        if not self.llm or not changes:
            return None

        # Build comparison context
        context_parts = ["## Existing Card (Currently in Anki)"]
        for field_name, field_value in existing.fields.items():
            context_parts.append(f"**{field_name}**: {field_value}")

        context_parts.extend(
            [
                "",
                "## Proposed Card (From Updated Obsidian Note)",
            ]
        )
        for field_name, field_value in proposed.fields.items():
            context_parts.append(f"**{field_name}**: {field_value}")

        context_parts.extend(
            [
                "",
                "## Detected Changes",
            ]
        )
        for change in changes:
            if change.field not in ("tags", "model_name", "deck_name"):
                context_parts.append(f"**{change.field}**:")
                context_parts.append(f"  - Old: {change.old_value}")
                context_parts.append(f"  - New: {change.new_value}")

        context = "\n".join(context_parts)

        try:
            # Call LLM
            messages = [
                SystemMessage(content=SEMANTIC_DIFF_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]

            response = self.llm.invoke(messages)
            # Extract string content (response.content can be str or list)
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    # Extract text from list of content blocks
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                    response_text = "\n".join(parts)
                else:
                    response_text = str(content)
            else:
                response_text = str(response)

            # Parse JSON
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            analysis = json.loads(response_text)

            logger.debug(
                "semantic_analysis_complete",
                note_id=existing.note_id,
                overall_recommendation=analysis.get("overall_assessment", {}).get(
                    "should_update"
                ),
            )

            return cast(dict[str, Any], analysis)

        except Exception as e:
            logger.error(
                "semantic_analysis_failed",
                note_id=existing.note_id,
                error=str(e),
            )
            return None

    def _classify_changes(
        self,
        changes: list[CardFieldChange],
        semantic_analysis: Optional[dict[str, Any]],
    ) -> list[CardFieldChange]:
        """Classify changes using semantic analysis if available.

        Args:
            changes: List of changes
            semantic_analysis: Optional semantic analysis from LLM

        Returns:
            Changes with enhanced classification
        """
        if not semantic_analysis:
            return changes

        # Enhance changes with semantic classification
        field_analyses = semantic_analysis.get("field_analyses", {})

        enhanced_changes = []
        for change in changes:
            if change.field in field_analyses:
                analysis = field_analyses[change.field]

                # Update severity based on LLM analysis
                severity_str = analysis.get("severity", "content")
                if severity_str == "cosmetic":
                    change = change.model_copy(
                        update={"severity": ChangeSeverity.COSMETIC}
                    )
                elif severity_str == "content":
                    change = change.model_copy(
                        update={"severity": ChangeSeverity.CONTENT}
                    )
                elif severity_str == "structural":
                    change = change.model_copy(
                        update={"severity": ChangeSeverity.STRUCTURAL}
                    )

                # Enhance message with LLM reasoning
                reasoning = analysis.get("reasoning", "")
                change_type = analysis.get("change_type", "")
                enhanced_message = (
                    f"{change.message} | Type: {change_type} | {reasoning}"
                )
                change = change.model_copy(update={"message": enhanced_message})

            enhanced_changes.append(change)

        return enhanced_changes

    def _determine_update_decision(
        self,
        changes: list[CardFieldChange],
        semantic_analysis: Optional[dict[str, Any]],
        conflict_detected: bool,
    ) -> tuple[bool, str, RiskLevel]:
        """Determine whether to update card.

        Args:
            changes: List of classified changes
            semantic_analysis: Semantic analysis results
            conflict_detected: Whether a conflict was detected

        Returns:
            Tuple of (should_update, reason, risk_level)
        """
        if not changes:
            return False, "No changes detected", RiskLevel.LOW

        # Handle conflicts
        if conflict_detected:
            if self.conflict_resolution == ConflictResolution.OBSIDIAN_WINS:
                # Proceed with update
                pass
            elif self.conflict_resolution == ConflictResolution.ANKI_WINS:
                return False, "Conflict detected: Anki version preserved", RiskLevel.LOW
            elif self.conflict_resolution == ConflictResolution.MANUAL:
                return (
                    False,
                    "Conflict detected: Manual review required",
                    RiskLevel.HIGH,
                )
            # MERGE and NEWEST_WINS would need more complex logic

        # Use semantic analysis if available
        if semantic_analysis:
            overall = semantic_analysis.get("overall_assessment", {})
            should_update = overall.get("should_update", False)
            reason = overall.get("update_reason", "Semantic analysis recommendation")
            risk_str = overall.get("risk_level", "medium")

            risk_level = {
                "low": RiskLevel.LOW,
                "medium": RiskLevel.MEDIUM,
                "high": RiskLevel.HIGH,
            }.get(risk_str, RiskLevel.MEDIUM)

            return should_update, reason, risk_level

        # Fallback to rule-based decision
        has_structural = any(c.severity == ChangeSeverity.STRUCTURAL for c in changes)
        has_content = any(c.severity == ChangeSeverity.CONTENT for c in changes)

        if has_structural and not self.allow_structural_updates:
            return False, "Structural changes not allowed", RiskLevel.HIGH

        if has_content and not self.allow_content_updates:
            return False, "Content changes not allowed", RiskLevel.MEDIUM

        # Determine risk
        if has_structural:
            risk = RiskLevel.HIGH
        elif has_content:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        return True, f"Update approved ({len(changes)} changes)", risk

    def _record_change(
        self,
        existing: ExistingAnkiNote,
        proposed: ProposedCard,
        changes: list[CardFieldChange],
        approved: bool,
    ) -> None:
        """Record change in history for potential rollback.

        Args:
            existing: Existing card
            proposed: Proposed card
            changes: List of changes
            approved: Whether update was approved
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "note_id": existing.note_id,
            "slug": proposed.slug,
            "changes": [
                {
                    "field": c.field,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "severity": c.severity.value,
                }
                for c in changes
            ],
            "approved": approved,
        }

        self._change_history.append(record)

        # Keep only last 1000 changes to avoid memory bloat
        if len(self._change_history) > 1000:
            self._change_history = self._change_history[-1000:]

    def get_change_history(
        self, note_id: Optional[int] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get change history.

        Args:
            note_id: Optional filter by note ID
            limit: Maximum number of records to return

        Returns:
            List of change records
        """
        if note_id:
            history = [r for r in self._change_history if r["note_id"] == note_id]
        else:
            history = self._change_history

        return history[-limit:]

    def rollback_change(self, note_id: int, timestamp: str) -> Optional[dict[str, Any]]:
        """Get information to rollback a specific change.

        Args:
            note_id: Note ID
            timestamp: Change timestamp

        Returns:
            Change record with rollback info, or None if not found
        """
        for record in reversed(self._change_history):
            if record["note_id"] == note_id and record["timestamp"] == timestamp:
                return record

        return None
