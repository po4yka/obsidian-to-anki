"""Adapter for bridging existing data models with LangChain agent models.

This module provides conversion functions between:
- Existing models (NoteMetadata, QAPair, Card) † † LangChain models (NoteContext, ProposedCard, CardDecision)
"""

from typing import Any, Optional

from obsidian_anki_sync.agents.langchain.models import (
    CardDecision,
    Difficulty,
    ExistingAnkiNote,
    Language,
    NoteContext,
    NoteContextFrontmatter,
    NoteContextSections,
    ProposedCard,
)
from obsidian_anki_sync.models import Card, Manifest, NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AgentSystemAdapter:
    """Adapter for converting between existing and LangChain models."""

    @staticmethod
    def to_note_context(
        metadata: NoteMetadata,
        qa_pair: QAPair,
        vault_root: str,
        config: Any,
        existing_anki_note: Optional[dict] = None,
    ) -> NoteContext:
        """Convert existing models to NoteContext.

        Args:
            metadata: Note metadata from parser
            qa_pair: Q&A pair from parser
            vault_root: Path to vault root
            config: Application config
            existing_anki_note: Optional existing Anki note info

        Returns:
            NoteContext for LangChain agents
        """
        logger.debug("adapter_to_note_context", note_id=metadata.id)

        # Map language
        primary_lang = Language.EN if "en" in metadata.language_tags else Language.RU

        # Map difficulty
        difficulty = Difficulty.MEDIUM
        if metadata.difficulty:
            try:
                difficulty = Difficulty(metadata.difficulty.lower())
            except ValueError:
                pass

        # Build frontmatter
        frontmatter = NoteContextFrontmatter(
            title=metadata.title,
            lang=primary_lang,
            topic=metadata.topic,
            difficulty=difficulty,
            tags=list(metadata.tags) if metadata.tags else [],
            card_type_hint=None,  # Not in existing model
            deck_hint=None,  # Will be constructed
            bilingual=len(metadata.language_tags) > 1,
        )

        # Build sections
        sections = NoteContextSections(
            question=qa_pair.question_en or qa_pair.question_ru or "",
            answer=qa_pair.answer_en or qa_pair.answer_ru or "",
            extra=None,  # Combine followups, references, related
            examples=qa_pair.context or None,
            subquestions=[],  # Not currently used
            raw_markdown=None,
        )

        # Combine extra content
        extra_parts = []
        if qa_pair.followups:
            extra_parts.append(f"**Follow-ups:**\n{qa_pair.followups}")
        if qa_pair.references:
            extra_parts.append(f"**References:**\n{qa_pair.references}")
        if qa_pair.related:
            extra_parts.append(f"**Related:**\n{qa_pair.related}")
        if extra_parts:
            sections = sections.model_copy(update={"extra": "\n\n".join(extra_parts)})

        # Build existing Anki note if provided
        existing_note = None
        if existing_anki_note:
            existing_note = ExistingAnkiNote(
                note_id=existing_anki_note.get("noteId", 0),
                model_name=existing_anki_note.get("modelName", ""),
                deck_name=existing_anki_note.get("deckName", ""),
                fields=existing_anki_note.get("fields", {}),
                tags=existing_anki_note.get("tags", []),
                last_sync_ts=existing_anki_note.get("lastSyncTs", ""),
                slug=existing_anki_note.get("slug", ""),
            )

        # Build NoteContext
        note_context = NoteContext(
            slug=metadata.id,  # Use note ID as slug
            note_path=str(metadata.id),  # Placeholder, actual path may differ
            vault_root=vault_root,
            frontmatter=frontmatter,
            sections=sections,
            existing_anki_note=existing_note,
            config_profile=None,
        )

        return note_context

    @staticmethod
    def from_proposed_card_to_card(
        proposed_card: ProposedCard,
        metadata: NoteMetadata,
        qa_pair: QAPair,
    ) -> Card:
        """Convert ProposedCard to existing Card model.

        Args:
            proposed_card: LangChain ProposedCard
            metadata: Original note metadata
            qa_pair: Original Q&A pair

        Returns:
            Card for existing sync system
        """
        logger.debug("adapter_from_proposed_card", slug=proposed_card.slug)

        # For now, this would require generating APF HTML
        # This is a placeholder - actual implementation would depend on
        # whether we want to use APF format or direct field mapping

        # Import needed for Card creation
        from obsidian_anki_sync.slug import compute_slug_hash

        # Generate APF-like HTML or use fields directly
        # For simplicity, we'll create a manifest-based card

        manifest = Manifest(
            slug=proposed_card.slug,
            slug_base=(
                proposed_card.slug.split("-")[0]
                if "-" in proposed_card.slug
                else proposed_card.slug
            ),
            lang=proposed_card.language.value,
            source_path=proposed_card.origin.note_path,
            source_anchor="",
            note_id=metadata.id,
            note_title=metadata.title,
            card_index=qa_pair.card_index,
            guid="",
        )

        # Simple HTML wrapper (actual APF generation would be more complex)
        apf_html = f"""<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: {proposed_card.slug} | CardType: {proposed_card.card_type.value} | Tags: {' '.join(proposed_card.tags)} -->

<!-- manifest: {manifest} -->

<!-- Front -->
{proposed_card.fields.get('Front', '')}

<!-- Back -->
{proposed_card.fields.get('Back', '')}

<!-- Additional -->
{proposed_card.fields.get('Extra', '')}

<!-- END_CARDS -->"""

        content_hash = compute_slug_hash(apf_html)

        card = Card(
            slug=proposed_card.slug,
            lang=proposed_card.language.value,
            apf_html=apf_html,
            manifest=manifest,
            content_hash=content_hash,
            note_type=proposed_card.model_name,
            tags=proposed_card.tags,
            guid="",  # Will be generated
        )

        return card

    @staticmethod
    def from_card_decision_to_sync_action(
        decision: CardDecision,
    ) -> tuple[str, Optional[Card]]:
        """Convert CardDecision to sync action.

        Args:
            decision: LangChain CardDecision

        Returns:
            Tuple of (action_type, card_or_none)
        """
        logger.debug(
            "adapter_from_card_decision", slug=decision.slug, action=decision.action
        )

        action_map = {
            "create": "create",
            "update": "update",
            "skip": "skip",
            "manual_review": "skip",  # Map manual_review to skip for now
        }

        action_type = action_map.get(decision.action.value, "skip")

        # If action is create or update, we'd need to convert ProposedCard to Card
        # For now, return None as placeholder
        return action_type, None

    @staticmethod
    def format_decision_summary(decision: CardDecision) -> str:
        """Format CardDecision as human-readable summary.

        Args:
            decision: CardDecision to format

        Returns:
            Formatted summary string
        """
        return decision.summary()
