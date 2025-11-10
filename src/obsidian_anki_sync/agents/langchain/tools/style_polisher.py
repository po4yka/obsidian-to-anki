"""Style Polisher Agent - Refines card wording and generates hints.

This optional LLM-powered agent:
- Shortens overly long fields while preserving meaning
- Ensures clarity and standalone readability
- Generates helpful hints
- Enforces style consistency
"""

import json
from typing import Optional, cast

from obsidian_anki_sync.agents.langchain.models import (
    NoteContext,
    ProposedCard,
    QAReport,
)
from obsidian_anki_sync.utils.logging import get_logger

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = get_logger(__name__)


STYLE_SYSTEM_PROMPT = """You are an expert at refining flashcard wording for optimal learning.

Your task is to polish a card's style without changing its meaning:
1. Shorten overly long Front while keeping it complete
2. Ensure Back is clear and standalone
3. Generate a helpful Hint if missing (subtle, no answer reveal)
4. Fix minor grammar/punctuation issues

Return the refined card fields as JSON:
{
  "fields": {
    "Front": "refined text",
    "Back": "refined text",
    "Extra": "refined text (optional)",
    "Hint": "generated or refined hint (optional)"
  },
  "changes": ["list of changes made"]
}

Only respond with JSON."""


class StylePolisherTool:
    """Tool for refining card style and generating hints."""

    def __init__(self, llm: "BaseChatModel", enabled: bool = True):
        """Initialize Style Polisher Tool.

        Args:
            llm: LangChain chat model
            enabled: Whether to actually polish (if False, returns unchanged)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for StylePolisherTool")

        self.llm = llm
        self.enabled = enabled
        logger.info("style_polisher_initialized", enabled=enabled)

    def polish(
        self,
        proposed_card: ProposedCard,
        note_context: Optional[NoteContext] = None,
        qa_report: Optional[QAReport] = None,
    ) -> ProposedCard:
        """Polish card style.

        Args:
            proposed_card: Card to polish
            note_context: Optional context for reference
            qa_report: Optional QA report with issues to address

        Returns:
            Polished ProposedCard (or unchanged if disabled)
        """
        if not self.enabled:
            return proposed_card

        logger.debug("style_polish_start", slug=proposed_card.slug)

        try:
            # Build context
            context = self._build_context(proposed_card, note_context, qa_report)

            # Call LLM
            messages = [
                SystemMessage(content=STYLE_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]

            response = self.llm.invoke(messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Parse and apply changes
            polished_fields = self._parse_response(response_text)

            # Create new ProposedCard with polished fields
            return cast(
                ProposedCard,
                proposed_card.model_copy(update={"fields": polished_fields}),
            )

        except Exception as e:
            logger.warning("style_polish_failed", slug=proposed_card.slug, error=str(e))
            return proposed_card

    def _build_context(
        self,
        card: ProposedCard,
        note_context: Optional[NoteContext],
        qa_report: Optional[QAReport],
    ) -> str:
        """Build context for style polishing."""
        parts = ["## Current Card Fields:"]
        for name, value in card.fields.items():
            parts.append(f"**{name}**: {value}")

        if qa_report and qa_report.issues:
            parts.append("\n## Issues to Address:")
            for issue in qa_report.issues:
                parts.append(f"- {issue.message}")

        parts.append("\nPlease refine the fields.")
        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> dict[str, str]:
        """Parse LLM response to extract refined fields."""
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        data = json.loads(response_text)
        return cast(dict[str, str], data.get("fields", {}))
