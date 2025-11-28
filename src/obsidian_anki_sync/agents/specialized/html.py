"""HTML validation repair agent."""

from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseSpecializedAgent
from .models import AgentResult

logger = get_logger(__name__)


class HTMLValidationAgent(BaseSpecializedAgent):
    """Agent specialized in repairing HTML validation issues."""

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair HTML validation issues."""
        from obsidian_anki_sync.apf.html_generator import HTMLTemplateGenerator

        try:
            html_generator = HTMLTemplateGenerator()

            card_data = self._extract_card_data(content)

            if card_data:
                result = html_generator.generate_card_html(card_data)

                if result.is_valid:
                    return AgentResult(
                        success=True,
                        content=result.html,
                        confidence=0.9,
                        reasoning="Regenerated HTML using structured templates",
                        warnings=result.warnings,
                    )

            return AgentResult(
                success=False,
                reasoning="Could not extract card data for HTML regeneration",
                warnings=["HTML validation agent needs structured card data"],
            )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"HTML validation agent error: {e}",
                warnings=["HTML validation agent execution failed"],
            )

    def _extract_card_data(self, content: str) -> dict[str, Any | None] | None:
        """Extract card data from HTML content for regeneration."""
        return None
