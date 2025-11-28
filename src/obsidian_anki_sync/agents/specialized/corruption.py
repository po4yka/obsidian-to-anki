"""Content corruption repair agent."""

import re
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseSpecializedAgent, ContentRepairAgent
from .models import AgentResult

logger = get_logger(__name__)


class ContentCorruptionAgent(BaseSpecializedAgent):
    """Agent specialized in repairing content corruption issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair content corruption like repetitive patterns."""
        # First try rule-based corruption repair
        rule_based_result = self._rule_based_corruption_repair(content)
        if rule_based_result.success:
            return rule_based_result

        # Fall back to LLM-based repair
        prompt = self._create_prompt(content, context)

        try:
            if self.agent is None:
                return AgentResult(
                    success=False,
                    reasoning="ContentRepairAgent not initialized",
                    warnings=["Agent not available"],
                )
            result = self.agent.generate_repair(
                content=content, prompt=prompt, max_retries=2
            )

            if result.success and result.repaired_content:
                return AgentResult(
                    success=True,
                    content=result.repaired_content,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    warnings=result.warnings,
                )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message
                    or "Content corruption repair failed",
                    warnings=["Content corruption repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Content corruption agent error: {e}",
                warnings=["Content corruption agent execution failed"],
            )

    def _rule_based_corruption_repair(self, content: str) -> AgentResult:
        """Try rule-based corruption pattern removal."""
        try:
            original_content = content

            # Remove repetitive alphanumeric corruption patterns
            content = re.sub(r"[a-zA-Z]\d{1,2}\s*", "", content)
            content = re.sub(r"\d{1,2}[a-zA-Z]\s*", "", content)
            content = re.sub(r"(.)\1{2,}", r"\1", content)

            # Clean up spacing issues
            content = re.sub(r"\s+", " ", content)
            content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

            if content != original_content:
                return AgentResult(
                    success=True,
                    content=content,
                    confidence=0.7,
                    reasoning="Removed corruption patterns using rules",
                    warnings=["Content was cleaned of corruption patterns"],
                )

            return AgentResult(
                success=False, reasoning="No corruption patterns detected"
            )

        except Exception as e:
            return AgentResult(
                success=False, reasoning=f"Rule-based corruption repair failed: {e}"
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create content corruption repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a content corruption repair specialist. Fix corrupted text patterns in this document.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify corrupted text patterns (repetitive characters like "a1a1a1", "b2b2b2", etc.)
2. Replace corrupted sections with appropriate Russian/English text
3. Maintain the document's meaning and structure
4. Preserve code blocks and technical content
5. Only repair obviously corrupted text, leave valid content unchanged

Return the complete repaired content with corruption removed."""
