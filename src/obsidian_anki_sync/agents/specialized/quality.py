"""Quality assurance agent."""

from typing import Any

from ...utils.logging import get_logger
from .base import BaseSpecializedAgent, ContentRepairAgent
from .models import AgentResult

logger = get_logger(__name__)


class QualityAssuranceAgent(BaseSpecializedAgent):
    """General quality assurance agent for unspecified issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """General quality assurance and repair."""
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
                    reasoning=result.error_message or "Quality assurance repair failed",
                    warnings=["Quality assurance repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Quality assurance agent error: {e}",
                warnings=["Quality assurance agent execution failed"],
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create general quality assurance prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a general quality assurance specialist. Fix any issues in this document to make it valid and usable.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify and fix any structural or formatting issues
2. Ensure the document follows proper markdown conventions
3. Preserve all meaningful content
4. Make minimal changes necessary to resolve the issues

Return the repaired content."""
