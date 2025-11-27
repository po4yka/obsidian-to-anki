"""Code block repair agent."""

from typing import Any, Dict

from ...utils.logging import get_logger
from .base import BaseSpecializedAgent
from .models import AgentResult

logger = get_logger(__name__)


class CodeBlockAgent(BaseSpecializedAgent):
    """Agent specialized in repairing code block issues."""

    def solve(self, content: str, context: Dict[str, Any]) -> AgentResult:
        """Repair code block formatting and fence issues."""
        repaired_content = self._repair_code_fences(content)

        if repaired_content != content:
            return AgentResult(
                success=True,
                content=repaired_content,
                confidence=0.8,
                reasoning="Repaired code fence issues with pattern matching",
                warnings=[],
            )
        else:
            return AgentResult(
                success=False,
                reasoning="No code fence issues detected or repairable",
                warnings=["Code block agent could not find issues to repair"],
            )

    def _repair_code_fences(self, content: str) -> str:
        """Repair code fence issues using pattern matching."""
        lines = content.splitlines()
        repaired_lines = []
        fence_stack = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if fence_stack:
                    fence_stack.pop()
                    repaired_lines.append(line)
                else:
                    fence_stack.append(stripped)
                    repaired_lines.append(line)
            else:
                repaired_lines.append(line)

        # Close any remaining open fences
        while fence_stack:
            repaired_lines.append("```")
            fence_stack.pop()

        return "\n".join(repaired_lines)
