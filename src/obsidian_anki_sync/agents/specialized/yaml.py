"""YAML frontmatter repair agent."""

from typing import Any

from ...utils.logging import get_logger
from .base import BaseSpecializedAgent, ContentRepairAgent
from .models import AgentResult

logger = get_logger(__name__)


class YAMLFrontmatterAgent(BaseSpecializedAgent):
    """Agent specialized in repairing YAML frontmatter issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair YAML frontmatter issues."""
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
                if self._validate_yaml_repair(result.repaired_content):
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
                        reasoning="YAML repair validation failed",
                        warnings=["Repaired content is still invalid YAML"],
                    )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message or "YAML repair failed",
                    warnings=["YAML frontmatter repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"YAML agent error: {e}",
                warnings=["YAML agent execution failed"],
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create YAML repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a YAML frontmatter repair specialist. Fix the corrupted YAML frontmatter in this Obsidian note.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify and fix YAML syntax errors (missing quotes, indentation, etc.)
2. Preserve all metadata fields (id, title, tags, etc.)
3. Ensure proper YAML structure with --- delimiters
4. Fix multi-line values that are improperly formatted
5. Maintain the exact same semantic meaning

Return ONLY the repaired frontmatter section, properly formatted as valid YAML."""

    def _validate_yaml_repair(self, content: str) -> bool:
        """Validate that the YAML repair is correct."""
        try:
            import frontmatter

            post = frontmatter.loads(content)
            return post.metadata is not None
        except Exception:
            return False
