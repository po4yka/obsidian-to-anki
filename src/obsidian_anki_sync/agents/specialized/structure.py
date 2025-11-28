"""Content structure repair agent."""

import re
from typing import Any

from ...utils.logging import get_logger
from .base import BaseSpecializedAgent, ContentRepairAgent
from .models import AgentResult

logger = get_logger(__name__)


class ContentStructureAgent(BaseSpecializedAgent):
    """Agent specialized in repairing content structure issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair content structure issues like missing sections."""
        # First try rule-based repair
        rule_based_result = self._rule_based_repair(content, context)
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
                    reasoning=result.error_message or "Content structure repair failed",
                    warnings=["Content structure repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Content structure agent error: {e}",
                warnings=["Content structure agent execution failed"],
            )

    def _rule_based_repair(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Try rule-based repair first."""
        try:
            languages = self._extract_languages_from_frontmatter(content)
            if not languages:
                return AgentResult(
                    success=False, reasoning="Could not determine languages"
                )

            lines = content.splitlines()
            repaired_lines = list(lines)

            frontmatter_end = self._find_frontmatter_end(lines)

            # Check for missing question sections
            for lang in languages:
                question_marker = f"# Question ({lang.upper()})"
                if not any(question_marker in line for line in lines):
                    insert_pos = frontmatter_end + 1
                    repaired_lines.insert(insert_pos, "")
                    repaired_lines.insert(insert_pos + 1, question_marker)
                    repaired_lines.insert(insert_pos + 2, "")
                    repaired_lines.insert(
                        insert_pos + 3, f"[Question content in {lang.upper()}]"
                    )

            # Check for missing answer sections
            for lang in languages:
                answer_marker = f"## Answer ({lang.upper()})"
                if not any(answer_marker in line for line in lines):
                    question_idx = next(
                        (
                            i
                            for i, line in enumerate(repaired_lines)
                            if question_marker in line
                        ),
                        -1,
                    )
                    if question_idx >= 0:
                        insert_pos = question_idx + 1
                        while insert_pos < len(repaired_lines) and not repaired_lines[
                            insert_pos
                        ].startswith("##"):
                            insert_pos += 1

                        repaired_lines.insert(insert_pos, "")
                        repaired_lines.insert(insert_pos + 1, answer_marker)
                        repaired_lines.insert(insert_pos + 2, "")
                        repaired_lines.insert(
                            insert_pos + 3, f"[Answer content in {lang.upper()}]"
                        )

            repaired_content = "\n".join(repaired_lines)

            if repaired_content != content:
                return AgentResult(
                    success=True,
                    content=repaired_content,
                    confidence=0.8,
                    reasoning="Added missing structural sections using rules",
                    warnings=["Added placeholder content for missing sections"],
                )

            return AgentResult(
                success=False,
                reasoning="No structural issues found with rule-based approach",
            )

        except Exception as e:
            return AgentResult(
                success=False, reasoning=f"Rule-based repair failed: {e}"
            )

    def _extract_languages_from_frontmatter(self, content: str) -> list[str]:
        """Extract language tags from frontmatter."""
        lines = content.splitlines()
        in_frontmatter = False

        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                if not in_frontmatter:
                    break
                continue

            if in_frontmatter and line.startswith("language_tags:"):
                match = re.search(r"language_tags:\s*\[(.*?)\]", line)
                if match:
                    return [
                        lang.strip().strip("\"'") for lang in match.group(1).split(",")
                    ]

        return ["en"]

    def _find_frontmatter_end(self, lines: list[str]) -> int:
        """Find the end of frontmatter."""
        for i, line in enumerate(lines):
            if line.strip() == "---" and i > 0:
                return i
        return 0

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create content structure repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a content structure repair specialist. Fix missing or malformed sections in this Obsidian interview note.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify missing required sections based on language_tags in frontmatter
2. Add missing question headers (# Question (EN), # Вопрос (RU), etc.)
3. Add missing answer headers (## Answer (EN), ## Ответ (RU), etc.)
4. Ensure proper bilingual structure
5. Do NOT modify the existing content, only add missing structural elements

Return the complete repaired content with proper structure."""
