"""QA Checker Agent - Validates semantic correctness and pedagogical quality.

This LLM-powered agent performs semantic QA checks to ensure:
- The answer correctly addresses the question
- The question doesn't leak the answer
- Language consistency
- Pedagogical quality
"""

import json
from typing import Optional

from obsidian_anki_sync.agents.langchain.models import (
    IssueSeverity,
    IssueType,
    NoteContext,
    ProposedCard,
    QAIssue,
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


QA_SYSTEM_PROMPT = """You are an expert at evaluating flashcard quality for spaced repetition learning.

Your task is to analyze a proposed Anki card and provide a quality assessment with specific issues.

## Quality Criteria:

**1. Answer Correctness (Critical)**
- Does the Back field correctly and completely answer the Front?
- Are there any factual errors or misleading statements?
- Is the answer at the right level of detail?

**2. No Answer Leakage (Critical)**
- Does the Front reveal the answer or contain spoilers?
- Are there obvious hints that trivialize the card?

**3. Language Consistency (High)**
- Is the language consistent with the declared language?
- For bilingual cards, is the language distribution appropriate?

**4. Completeness (Medium)**
- Are critical details missing (e.g., complexity, edge cases, prerequisites)?
- Is the answer self-contained and standalone?

**5. Pedagogical Quality (Medium)**
- Is the question clear and unambiguous?
- Is the answer structured for easy recall?
- Are examples helpful and not cluttering?

**6. Style Issues (Low)**
- Formatting, punctuation, or wording improvements

## Scoring Rubric:
- 1.0: Perfect card, ready for immediate use
- 0.9: Excellent, minor style improvements possible
- 0.8: Good, may need small fixes
- 0.7: Acceptable, has noticeable issues
- 0.6: Below standard, needs revision
- 0.5 and below: Significant problems, major revision needed

## Response Format:
Respond with a JSON object:
{
  "qa_score": 0.0-1.0,
  "issues": [
    {
      "type": "answer_mismatch" | "front_leaks_answer" | "language_mismatch" | "style_issue" | "missing_critical_detail",
      "severity": "low" | "medium" | "high",
      "message": "Clear description of the issue",
      "suggested_change": "Specific suggestion for fixing (optional)"
    }
  ],
  "auto_fixed": []
}

IMPORTANT: Only respond with the JSON object, no additional text."""


class QACheckerTool:
    """Tool for semantic QA validation of cards."""

    def __init__(
        self,
        llm: "BaseChatModel",
        min_acceptable_score: float = 0.7,
    ):
        """Initialize QA Checker Tool.

        Args:
            llm: LangChain chat model
            min_acceptable_score: Minimum score to pass QA
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for QACheckerTool")

        self.llm = llm
        self.min_acceptable_score = min_acceptable_score

        logger.info(
            "qa_checker_initialized",
            llm_type=type(llm).__name__,
            min_score=min_acceptable_score,
        )

    def check(
        self,
        note_context: NoteContext,
        proposed_card: ProposedCard,
    ) -> QAReport:
        """Perform QA check on a proposed card.

        Args:
            note_context: Original note context
            proposed_card: Card to validate

        Returns:
            QAReport with score and issues
        """
        logger.info(
            "qa_check_start",
            slug=proposed_card.slug,
            card_type=proposed_card.card_type,
        )

        # Build context for QA
        context = self._build_qa_context(note_context, proposed_card)

        # Call LLM
        try:
            report = self._perform_qa_check(context)

            logger.info(
                "qa_check_complete",
                slug=proposed_card.slug,
                score=report.qa_score,
                issues=len(report.issues),
                high_severity=sum(
                    1 for i in report.issues if i.severity == IssueSeverity.HIGH
                ),
            )

            return report

        except Exception as e:
            logger.error(
                "qa_check_failed",
                slug=proposed_card.slug,
                error=str(e),
            )
            # Return a failed report
            return QAReport(
                qa_score=0.0,
                issues=[
                    QAIssue(
                        type=IssueType.STYLE_ISSUE,
                        severity=IssueSeverity.HIGH,
                        message=f"QA check failed: {str(e)}",
                    )
                ],
            )

    def _build_qa_context(
        self, note_context: NoteContext, proposed_card: ProposedCard
    ) -> str:
        """Build context string for QA check."""
        context_parts = [
            "## Source Note",
            f"**Question**: {note_context.sections.question}",
            f"**Answer**: {note_context.sections.answer}",
        ]

        if note_context.sections.extra:
            context_parts.append(f"**Extra**: {note_context.sections.extra}")

        context_parts.extend(
            [
                "",
                "## Proposed Card",
                f"**Type**: {proposed_card.card_type.value}",
                f"**Language**: {proposed_card.language.value}",
                "",
                "**Fields**:",
            ]
        )

        for field_name, field_value in proposed_card.fields.items():
            context_parts.append(f"- **{field_name}**: {field_value}")

        context_parts.extend(
            [
                "",
                "**Tags**: " + ", ".join(proposed_card.tags),
                "",
                "Please evaluate this card and provide a quality assessment.",
            ]
        )

        return "\n".join(context_parts)

    def _perform_qa_check(self, context: str) -> QAReport:
        """Perform QA check with LLM."""
        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        try:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            report_dict = json.loads(response_text)
            return QAReport(**report_dict)

        except (json.JSONDecodeError, Exception) as e:
            logger.error("qa_report_parse_failed", error=str(e), response=response_text[:300])
            raise ValueError(f"Failed to parse QA report: {e}")
