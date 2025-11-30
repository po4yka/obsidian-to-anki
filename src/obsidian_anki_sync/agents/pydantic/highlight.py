"""Highlight agent using PydanticAI to suggest QA candidates."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from obsidian_anki_sync.agents.exceptions import HighlightError, StructuredOutputError
from obsidian_anki_sync.agents.improved_prompts import HIGHLIGHT_SYSTEM_PROMPT
from obsidian_anki_sync.agents.models import HighlightedQA, HighlightResult
from obsidian_anki_sync.utils.logging import get_logger

from .models import HighlightDeps, HighlightOutput

if TYPE_CHECKING:
    from pydantic_ai.models.openai import OpenAIChatModel

    from obsidian_anki_sync.models import NoteMetadata

logger = get_logger(__name__)


class HighlightAgentAI:
    """Agent that extracts potential Q&A pairs from incomplete notes."""

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.agent: Agent[HighlightDeps, HighlightOutput] = Agent(
            model=self.model,
            output_type=HighlightOutput,
            system_prompt=HIGHLIGHT_SYSTEM_PROMPT,
        )
        logger.info("highlight_agent_initialized", model=str(model))

    async def highlight(
        self,
        note_content: str,
        metadata: NoteMetadata,
        max_candidates: int = 3,
    ) -> HighlightResult:
        """Run highlight analysis to propose QA candidates."""

        deps = HighlightDeps(
            note_content=note_content,
            metadata=metadata,
            max_candidates=max_candidates,
        )

        preview = note_content[:2000]
        prompt = f"""Analyze this note and propose candidate question/answer pairs
with short summaries and actionable suggestions. Focus on identifying
well-formed Q&A snippets that could be turned into Anki cards.

Title: {metadata.title}
Topic: {metadata.topic}
Tags: {", ".join(metadata.tags)}
Language Tags: {", ".join(metadata.language_tags)}
Max Candidates: {max_candidates}

Note Content Preview (truncated):\n{preview}
"""

        start_time = time.time()
        try:
            result = await self.agent.run(prompt, deps=deps)
            output: HighlightOutput = result.output
            qa_candidates = [
                HighlightedQA(
                    question=qa.question.strip(),
                    answer=qa.answer.strip(),
                    confidence=max(0.0, min(1.0, qa.confidence)),
                    source_excerpt=qa.source_excerpt,
                    anchor=qa.anchor,
                )
                for qa in output.qa_candidates
            ]

            highlight_result = HighlightResult(
                qa_candidates=qa_candidates,
                summaries=[summary.strip() for summary in output.summaries],
                suggestions=[s.strip() for s in output.suggestions],
                detected_sections=[s.strip() for s in output.detected_sections],
                confidence=max(0.0, min(1.0, output.confidence)),
                note_status=output.note_status or "unknown",
                analysis_time=output.analysis_time or (time.time() - start_time),
                raw_excerpt=output.raw_excerpt,
            )

            logger.info(
                "highlight_agent_complete",
                candidates=len(highlight_result.qa_candidates),
                confidence=highlight_result.confidence,
            )
            return highlight_result
        except ValueError as exc:
            logger.error("highlight_agent_output_parse_error", error=str(exc))
            error_msg = "Failed to parse highlight agent output"
            raise StructuredOutputError(
                error_msg,
                details={"title": metadata.title},
            ) from exc
        except Exception as exc:
            logger.error("highlight_agent_failed", error=str(exc))
            error_msg = f"Highlight agent failed: {exc!s}"
            raise HighlightError(
                error_msg,
                details={"title": metadata.title},
            ) from exc
