"""Duplicate detection agent using PydanticAI.

Analyzes cards to identify duplicates and overlapping content.
"""

import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.agents.duplicate_detection_prompts import (
    DUPLICATE_DETECTION_PROMPT,
)
from obsidian_anki_sync.agents.models import (
    DuplicateDetectionResult,
    DuplicateMatch,
    GeneratedCard,
)
from obsidian_anki_sync.utils.logging import get_logger

from .models import DuplicateDetectionDeps, DuplicateDetectionOutput

logger = get_logger(__name__)


class DuplicateDetectionAgentAI:
    """PydanticAI-based duplicate detection agent.

    Analyzes cards to identify duplicates and overlapping content.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize duplicate detection agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[DuplicateDetectionDeps, DuplicateDetectionOutput] = Agent(
            model=self.model,
            output_type=DuplicateDetectionOutput,
            system_prompt=DUPLICATE_DETECTION_PROMPT,
        )

        logger.info(
            "pydantic_ai_duplicate_detection_agent_initialized", model=str(model)
        )

    async def check_duplicate(
        self, new_card: GeneratedCard, existing_card: GeneratedCard
    ) -> tuple[bool, float, str]:
        """Check if new card is a duplicate of existing card.

        Args:
            new_card: Newly generated card
            existing_card: Existing card to compare against

        Returns:
            Tuple of (is_duplicate, similarity_score, reasoning)
        """
        start_time = time.time()

        try:
            # Extract question/answer from APF HTML
            new_q, new_a = self._extract_qa_from_apf(new_card.apf_html)
            existing_q, existing_a = self._extract_qa_from_apf(existing_card.apf_html)

            # Create dependencies
            deps = DuplicateDetectionDeps(
                new_card_question=new_q,
                new_card_answer=new_a,
                existing_card_question=existing_q,
                existing_card_answer=existing_a,
                existing_card_slug=existing_card.slug,
            )

            # Build prompt
            prompt = f"""Compare these two cards and determine if they are duplicates:

**New Card**:
Q: {new_q}
A: {new_a}

**Existing Card (slug: {existing_card.slug})**:
Q: {existing_q}
A: {existing_a}

Analyze similarity and provide your assessment."""

            logger.info(
                "pydantic_ai_duplicate_check_start",
                new_slug=new_card.slug,
                existing_slug=existing_card.slug,
            )

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: DuplicateDetectionOutput = result.output

            elapsed = time.time() - start_time

            logger.info(
                "pydantic_ai_duplicate_check_complete",
                is_duplicate=output.is_duplicate,
                similarity=output.similarity_score,
                type=output.duplicate_type,
                elapsed=elapsed,
            )

            return (output.is_duplicate, output.similarity_score, output.reasoning)

        except Exception as e:
            logger.error("pydantic_ai_duplicate_check_failed", error=str(e))
            # Conservative fallback: assume not duplicate
            return (False, 0.0, f"Detection failed: {e!s}")

    async def find_duplicates(
        self, new_card: GeneratedCard, existing_cards: list[GeneratedCard]
    ) -> DuplicateDetectionResult:
        """Find all potential duplicates of a new card.

        Args:
            new_card: Newly generated card
            existing_cards: List of existing cards to check against

        Returns:
            DuplicateDetectionResult with all matches
        """
        start_time = time.time()

        try:
            matches = []

            # Check against each existing card
            for existing_card in existing_cards:
                _is_dup, score, reasoning = await self.check_duplicate(
                    new_card, existing_card
                )

                # Determine duplicate type based on score
                if score >= 0.95:
                    dup_type = "exact"
                elif score >= 0.80:
                    dup_type = "semantic"
                elif score >= 0.50:
                    dup_type = "partial_overlap"
                else:
                    dup_type = "unique"

                if score >= 0.50:
                    match = DuplicateMatch(
                        card_slug=existing_card.slug,
                        similarity_score=score,
                        duplicate_type=dup_type,  # type: ignore[arg-type]
                        reasoning=reasoning,
                    )
                    matches.append(match)

            # Sort by similarity (highest first)
            matches.sort(key=lambda m: m.similarity_score, reverse=True)

            # Determine overall result
            best_match = matches[0] if matches else None
            is_duplicate = (
                best_match is not None and best_match.similarity_score >= 0.80
            )

            # Recommendation based on best match
            if best_match:
                if best_match.similarity_score >= 0.95:
                    recommendation = "delete"
                elif best_match.similarity_score >= 0.80:
                    recommendation = "merge"
                elif best_match.similarity_score >= 0.65:
                    recommendation = "review_manually"
                else:
                    recommendation = "keep_both"
            else:
                recommendation = "keep_both"

            result = DuplicateDetectionResult(
                is_duplicate=is_duplicate,
                best_match=best_match,
                all_matches=matches,
                recommendation=recommendation,  # type: ignore[arg-type]
                better_card=None,
                merge_suggestion=None,
                detection_time=time.time() - start_time,
            )

            logger.info(
                "pydantic_ai_find_duplicates_complete",
                new_slug=new_card.slug,
                matches_found=len(matches),
                is_duplicate=is_duplicate,
                recommendation=recommendation,
            )

            return result

        except Exception as e:
            logger.error("pydantic_ai_find_duplicates_failed", error=str(e))
            # Return safe fallback
            return DuplicateDetectionResult(
                is_duplicate=False,
                best_match=None,
                all_matches=[],
                recommendation="keep_both",  # type: ignore[arg-type]
                better_card=None,
                merge_suggestion=None,
                detection_time=0.0,
            )

    def _extract_qa_from_apf(self, apf_html: str) -> tuple[str, str]:
        """Extract question and answer from APF HTML.

        Args:
            apf_html: APF format HTML

        Returns:
            Tuple of (question, answer)
        """
        question = ""
        answer = ""

        # Extract Front (question)
        front_match = re.search(r'<div class="front">(.*?)</div>', apf_html, re.DOTALL)
        if front_match:
            question = re.sub(r"<[^>]+>", "", front_match.group(1)).strip()

        # Extract Back (answer)
        back_match = re.search(r'<div class="back">(.*?)</div>', apf_html, re.DOTALL)
        if back_match:
            answer = re.sub(r"<[^>]+>", "", back_match.group(1)).strip()

        return question or "Unknown question", answer or "Unknown answer"
