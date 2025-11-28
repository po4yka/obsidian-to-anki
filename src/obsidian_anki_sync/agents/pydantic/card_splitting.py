"""Card splitting agent using PydanticAI.

Determines whether a note should generate one card or multiple cards.
"""

import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.agents.card_splitting_prompts import (
    CARD_SPLITTING_DECISION_PROMPT,
)
from obsidian_anki_sync.agents.exceptions import ModelError, StructuredOutputError
from obsidian_anki_sync.agents.models import CardSplitPlan, CardSplittingResult
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger

from .models import CardSplittingDeps, CardSplittingOutput

logger = get_logger(__name__)


class CardSplittingAgentAI:
    """PydanticAI-based card splitting agent.

    Determines whether a note should generate one card or multiple cards,
    and provides a splitting plan if needed.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize card splitting agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[CardSplittingDeps, CardSplittingOutput] = Agent(
            model=self.model,
            output_type=CardSplittingOutput,
            system_prompt=CARD_SPLITTING_DECISION_PROMPT,
        )

        logger.info("pydantic_ai_card_splitting_agent_initialized", model=str(model))

    async def analyze(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
    ) -> CardSplittingResult:
        """Analyze note and determine if splitting is needed.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: Parsed Q/A pairs

        Returns:
            CardSplittingResult with splitting decision and plan
        """
        logger.info(
            "pydantic_ai_card_splitting_start",
            title=metadata.title,
            qa_count=len(qa_pairs),
        )

        # Create dependencies
        deps = CardSplittingDeps(
            note_content=note_content, metadata=metadata, qa_pairs=qa_pairs
        )

        # Build analysis prompt
        prompt = f"""Analyze this note and determine if it should generate one or multiple cards:

Title: {metadata.title}
Topic: {metadata.topic}
Q&A Pairs: {len(qa_pairs)}

Content Preview:
{note_content[:800]}...

Questions:
"""
        for idx, qa in enumerate(qa_pairs[:3], 1):
            prompt += f"\n{idx}. {qa.question_en[:150]}"
            if len(qa.question_en) > 150:
                prompt += "..."

        prompt += "\n\nShould this note generate one card or split into multiple cards? Provide a detailed splitting plan."

        try:
            start_time = time.time()

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: CardSplittingOutput = result.data

            # Convert split plan
            split_plans = []
            for plan_output in output.split_plan:
                split_plan = CardSplitPlan(
                    card_number=plan_output.card_number,
                    concept=plan_output.concept,
                    question=plan_output.question,
                    answer_summary=plan_output.answer_summary,
                    rationale=plan_output.rationale,
                )
                split_plans.append(split_plan)

            # Create result
            splitting_result = CardSplittingResult(
                should_split=output.should_split,
                card_count=output.card_count,
                # type: ignore[arg-type]
                splitting_strategy=output.splitting_strategy,
                split_plan=split_plans,
                reasoning=output.reasoning,
                decision_time=time.time() - start_time,
                confidence=output.confidence,
                fallback_strategy=output.fallback_strategy,
            )

            logger.info(
                "pydantic_ai_card_splitting_complete",
                should_split=output.should_split,
                card_count=output.card_count,
                strategy=output.splitting_strategy,
                confidence=output.confidence,
                fallback_strategy=output.fallback_strategy,
            )

            # Log warning if confidence is low
            if output.confidence < 0.6:
                logger.warning(
                    "card_splitting_low_confidence",
                    confidence=output.confidence,
                    strategy=output.splitting_strategy,
                    fallback=output.fallback_strategy,
                )

            return splitting_result

        except ValueError as e:
            logger.error("pydantic_ai_card_splitting_parse_error", error=str(e))
            msg = "Failed to parse card splitting output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_card_splitting_timeout", error=str(e))
            msg = "Card splitting analysis timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_card_splitting_failed", error=str(e))
            # Return conservative fallback
            logger.warning("card_splitting_agent_fallback", error=str(e))
            return CardSplittingResult(
                should_split=False,
                card_count=1,
                splitting_strategy="none",
                split_plan=[
                    CardSplitPlan(
                        card_number=1,
                        concept=metadata.title,
                        question=qa_pairs[0].question_en if qa_pairs else "Question",
                        answer_summary=(
                            qa_pairs[0].answer_en[:100] if qa_pairs else "Answer"
                        ),
                        rationale=f"Fallback: Agent failed ({e!s})",
                    )
                ],
                reasoning=f"Card splitting agent failed: {e!s}. Defaulting to single card.",
                decision_time=0.0,
                confidence=0.3,
                fallback_strategy="none",
            )
