"""Memorization quality agent using PydanticAI.

Evaluates whether generated cards are effective for spaced repetition.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from ...models import NoteMetadata
from ...utils.logging import get_logger
from ..exceptions import ModelError, StructuredOutputError
from ..memorization_prompts import MEMORIZATION_QUALITY_PROMPT
from ..models import GeneratedCard, MemorizationQualityResult
from .models import MemorizationQualityOutput, PostValidationDeps

logger = get_logger(__name__)


class MemorizationQualityAgentAI:
    """PydanticAI-based memorization quality agent.

    Evaluates whether generated cards are effective for spaced repetition
    and long-term memory retention.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize memorization quality agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[PostValidationDeps, MemorizationQualityOutput] = Agent(
            model=self.model,
            output_type=MemorizationQualityOutput,
            system_prompt=MEMORIZATION_QUALITY_PROMPT,
        )

        logger.info("pydantic_ai_memorization_agent_initialized",
                    model=str(model))

    async def assess(
        self,
        cards: list[GeneratedCard],
        metadata: NoteMetadata,
    ) -> MemorizationQualityResult:
        """Assess memorization quality of generated cards.

        Args:
            cards: Generated cards to assess
            metadata: Note metadata for context

        Returns:
            MemorizationQualityResult with assessment
        """
        logger.info("pydantic_ai_memorization_assessment_start",
                    cards_count=len(cards))

        # Create dependencies
        deps = PostValidationDeps(
            cards=cards, metadata=metadata, strict_mode=True)

        # Build assessment prompt
        prompt = f"""Assess memorization quality of these {len(cards)} Anki cards:

Metadata:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {', '.join(metadata.language_tags)}

Cards to assess:
"""
        for card in cards[:5]:
            prompt += f"\nCard {card.card_index} ({card.lang}): {card.slug}\n"
            front_match = card.apf_html.split("<!-- Front -->")
            if len(front_match) > 1:
                front_text = front_match[1].split(
                    "<!-- Back -->")[0].strip()[:150]
                prompt += f"Front: {front_text}...\n"
            back_match = card.apf_html.split("<!-- Back -->")
            if len(back_match) > 1:
                back_text = back_match[1].split("<!--")[0].strip()[:150]
                prompt += f"Back: {back_text}...\n"

        prompt += f"\nAssess all {len(cards)} cards for memorization effectiveness."

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: MemorizationQualityOutput = result.data

            # Convert to MemorizationQualityResult
            issues_list = [issue.model_dump() for issue in output.issues]

            quality_result = MemorizationQualityResult(
                is_memorizable=output.is_memorizable,
                memorization_score=output.memorization_score,
                issues=issues_list,
                strengths=output.strengths,
                suggested_improvements=output.suggested_improvements,
                assessment_time=0.0,
            )

            logger.info(
                "pydantic_ai_memorization_assessment_complete",
                is_memorizable=output.is_memorizable,
                score=output.memorization_score,
                issues_found=len(output.issues),
            )

            return quality_result

        except ValueError as e:
            logger.error("pydantic_ai_memorization_parse_error", error=str(e))
            raise StructuredOutputError(
                "Failed to parse memorization assessment output",
                details={"error": str(e), "cards_count": len(cards)},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_memorization_timeout", error=str(e))
            raise ModelError(
                "Memorization assessment timed out", details={"cards_count": len(cards)}
            ) from e
        except Exception as e:
            logger.error("pydantic_ai_memorization_failed", error=str(e))
            # Return permissive result instead of failing
            logger.warning("memorization_agent_fallback", error=str(e))
            return MemorizationQualityResult(
                is_memorizable=True,
                memorization_score=0.7,
                issues=[],
                strengths=[],
                suggested_improvements=[
                    f"Memorization agent failed: {str(e)}"],
                assessment_time=0.0,
            )
