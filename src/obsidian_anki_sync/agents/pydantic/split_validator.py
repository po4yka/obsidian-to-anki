"""Split validation agent using PydanticAI.

Validates whether a proposed card split is necessary and optimal.
"""

import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from ...models import NoteMetadata
from ...utils.logging import get_logger
from ..card_splitting_prompts import SPLIT_VALIDATION_PROMPT
from ..exceptions import ModelError, StructuredOutputError
from ..models import CardSplittingResult, SplitValidationResult

logger = get_logger(__name__)


class SplitValidatorAgentAI:
    """PydanticAI-based split validation agent.

    Reviews proposed card splits to prevent over-fragmentation and ensure quality.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.0):
        """Initialize split validator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[None, SplitValidationResult] = Agent(
            model=self.model,
            output_type=SplitValidationResult,
            system_prompt=SPLIT_VALIDATION_PROMPT,
        )

        logger.info("pydantic_ai_split_validator_initialized",
                    model=str(model))

    async def validate(
        self,
        note_content: str,
        metadata: NoteMetadata,
        splitting_result: CardSplittingResult,
    ) -> SplitValidationResult:
        """Validate a proposed split plan.

        Args:
            note_content: Full note content
            metadata: Note metadata
            splitting_result: The proposed split plan to validate

        Returns:
            SplitValidationResult with validation decision
        """
        logger.info(
            "pydantic_ai_split_validation_start",
            title=metadata.title,
            proposed_cards=splitting_result.card_count,
            strategy=splitting_result.splitting_strategy,
        )

        # Format the input for the model
        split_plan_str = "Proposed Split Plan:\n"
        split_plan_str += f"Strategy: {splitting_result.splitting_strategy}\n"
        split_plan_str += f"Card Count: {splitting_result.card_count}\n"
        split_plan_str += f"Reasoning: {splitting_result.reasoning}\n\n"
        split_plan_str += "Cards:\n"

        for i, card in enumerate(splitting_result.split_plan, 1):
            split_plan_str += f"{i}. Concept: {card.concept}\n"
            split_plan_str += f"   Question: {card.question}\n"
            split_plan_str += f"   Answer Summary: {card.answer_summary}\n"
            split_plan_str += f"   Rationale: {card.rationale}\n\n"

        prompt = f"""Review this split plan for the following note:

Title: {metadata.title}
Topic: {metadata.topic}

Original Note Content:
{note_content}

{split_plan_str}
"""

        try:
            start_time = time.time()

            # Run agent
            result = await self.agent.run(prompt)
            output: SplitValidationResult = result.data
            output.validation_time = time.time() - start_time

            logger.info(
                "pydantic_ai_split_validation_complete",
                is_valid=output.is_valid,
                score=output.validation_score,
                feedback=output.feedback[:100],
            )

            return output

        except ValueError as e:
            logger.error(
                "pydantic_ai_split_validation_parse_error", error=str(e))
            raise StructuredOutputError(
                "Failed to parse split validation output",
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_split_validation_timeout", error=str(e))
            raise ModelError(
                "Split validation timed out", details={"title": metadata.title}
            ) from e
        except Exception as e:
            logger.error("pydantic_ai_split_validation_failed", error=str(e))
            # Fallback: assume valid if validation fails to avoid blocking
            logger.warning("split_validation_agent_fallback", error=str(e))
            return SplitValidationResult(
                is_valid=True,
                validation_score=0.5,
                feedback=f"Validation failed: {str(e)}. Defaulting to valid.",
                validation_time=0.0,
            )
