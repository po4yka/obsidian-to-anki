"""Card generation agent using PydanticAI.

Generates APF cards from Q/A pairs using structured outputs.
"""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from obsidian_anki_sync.agents.exceptions import (
    GenerationError,
    ModelError,
    StructuredOutputError,
)
from obsidian_anki_sync.agents.improved_prompts import CARD_GENERATION_SYSTEM_PROMPT
from obsidian_anki_sync.agents.models import GeneratedCard, GenerationResult
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.logging import get_logger

from .models import CardGenerationOutput, GenerationDeps

logger = get_logger(__name__)


class GeneratorAgentAI:
    """PydanticAI-based card generation agent.

    Generates APF cards from Q/A pairs using structured outputs.
    Ensures type-safe card generation with validation.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.3):
        """Initialize generator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature for creativity
        """
        self.model = model
        self.temperature = temperature

        # Use improved system prompt with few-shot examples
        self.system_prompt = CARD_GENERATION_SYSTEM_PROMPT

        # Create PydanticAI agent with increased retry limit for output validation
        # CardGenerationOutput has nested GeneratedCard objects - give more attempts
        self.agent: Agent[GenerationDeps, CardGenerationOutput] = Agent(
            model=self.model,
            output_type=CardGenerationOutput,
            system_prompt=self.system_prompt,
            retries=5,  # Increased: complex nested schema needs more attempts
        )

        logger.info("pydantic_ai_generator_initialized", model=str(model))

    async def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
        rag_enrichment: dict | None = None,
        rag_examples: list[dict] | None = None,
    ) -> GenerationResult:
        """Generate APF cards from Q/A pairs.

        Args:
            note_content: Full note content for context
            metadata: Note metadata
            qa_pairs: Q/A pairs to convert
            slug_base: Base slug for card identifiers
            rag_enrichment: Optional RAG context enrichment data
            rag_examples: Optional few-shot examples from RAG

        Returns:
            GenerationResult with all generated cards
        """
        logger.info(
            "pydantic_ai_generation_start",
            title=metadata.title,
            qa_count=len(qa_pairs),
            has_rag_enrichment=rag_enrichment is not None,
            has_rag_examples=rag_examples is not None,
        )

        # Create dependencies
        deps = GenerationDeps(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            slug_base=slug_base,
        )

        # Build generation prompt
        prompt = f"""Generate APF cards for these Q&A pairs:

Title: {metadata.title}
Topic: {metadata.topic}
Languages: {", ".join(metadata.language_tags)}
Slug Base: {slug_base}
"""

        # Add RAG context enrichment if available
        if rag_enrichment:
            related_concepts = rag_enrichment.get("related_concepts", [])
            if related_concepts:
                prompt += "\n## Related Concepts (from knowledge base)\n"
                for concept in related_concepts[:3]:
                    title = concept.get("title", "")
                    content = concept.get("content", "")[:200]
                    if title:
                        prompt += f"- {title}: {content}\n"
                    else:
                        prompt += f"- {content}\n"

        # Add few-shot examples if available
        if rag_examples:
            prompt += "\n## Example Q&A Cards (for reference)\n"
            for i, example in enumerate(rag_examples[:2], 1):
                q = example.get("question", "")[:150]
                a = example.get("answer", "")[:200]
                prompt += f"\nExample {i}:\nQ: {q}\nA: {a}\n"

        prompt += f"\nQ&A Pairs ({len(qa_pairs)}):\n"

        for idx, qa in enumerate(qa_pairs, 1):
            prompt += f"\n{idx}. Q: {qa.question_en[:100]}...\n   A: {qa.answer_en[:100]}...\n"

        prompt += (
            "\nGenerate complete APF HTML cards for all Q&A pairs in all languages."
        )

        try:
            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: CardGenerationOutput = result.output

            # Convert cards to GeneratedCard instances with content hashes
            generated_cards: list[GeneratedCard] = []
            qa_lookup = {qa.card_index: qa for qa in qa_pairs}
            for card in output.cards:
                try:
                    qa_pair = qa_lookup.get(card.card_index)
                    content_hash = ""
                    if qa_pair is not None:
                        content_hash = compute_content_hash(qa_pair, metadata, card.lang)

                    # Create new card with computed content_hash
                    generated_card = GeneratedCard(
                        card_index=card.card_index,
                        slug=card.slug,
                        lang=card.lang,
                        apf_html=card.apf_html,
                        confidence=card.confidence if card.confidence else output.confidence,
                        content_hash=content_hash,
                    )
                    generated_cards.append(generated_card)
                except (AttributeError, ValueError) as e:
                    logger.warning(
                        "invalid_generated_card", error=str(e), card=str(card)
                    )
                    continue

            if not generated_cards:
                msg = "No valid cards generated"
                raise GenerationError(
                    msg,
                    details={
                        "title": metadata.title,
                        "qa_pairs_count": len(qa_pairs),
                        "raw_cards_count": len(output.cards),
                    },
                )

            generation_result = GenerationResult(
                cards=generated_cards,
                total_cards=len(generated_cards),
                generation_time=0.0,
                model_used=str(self.model),
            )

            logger.info(
                "pydantic_ai_generation_complete",
                cards_generated=len(generated_cards),
                confidence=output.confidence,
            )

            return generation_result

        except GenerationError:
            raise
        except ValueError as e:
            logger.error("pydantic_ai_generation_parse_error", error=str(e))
            msg = "Failed to parse generation output"
            raise StructuredOutputError(
                msg,
                details={"error": str(e), "title": metadata.title},
            ) from e
        except TimeoutError as e:
            logger.error("pydantic_ai_generation_timeout", error=str(e))
            msg = "Card generation timed out"
            raise ModelError(msg, details={"title": metadata.title}) from e
        except Exception as e:
            logger.error("pydantic_ai_generation_failed", error=str(e))
            msg = f"Card generation failed: {e!s}"
            raise GenerationError(msg, details={"title": metadata.title}) from e
