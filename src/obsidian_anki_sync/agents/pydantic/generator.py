"""Card generation agent using PydanticAI.

Generates APF cards from Q/A pairs using structured JSON outputs,
then converts JSON to APF HTML via the APFRenderer.
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
from obsidian_anki_sync.apf.renderer import APFRenderer, APFSentinelValidator
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.logging import get_logger

from .card_schema import CardGenerationSpec, CardSpec
from .models import GenerationDeps

logger = get_logger(__name__)


class GeneratorAgentAI:
    """PydanticAI-based card generation agent.

    Generates structured JSON card specifications from Q/A pairs,
    then converts them to APF HTML using the APFRenderer.
    This approach ensures deterministic, well-formed APF output.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.3):
        """Initialize generator agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature for creativity
        """
        self.model = model
        self.temperature = temperature
        self.renderer = APFRenderer()
        self.validator = APFSentinelValidator()

        # Use improved system prompt for JSON output
        self.system_prompt = CARD_GENERATION_SYSTEM_PROMPT

        # Create PydanticAI agent with CardGenerationSpec output type
        # The agent outputs structured JSON, which we convert to APF HTML
        self.agent: Agent[GenerationDeps, CardGenerationSpec] = Agent(
            model=self.model,
            output_type=CardGenerationSpec,
            system_prompt=self.system_prompt,
            output_retries=5,  # PydanticAI output validation retries
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

        The process:
        1. Build prompt with Q/A pairs and metadata
        2. Get structured JSON (CardGenerationSpec) from LLM
        3. Convert each CardSpec to APF HTML using APFRenderer
        4. Validate APF output has all required sentinels
        5. Return GenerationResult with all cards

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
        prompt = self._build_prompt(
            metadata=metadata,
            qa_pairs=qa_pairs,
            slug_base=slug_base,
            rag_enrichment=rag_enrichment,
            rag_examples=rag_examples,
        )

        try:
            # Run agent to get structured JSON
            result = await self.agent.run(prompt, deps=deps)
            spec: CardGenerationSpec = result.output

            logger.debug(
                "pydantic_ai_json_received",
                card_count=len(spec.cards),
                confidence=spec.confidence,
            )

            # Convert JSON cards to APF HTML
            generated_cards = self._convert_specs_to_cards(
                specs=spec.cards,
                qa_pairs=qa_pairs,
                metadata=metadata,
                slug_base=slug_base,
                overall_confidence=spec.confidence,
            )

            if not generated_cards:
                msg = "No valid cards generated"
                raise GenerationError(
                    msg,
                    details={
                        "title": metadata.title,
                        "qa_pairs_count": len(qa_pairs),
                        "raw_specs_count": len(spec.cards),
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
                confidence=spec.confidence,
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

    def _build_prompt(
        self,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
        rag_enrichment: dict | None = None,
        rag_examples: list[dict] | None = None,
    ) -> str:
        """Build the generation prompt for the LLM.

        Args:
            metadata: Note metadata
            qa_pairs: Q/A pairs to convert
            slug_base: Base slug for card identifiers
            rag_enrichment: Optional RAG context enrichment
            rag_examples: Optional few-shot examples

        Returns:
            Formatted prompt string
        """
        prompt = f"""Generate JSON card specifications for these Q&A pairs:

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

        prompt += f"\n## Q&A Pairs to Convert ({len(qa_pairs)} pairs)\n"

        for idx, qa in enumerate(qa_pairs, 1):
            # Include both English and Russian if available
            q_en = qa.question_en[:200] if qa.question_en else ""
            a_en = qa.answer_en[:300] if qa.answer_en else ""
            q_ru = qa.question_ru[:200] if qa.question_ru else ""
            a_ru = qa.answer_ru[:300] if qa.answer_ru else ""

            prompt += f"\n### Pair {idx} (card_index: {qa.card_index})\n"
            if q_en:
                prompt += f"Q (EN): {q_en}\n"
                prompt += f"A (EN): {a_en}\n"
            if q_ru:
                prompt += f"Q (RU): {q_ru}\n"
                prompt += f"A (RU): {a_ru}\n"

        prompt += f"""
## Instructions
- Generate cards for ALL Q&A pairs above
- Create separate cards for each language (EN and RU if both present)
- Use slug format: "{slug_base}-{{card_index}}-{{lang}}"
- Return valid JSON matching the CardGenerationSpec schema
"""

        return prompt

    def _convert_specs_to_cards(
        self,
        specs: list[CardSpec],
        qa_pairs: list[QAPair],
        metadata: NoteMetadata,
        slug_base: str,
        overall_confidence: float,
    ) -> list[GeneratedCard]:
        """Convert JSON CardSpecs to GeneratedCard with APF HTML.

        Args:
            specs: List of CardSpec from LLM
            qa_pairs: Original Q/A pairs for content hash
            metadata: Note metadata
            slug_base: Base slug for identifiers
            overall_confidence: Overall generation confidence

        Returns:
            List of GeneratedCard with APF HTML content
        """
        generated_cards: list[GeneratedCard] = []
        qa_lookup = {qa.card_index: qa for qa in qa_pairs}

        for spec in specs:
            try:
                # Fill in manifest data from our known values
                spec.slug_base = slug_base
                if not spec.source_path and metadata.file_path:
                    spec.source_path = str(metadata.file_path)

                # Render JSON spec to APF HTML
                apf_html = self.renderer.render(spec)

                # Validate APF has all sentinels
                missing = self.validator.validate(apf_html)
                if missing:
                    logger.warning(
                        "apf_render_missing_sentinels",
                        slug=spec.slug,
                        missing=missing,
                    )
                    # This shouldn't happen with our renderer, but log it
                    continue

                # Compute content hash for change detection
                qa_pair = qa_lookup.get(spec.card_index)
                content_hash = ""
                if qa_pair is not None:
                    content_hash = compute_content_hash(qa_pair, metadata, spec.lang)

                # Create GeneratedCard
                generated_card = GeneratedCard(
                    card_index=spec.card_index,
                    slug=spec.slug,
                    lang=spec.lang,
                    apf_html=apf_html,
                    confidence=overall_confidence,
                    content_hash=content_hash,
                )
                generated_cards.append(generated_card)

                logger.debug(
                    "card_spec_converted",
                    slug=spec.slug,
                    lang=spec.lang,
                    apf_length=len(apf_html),
                )

            except Exception as e:
                logger.warning(
                    "card_spec_conversion_failed",
                    slug=getattr(spec, "slug", "unknown"),
                    error=str(e),
                )
                continue

        return generated_cards
