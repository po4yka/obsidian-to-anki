"""Context enrichment agent using PydanticAI.

Enhances cards with examples, mnemonics, and helpful context.
"""

import re
import time

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from ...models import NoteMetadata
from ...utils.logging import get_logger
from ..context_enrichment_prompts import CONTEXT_ENRICHMENT_PROMPT
from ..models import ContextEnrichmentResult, EnrichmentAddition, GeneratedCard
from .models import ContextEnrichmentDeps, ContextEnrichmentOutput

logger = get_logger(__name__)


class ContextEnrichmentAgentAI:
    """PydanticAI-based context enrichment agent.

    Enhances cards with examples, mnemonics, and helpful context.
    """

    def __init__(self, model: OpenAIChatModel, temperature: float = 0.3):
        """Initialize context enrichment agent.

        Args:
            model: PydanticAI model instance
            temperature: Sampling temperature (0.3 for some creativity)
        """
        self.model = model
        self.temperature = temperature

        # Create PydanticAI agent
        self.agent: Agent[ContextEnrichmentDeps, ContextEnrichmentOutput] = Agent(
            model=self.model,
            output_type=ContextEnrichmentOutput,
            system_prompt=CONTEXT_ENRICHMENT_PROMPT,
        )

        logger.info(
            "pydantic_ai_context_enrichment_agent_initialized", model=str(model)
        )

    async def enrich(
        self, card: GeneratedCard, metadata: NoteMetadata
    ) -> ContextEnrichmentResult:
        """Enrich a card with additional context and examples.

        Args:
            card: Card to enrich
            metadata: Note metadata for context

        Returns:
            ContextEnrichmentResult with enriched card
        """
        start_time = time.time()

        try:
            # Extract Q/A from APF HTML
            question, answer = self._extract_qa_from_apf(card.apf_html)
            extra = self._extract_extra_from_apf(card.apf_html)

            # Create dependencies
            deps = ContextEnrichmentDeps(
                question=question,
                answer=answer,
                extra=extra or "",
                card_slug=card.slug,
                note_title=metadata.title,
            )

            # Build prompt
            prompt = f"""Analyze this flashcard and determine if it needs enrichment:

**Card (slug: {card.slug})**
Note: {metadata.title}

Q: {question}
A: {answer}
Extra: {extra or "(none)"}

Consider adding:
- Concrete examples (code, real-world scenarios)
- Mnemonics or memory aids
- Visual structure (formatting, bullets)
- Related concepts or comparisons
- Practical tips or common pitfalls

Provide your enrichment assessment."""

            logger.info("pydantic_ai_enrichment_start", slug=card.slug)

            # Run agent
            result = await self.agent.run(prompt, deps=deps)
            output: ContextEnrichmentOutput = result.data

            # If enrichment recommended, create enriched card
            enriched_card = None
            additions = []

            if output.should_enrich:
                # Reconstruct APF HTML with enrichments
                enriched_apf = self._rebuild_apf_html(
                    card.apf_html,
                    output.enriched_answer or answer,
                    output.enriched_extra or extra,
                )

                enriched_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=enriched_apf,
                    confidence=card.confidence,
                    content_hash=card.content_hash,
                )

                # Create additions list
                for enrich_type in output.enrichment_type:
                    addition = EnrichmentAddition(
                        enrichment_type=enrich_type,  # type: ignore[arg-type]
                        content=(
                            output.enriched_extra[:200] + "..."
                            if len(output.enriched_extra) > 200
                            else output.enriched_extra
                        ),
                        rationale=output.rationale,
                    )
                    additions.append(addition)

            enrichment_result = ContextEnrichmentResult(
                should_enrich=output.should_enrich,
                enriched_card=enriched_card,
                additions=additions,
                additions_summary=output.additions_summary,
                enrichment_rationale=output.rationale,
                enrichment_time=time.time() - start_time,
            )

            logger.info(
                "pydantic_ai_enrichment_complete",
                slug=card.slug,
                should_enrich=output.should_enrich,
                types=output.enrichment_type,
                confidence=output.confidence,
            )

            return enrichment_result

        except Exception as e:
            logger.error("pydantic_ai_enrichment_failed", error=str(e), slug=card.slug)
            # Return safe fallback
            return ContextEnrichmentResult(
                should_enrich=False,
                enriched_card=None,
                additions=[],
                additions_summary=f"Enrichment failed: {str(e)}",
                enrichment_rationale="Agent encountered error",
                enrichment_time=0.0,
            )

    def _extract_qa_from_apf(self, apf_html: str) -> tuple[str, str]:
        """Extract question and answer from APF HTML."""
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

        return question or "Unknown", answer or "Unknown"

    def _extract_extra_from_apf(self, apf_html: str) -> str:
        """Extract Extra section from APF HTML."""
        extra_match = re.search(r'<div class="extra">(.*?)</div>', apf_html, re.DOTALL)
        if extra_match:
            return re.sub(r"<[^>]+>", "", extra_match.group(1)).strip()
        return ""

    def _rebuild_apf_html(
        self, original_apf: str, new_answer: str, new_extra: str
    ) -> str:
        """Rebuild APF HTML with enriched content."""
        # Replace answer
        apf_html = re.sub(
            r'(<div class="back">)(.*?)(</div>)',
            rf"\1{new_answer}\3",
            original_apf,
            flags=re.DOTALL,
        )

        # Replace or add extra
        if '<div class="extra">' in apf_html:
            apf_html = re.sub(
                r'(<div class="extra">)(.*?)(</div>)',
                rf"\1{new_extra}\3",
                apf_html,
                flags=re.DOTALL,
            )
        else:
            # Add extra before closing tag
            apf_html = apf_html.replace(
                "</div>\n</div>",
                f'</div>\n<div class="extra">{new_extra}</div>\n</div>',
            )

        return apf_html
