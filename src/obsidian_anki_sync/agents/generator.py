"""Generator agent for APF card generation using local LLM.

This agent generates APF cards from Q/A pairs using Ollama with powerful models
like qwen3:32b for high-quality card generation.
"""

import json
import time
from pathlib import Path

from ..apf.generator import APFGenerator
from ..models import Manifest, NoteMetadata, QAPair
from ..utils.logging import get_logger
from .models import GeneratedCard, GenerationResult
from .ollama_client import OllamaClient

logger = get_logger(__name__)


class GeneratorAgent:
    """Agent for generating APF cards using local LLM.

    Uses powerful model (qwen3:32b) for high-quality card generation.
    Reuses existing APF generation logic from APFGenerator.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str = "qwen3:32b",
        temperature: float = 0.3,
    ):
        """Initialize generator agent.

        Args:
            ollama_client: Ollama client instance
            model: Model to use for generation
            temperature: Sampling temperature
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature

        # Load system prompt from CARDS_PROMPT.md
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = "Generate APF cards following strict APF v2.1 format."

        logger.info("generator_agent_initialized", model=model)

    def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> GenerationResult:
        """Generate APF cards for all Q/A pairs in a note.

        Args:
            note_content: Full note content (for context)
            metadata: Note metadata
            qa_pairs: List of Q/A pairs to convert to cards
            slug_base: Base slug for card generation

        Returns:
            GenerationResult with all generated cards
        """
        start_time = time.time()

        logger.info(
            "card_generation_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
            languages=metadata.language_tags,
        )

        generated_cards: list[GeneratedCard] = []

        # Generate cards for each Q/A pair in each language
        for qa_pair in qa_pairs:
            for lang in metadata.language_tags:
                # Create manifest for this card
                manifest = self._create_manifest(
                    qa_pair=qa_pair,
                    metadata=metadata,
                    slug_base=slug_base,
                    lang=lang,
                )

                # Generate APF card
                card = self._generate_single_card(
                    qa_pair=qa_pair, metadata=metadata, manifest=manifest, lang=lang
                )

                generated_cards.append(card)

        generation_time = time.time() - start_time

        logger.info(
            "card_generation_complete",
            cards_generated=len(generated_cards),
            time=generation_time,
        )

        return GenerationResult(
            cards=generated_cards,
            total_cards=len(generated_cards),
            generation_time=generation_time,
            model_used=self.model,
        )

    def _create_manifest(
        self, qa_pair: QAPair, metadata: NoteMetadata, slug_base: str, lang: str
    ) -> Manifest:
        """Create card manifest.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            slug_base: Base slug
            lang: Language code

        Returns:
            Manifest instance
        """
        # Generate slug
        slug = f"{slug_base}-{qa_pair.card_index}-{lang}"

        # Generate GUID (deterministic based on slug)
        guid = str(hash(slug))

        return Manifest(
            slug=slug,
            slug_base=slug_base,
            lang=lang,
            source_path=metadata.title,  # Simplified for agent system
            source_anchor=f"qa-{qa_pair.card_index}",
            note_id=metadata.id,
            note_title=metadata.title,
            card_index=qa_pair.card_index,
            guid=guid,
            hash6=None,
        )

    def _generate_single_card(
        self, qa_pair: QAPair, metadata: NoteMetadata, manifest: Manifest, lang: str
    ) -> GeneratedCard:
        """Generate a single APF card.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            manifest: Card manifest
            lang: Language code

        Returns:
            GeneratedCard instance
        """
        # Select language-specific content
        question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

        # Build user prompt (reuse logic from APFGenerator)
        user_prompt = self._build_user_prompt(
            question=question,
            answer=answer,
            qa_pair=qa_pair,
            metadata=metadata,
            manifest=manifest,
            lang=lang,
        )

        logger.debug(
            "calling_llm",
            model=self.model,
            slug=manifest.slug,
            prompt_length=len(user_prompt),
        )

        try:
            # Call Ollama LLM
            result = self.ollama_client.generate(
                model=self.model,
                prompt=user_prompt,
                system=self.system_prompt,
                temperature=self.temperature,
            )

            apf_html = result.get("response", "")

            if not apf_html:
                raise ValueError("LLM returned empty response")

            logger.debug(
                "llm_response_received", slug=manifest.slug, length=len(apf_html)
            )

            # Post-process APF HTML (normalize code blocks, ensure manifest)
            apf_html = self._post_process_apf(apf_html, metadata, manifest)

            # Extract confidence from LLM response (if available)
            # For now, use a default confidence
            confidence = 0.9

            return GeneratedCard(
                card_index=qa_pair.card_index,
                slug=manifest.slug,
                lang=lang,
                apf_html=apf_html,
                confidence=confidence,
            )

        except Exception as e:
            logger.error("card_generation_failed", slug=manifest.slug, error=str(e))
            raise

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> str:
        """Build user prompt for LLM.

        Reuses logic from APFGenerator._build_user_prompt.
        """
        ref_link = f"[[{manifest.source_path}#{manifest.source_anchor}]]"

        prompt = f"""Generate an APF card in HTML format.

Metadata:
- Topic: {metadata.topic}
- Subtopics: {", ".join(metadata.subtopics) if metadata.subtopics else "None"}
- Difficulty: {metadata.difficulty or "Not specified"}
- Language: {lang}
- Slug: {manifest.slug}

Question:
{question}

Answer:
{answer}
"""

        # Add optional context
        if qa_pair.followups:
            prompt += f"\nFollow-ups:\n{qa_pair.followups}\n"

        if qa_pair.references:
            prompt += f"\nReferences:\n{qa_pair.references}\n"

        if qa_pair.related:
            prompt += f"\nRelated Questions:\n{qa_pair.related}\n"

        if qa_pair.context:
            prompt += f"\nAdditional Context:\n{qa_pair.context}\n"

        # Derive code language hint
        code_lang = self._code_language_hint(metadata)

        # Add requirements
        prompt += f"""
Requirements:
- CardType: Simple (or Missing if {{{{c}}}} detected in answer, or Draw if diagram marker present)
- Tags: Derive from topic/subtopics, 3-6 snake_case tags, include primary language/tech
- Primary language tag: {lang}
- Topic-based tags: {metadata.topic.lower().replace(" ", "_")}
- Preserve every code block using <pre><code class="language-{code_lang}"> ... </code></pre> with original indentation
- Include manifest at end with slug "{manifest.slug}"
- Add "Ref: {ref_link}" in Other notes section
- Follow APF v2.1 format strictly
- Output ONLY the card HTML, no explanations
"""

        return prompt

    def _code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks.

        Reuses logic from APFGenerator._code_language_hint.
        """
        candidates = list(metadata.tags) + metadata.subtopics + [metadata.topic]
        known_languages = {
            "kotlin",
            "java",
            "python",
            "swift",
            "cpp",
            "c",
            "csharp",
            "go",
            "rust",
            "javascript",
            "typescript",
            "sql",
            "bash",
            "shell",
            "yaml",
            "json",
            "html",
            "css",
            "gradle",
            "groovy",
        }

        for raw in candidates:
            if not raw:
                continue
            normalized = (
                raw.lower()
                .replace("language/", "")
                .replace("lang/", "")
                .replace("topic/", "")
                .replace("/", "_")
            )
            if normalized in known_languages:
                return normalized

        return "plaintext"

    def _post_process_apf(
        self, apf_html: str, metadata: NoteMetadata, manifest: Manifest
    ) -> str:
        """Post-process APF HTML to ensure correctness.

        Args:
            apf_html: Raw APF HTML from LLM
            metadata: Note metadata
            manifest: Card manifest

        Returns:
            Post-processed APF HTML
        """
        # For now, return as-is
        # In full implementation, this would:
        # 1. Normalize code blocks (convert Markdown fences to <pre><code>)
        # 2. Ensure manifest comment is present and correct
        # 3. Validate HTML structure

        # Basic manifest check
        if "<!-- manifest:" not in apf_html:
            # Add manifest if missing
            manifest_json = json.dumps(
                {
                    "slug": manifest.slug,
                    "slug_base": manifest.slug_base,
                    "lang": manifest.lang,
                    "guid": manifest.guid,
                    "type": "Simple",
                    "tags": [],
                },
                ensure_ascii=False,
            )
            apf_html += f"\n<!-- manifest: {manifest_json} -->"

        return apf_html
