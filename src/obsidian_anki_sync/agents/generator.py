"""Generator agent for APF card generation using local LLM.

This agent generates APF cards from Q/A pairs using Ollama with powerful models
like qwen3:32b for high-quality card generation.

This is now a thin orchestrator that delegates to Clean Architecture services.
"""

from obsidian_anki_sync.agents.models import GenerationResult
from obsidian_anki_sync.application.services.card_generator import CardGeneratorService
from obsidian_anki_sync.application.services.card_structure_parser import (
    CardStructureParser,
)
from obsidian_anki_sync.domain.interfaces.card_generation import ParsedCardStructure
from obsidian_anki_sync.domain.services.tag_generator import CodeDetector, TagGenerator
from obsidian_anki_sync.infrastructure.html_post_processor import APFHTMLPostProcessor
from obsidian_anki_sync.infrastructure.llm.card_generator import LLMCardGenerator
from obsidian_anki_sync.infrastructure.manifest.manifest_generator import (
    ManifestGenerator,
)
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class GeneratorAgent:
    """Agent for generating APF cards using local LLM.

    This is now a thin orchestrator that uses Clean Architecture services.
    Uses powerful model (qwen3:32b) for high-quality card generation.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:32b",
        temperature: float = 0.3,
        slow_request_threshold: float = 60.0,
    ):
        """Initialize generator agent.

        Args:
            ollama_client: LLM provider instance (BaseLLMProvider)
            model: Model to use for generation
            temperature: Sampling temperature
        """
        # Initialize domain services
        tag_generator = TagGenerator()
        code_detector = CodeDetector()

        # Initialize infrastructure services
        html_post_processor = APFHTMLPostProcessor(tag_generator)
        manifest_generator = ManifestGenerator()
        llm_card_generator = LLMCardGenerator(
            ollama_client=ollama_client,
            model=model,
            temperature=temperature,
            slow_request_threshold=slow_request_threshold,
            tag_generator=tag_generator,
            code_detector=code_detector,
            html_post_processor=html_post_processor,
            manifest_generator=manifest_generator,
        )

        # Initialize application services
        card_structure_parser = CardStructureParser()
        self.card_generator_service = CardGeneratorService(
            single_card_generator=llm_card_generator,
            translated_card_generator=llm_card_generator,
            card_structure_parser=card_structure_parser,
            manifest_generator=manifest_generator,
        )

        self.model = model
        self.ollama_client = ollama_client
        self.tag_generator = tag_generator
        self.code_detector = code_detector
        self.manifest_generator = manifest_generator
        self.html_post_processor = html_post_processor
        self.card_structure_parser = card_structure_parser
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
        result = self.card_generator_service.generate_cards(
            note_content=note_content,
            metadata=metadata,
            qa_pairs=qa_pairs,
            slug_base=slug_base,
        )

        # Override model info since the service doesn't know it
        result.model_used = self.model

        return result

    # ------------------------------------------------------------------
    # Legacy helper methods retained for backward-compatible tests
    # ------------------------------------------------------------------
    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code snippet."""
        return self.code_detector.detect_code_language(code)

    def _generate_tags(self, metadata: NoteMetadata, lang: str) -> list[str]:
        """Generate deterministic tags for a given language."""
        return self.tag_generator.generate_tags(metadata, lang)

    def _generate_manifest(
        self, manifest: Manifest, card_type: str, tags: list[str]
    ) -> str:
        """Generate manifest comment string."""
        return self.manifest_generator.generate_manifest(manifest, card_type, tags)

    def _post_process_apf(
        self, apf_html: str, metadata: NoteMetadata, manifest: Manifest
    ) -> str:
        """Post-process APF HTML to ensure compliance."""
        return self.html_post_processor.post_process_apf(
            apf_html=apf_html, metadata=metadata, manifest=manifest
        )

    def _parse_card_structure(self, apf_html: str) -> ParsedCardStructure:
        """Parse APF HTML into structured components."""
        return self.card_structure_parser.parse_card_structure(apf_html)

    def _build_translation_prompt(
        self,
        question: str,
        answer: str,
        english_structure: ParsedCardStructure,
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> str:
        """Build translation prompt preserving structure."""
        import json

        structure_json = json.dumps(
            {
                "title": english_structure.title,
                "key_point_notes": english_structure.key_point_notes,
                "other_notes": english_structure.other_notes,
                "key_point_code": english_structure.key_point_code,
            },
            ensure_ascii=False,
        )
        tags = self._generate_tags(metadata, lang)
        manifest_comment = self._generate_manifest(manifest, "Simple", tags)
        return (
            "Translate the TEXT CONTENT to the target language.\n"
            "PRESERVE the exact same card structure.\n"
            "English Card Structure:\n"
            f"{structure_json}\n"
            "TRANSLATE ONLY the text content.\n"
            f"{manifest_comment}\n"
            f"Question: {question}\nAnswer: {answer}\n"
        )

    def _assemble_translated_card_html(
        self,
        english_structure: ParsedCardStructure,
        translated_html: str,
        metadata: NoteMetadata,
        manifest: Manifest,
    ) -> str:
        """Assemble translated APF HTML using translated content and English code."""
        import json

        try:
            translated = json.loads(translated_html)
        except Exception:
            translated = {}

        title = translated.get("title") or english_structure.title
        key_point_notes = translated.get("key_point_notes") or english_structure.key_point_notes
        other_notes = translated.get("other_notes") or english_structure.other_notes

        tags = self._generate_tags(metadata, manifest.lang)
        manifest_comment = self._generate_manifest(manifest, "Simple", tags)

        key_point_notes_html = "\n".join(f"<li>{note}</li>" for note in key_point_notes)
        other_notes_html = "\n".join(f"<li>{note}</li>" for note in other_notes)

        card_html = f"""<!-- Card 1 | slug: {manifest.slug} | CardType: Simple | Tags: {' '.join(tags)} -->
<!-- Title -->
{title}

<!-- Key point (code block / image) -->
{english_structure.key_point_code or ''}

<!-- Key point notes -->
<ul>
{key_point_notes_html}
</ul>

<!-- Other notes -->
<ul>
{other_notes_html}
</ul>

{manifest_comment}
"""
        wrapped = "\n".join(  # noqa: FLY002
            [
                "<!-- PROMPT_VERSION: apf-v2.1 -->",
                "<!-- BEGIN_CARDS -->",
                card_html.strip(),
                "<!-- END_CARDS -->",
                "END_OF_CARDS",
            ]
        )
        return wrapped


__all__ = ["GeneratorAgent", "ParsedCardStructure"]
