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
from obsidian_anki_sync.domain.services.tag_generator import CodeDetector, TagGenerator
from obsidian_anki_sync.infrastructure.html_post_processor import APFHTMLPostProcessor
from obsidian_anki_sync.infrastructure.llm.card_generator import LLMCardGenerator
from obsidian_anki_sync.infrastructure.manifest.manifest_generator import (
    ManifestGenerator,
)
from obsidian_anki_sync.models import NoteMetadata, QAPair
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
