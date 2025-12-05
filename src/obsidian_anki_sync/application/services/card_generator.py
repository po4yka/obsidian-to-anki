"""Application service for card generation orchestration.

This module provides the main card generation service that orchestrates
the generation process, following Clean Architecture principles.
"""

import time

from obsidian_anki_sync.agents.models import GeneratedCard, GenerationResult
from obsidian_anki_sync.domain.interfaces.card_generation import (
    ICardGenerator,
    ICardStructureParser,
    ISingleCardGenerator,
    ITranslatedCardGenerator,
)
from obsidian_anki_sync.infrastructure.manifest.manifest_generator import (
    ManifestGenerator,
)
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class CardGeneratorService(ICardGenerator):
    """Application service that orchestrates card generation across languages."""

    def __init__(
        self,
        single_card_generator: ISingleCardGenerator,
        translated_card_generator: ITranslatedCardGenerator,
        card_structure_parser: ICardStructureParser,
        manifest_generator: ManifestGenerator | None = None,
    ):
        """Initialize card generator service.

        Args:
            single_card_generator: Service for generating individual cards
            translated_card_generator: Service for generating translated cards
            card_structure_parser: Service for parsing card structures
            manifest_generator: Service for generating manifests
        """
        self.single_card_generator = single_card_generator
        self.translated_card_generator = translated_card_generator
        self.card_structure_parser = card_structure_parser
        self.manifest_generator = manifest_generator or ManifestGenerator()

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
        # card_index -> parsed structure
        english_structures = {}

        # Generate English cards first to establish canonical structure
        for qa_pair in qa_pairs:
            if "en" in metadata.language_tags:
                # Create manifest for English card
                manifest_en = self.create_manifest(
                    qa_pair=qa_pair,
                    metadata=metadata,
                    slug_base=slug_base,
                    lang="en",
                )

                # Generate English APF card
                card_en = self.single_card_generator.generate_single_card(
                    qa_pair=qa_pair, metadata=metadata, manifest=manifest_en, lang="en"
                )
                generated_cards.append(card_en)

                # Parse and cache the English structure for translation
                english_structures[qa_pair.card_index] = self.card_structure_parser.parse_card_structure(
                    card_en.apf_html
                )

        # Generate cards in other languages using English structure as template
        for qa_pair in qa_pairs:
            for lang in metadata.language_tags:
                if lang == "en":
                    # Already generated English above
                    continue

                # Create manifest for this language
                manifest = self.create_manifest(
                    qa_pair=qa_pair,
                    metadata=metadata,
                    slug_base=slug_base,
                    lang=lang,
                )

                # Get the canonical English structure
                english_structure = english_structures.get(qa_pair.card_index)
                if not english_structure:
                    logger.warning(
                        "no_english_structure_for_translation",
                        card_index=qa_pair.card_index,
                        lang=lang,
                        slug=manifest.slug,
                    )
                    # Fallback to regular generation if no English structure available
                    card = self.single_card_generator.generate_single_card(
                        qa_pair=qa_pair, metadata=metadata, manifest=manifest, lang=lang
                    )
                else:
                    # Generate translated card using English structure
                    card = self.translated_card_generator.generate_translated_card(
                        qa_pair=qa_pair,
                        metadata=metadata,
                        manifest=manifest,
                        english_structure=english_structure,
                        lang=lang,
                    )

                generated_cards.append(card)

        generation_time = time.time() - start_time

        # Calculate aggregate statistics for future metrics
        total_confidence = sum(card.confidence for card in generated_cards)
        avg_confidence = (
            total_confidence / len(generated_cards) if generated_cards else 0
        )
        logger.debug(
            "generation_confidence_stats",
            avg_confidence=avg_confidence,
            total=total_confidence,
        )

        logger.info(
            "card_generation_complete",
            cards_generated=len(generated_cards),
            time=generation_time,
            avg_time_per_card=(
                round(generation_time / len(generated_cards), 2)
                if generated_cards
                else 0
            ),
        )

        return GenerationResult(
            cards=generated_cards,
            total_cards=len(generated_cards),
            generation_time=generation_time,
            model_used="unknown",  # This will be overridden by the infrastructure layer
        )

    def create_manifest(
        self, qa_pair: QAPair, metadata: NoteMetadata, slug_base: str, lang: str
    ) -> "Manifest":
        """Create card manifest.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            slug_base: Base slug
            lang: Language code

        Returns:
            Manifest instance
        """
        return self.manifest_generator.create_manifest(qa_pair, metadata, slug_base, lang)
