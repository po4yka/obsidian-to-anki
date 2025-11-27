"""Mock implementation of ICardGenerator for testing."""

from typing import Dict, List, Optional

from obsidian_anki_sync.domain.entities.card import Card, CardManifest
from obsidian_anki_sync.domain.entities.note import Note, QAPair
from obsidian_anki_sync.domain.interfaces.card_generator import ICardGenerator


class MockCardGenerator(ICardGenerator):
    """Mock implementation of card generator for testing.

    Provides controllable card generation for testing sync operations
    without requiring real LLM or APF generation.
    """

    def __init__(self):
        """Initialize mock generator."""
        self.generated_cards = []  # List to track generated cards
        self.should_fail = False
        self.fail_message = "Mock card generator failure"
        self.card_templates = {}  # note_id -> card template

    def generate_cards_from_note(self, note: Note) -> List[Card]:
        """Generate cards from note."""
        if self.should_fail:
            raise Exception(self.fail_message)

        # Check if we have a template for this note
        if note.id in self.card_templates:
            template = self.card_templates[note.id]
            cards = []
            for i, card_data in enumerate(template):
                card = self._create_mock_card(note, i, card_data)
                cards.append(card)
                self.generated_cards.append(card)
            return cards

        # Default mock generation - create one card per language
        cards = []
        for i, lang in enumerate(note.metadata.language_tags):
            card_data = {
                "question": f"Mock question for {note.title} ({lang})",
                "answer": f"Mock answer for {note.title} ({lang})",
                "slug_suffix": f"mock-{i}",
            }
            card = self._create_mock_card(note, i, card_data)
            cards.append(card)
            self.generated_cards.append(card)

        return cards

    def generate_card_from_qa_pair(
        self,
        qa_pair: QAPair,
        note: Note,
        language: str
    ) -> Card:
        """Generate card from Q&A pair."""
        if self.should_fail:
            raise Exception(self.fail_message)

        card_data = {
            "question": qa_pair.question_en if language == "en" else qa_pair.question_ru,
            "answer": qa_pair.answer_en if language == "en" else qa_pair.answer_ru,
            "slug_suffix": f"qa-{qa_pair.card_index}",
        }

        card = self._create_mock_card(
            note, qa_pair.card_index, card_data, language)
        self.generated_cards.append(card)
        return card

    def generate_apf_html(
        self,
        qa_pair: QAPair,
        language: str,
        note_title: str,
        card_index: int
    ) -> str:
        """Generate APF HTML."""
        if self.should_fail:
            raise Exception(self.fail_message)

        question = qa_pair.question_en if language == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if language == "en" else qa_pair.answer_ru

        return f"""<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
# Question (EN)
{question}

# Answer (EN)
{answer}
<!-- END_CARDS -->"""

    def validate_card_content(self, card: Card) -> List[str]:
        """Validate card content."""
        errors = []

        if not card.apf_html:
            errors.append("Missing APF HTML content")

        if not card.slug:
            errors.append("Missing card slug")

        if not card.manifest:
            errors.append("Missing card manifest")

        return errors

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["en", "ru", "es", "fr"]

    def get_note_type_for_card(self, card: Card) -> str:
        """Get note type for card."""
        return "APF::Simple"

    def create_manifest(
        self,
        note: Note,
        card_index: int,
        language: str,
        slug: str,
        slug_base: str
    ) -> dict:
        """Create manifest."""
        return {
            "slug": slug,
            "slug_base": slug_base,
            "lang": language,
            "source_path": str(note.file_path),
            "source_anchor": f"p{card_index}",
            "note_id": note.id,
            "note_title": note.title,
            "card_index": card_index,
            "guid": None,
            "hash6": None,
        }

    def _create_mock_card(
        self,
        note: Note,
        card_index: int,
        card_data: Dict,
        language: str = "en"
    ) -> Card:
        """Create a mock card."""
        slug_base = f"{note.id}-{card_index}"
        slug = f"{slug_base}-{language}"

        manifest = CardManifest(
            slug=slug,
            slug_base=slug_base,
            lang=language,
            source_path=str(note.file_path),
            source_anchor=f"p{card_index}",
            note_id=note.id,
            note_title=note.title,
            card_index=card_index,
        )

        apf_html = self.generate_apf_html(
            QAPair(
                card_index=card_index,
                question_en=card_data["question"],
                question_ru=card_data.get(
                    "question_ru", card_data["question"]),
                answer_en=card_data["answer"],
                answer_ru=card_data.get("answer_ru", card_data["answer"]),
            ),
            language,
            note.title,
            card_index
        )

        return Card(
            slug=slug,
            language=language,
            apf_html=apf_html,
            manifest=manifest,
            note_type="APF::Simple",
            tags=["mock"],
        )

    # Test helper methods

    def set_card_template(self, note_id: str, cards_data: List[Dict]) -> None:
        """Set card template for a note."""
        self.card_templates[note_id] = cards_data

    def set_failure(self, message: str = "Mock card generator failure") -> None:
        """Make generator fail on next calls."""
        self.should_fail = True
        self.fail_message = message

    def clear_failure(self) -> None:
        """Clear failure state."""
        self.should_fail = False

    def get_generated_cards(self) -> List[Card]:
        """Get list of generated cards."""
        return self.generated_cards.copy()

    def get_generation_count(self) -> int:
        """Get number of cards generated."""
        return len(self.generated_cards)

    def reset(self) -> None:
        """Reset mock state."""
        self.generated_cards.clear()
        self.card_templates.clear()
        self.should_fail = False
        self.fail_message = "Mock card generator failure"
