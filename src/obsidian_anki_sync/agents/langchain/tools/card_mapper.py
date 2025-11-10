"""Card Mapping Agent - Converts NoteContext to ProposedCard.

This LLM-powered agent understands note structure and maps it to
appropriate Anki card fields with correct type and deck selection.
"""

import json
from typing import Any, Optional

from obsidian_anki_sync.agents.langchain.models import (
    BilingualMode,
    CardType,
    Language,
    NoteContext,
    NoteContextOrigin,
    ProposedCard,
)
from obsidian_anki_sync.utils.logging import get_logger

try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = get_logger(__name__)


CARD_MAPPING_SYSTEM_PROMPT = """You are an expert at converting knowledge notes into high-quality Anki flashcards.

Your task is to analyze an Obsidian note and generate a properly structured Anki card that:
1. Maps the note content to appropriate card fields (Front, Back, Extra, Hint)
2. Selects the correct card type (Basic, Cloze, or Custom)
3. Chooses the appropriate Anki model and deck
4. Handles bilingual content correctly
5. Generates relevant tags

## Card Type Selection Rules:

**Basic Card**: Use when there's a clear question/answer pair
- Front: The question or prompt
- Back: The answer or explanation
- Extra: Additional context, examples, or references (optional)
- Hint: A subtle hint without revealing the answer (optional)

**Cloze Card**: Use when the answer involves filling in missing information
- Text: The text with {{{{c1::deletion}}}} markers
- Extra: Additional context or notes
- Format: Replace key terms with {{{{c1::term}}}}, {{{{c2::term}}}}, etc.

**Custom Card**: Only use if explicitly specified in card_type_hint

## Model Selection:
- "APF::Simple" or "APF: Simple (3.0.0)": For basic Q&A cards
- "APF::Missing (Cloze)": For cloze deletion cards
- "InterviewBasic": For interview-style basic cards
- "InterviewCloze": For interview-style cloze cards

## Deck Name Resolution:
- Use deck_hint from frontmatter if provided
- Otherwise construct from topic: "Interview::<Topic>"
- Examples: "Interview::Android", "Interview::System Design"

## Tag Generation:
- Always include tags from frontmatter
- Add topic as a tag (lowercased, spaces replaced with dashes)
- Add difficulty level if present
- Add language tags (en, ru, or both)
- Limit to 10 most relevant tags

## Bilingual Handling:
- If both language tags present: bilingual_mode = "front_back"
  - Put primary language on Front
  - Put secondary language on Back or Extra
- If single language: bilingual_mode = "none"

## Quality Guidelines:
- Front should be concise but complete (avoid cryptic questions)
- Back should be self-contained (don't assume Front context in wording)
- Extra should add value (examples, edge cases, references)
- Hint should guide without revealing (e.g., "Think about data structure properties")

## Response Format:
Respond with a JSON object matching this schema:
{
  "card_type": "Basic" | "Cloze" | "Custom",
  "model_name": "string",
  "deck_name": "string",
  "fields": {
    "Front": "string",
    "Back": "string",
    "Extra": "string (optional)",
    "Hint": "string (optional)"
  },
  "tags": ["string"],
  "language": "en" | "ru",
  "bilingual_mode": "none" | "front_back" | "separate_cards",
  "slug": "string (copy from input)",
  "origin": {
    "note_path": "string (copy from input)",
    "source_note_lang": "en" | "ru"
  },
  "confidence": 0.0-1.0,
  "notes": "Brief explanation of your mapping decisions"
}

IMPORTANT: Only respond with the JSON object, no additional text."""


class CardMapperTool:
    """Tool for mapping NoteContext to ProposedCard using LLM.

    This agent analyzes the note structure, content, and metadata to
    generate a properly formatted Anki card.
    """

    def __init__(
        self,
        llm: "BaseChatModel",
        default_deck_prefix: str = "Interview",
        default_model_basic: str = "APF: Simple (3.0.0)",
        default_model_cloze: str = "APF::Missing (Cloze)",
        max_retries: int = 2,
    ):
        """Initialize Card Mapping Tool.

        Args:
            llm: LangChain chat model for LLM calls
            default_deck_prefix: Default prefix for deck names
            default_model_basic: Default model for Basic cards
            default_model_cloze: Default model for Cloze cards
            max_retries: Maximum retries on parsing failures
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for CardMapperTool")

        self.llm = llm
        self.default_deck_prefix = default_deck_prefix
        self.default_model_basic = default_model_basic
        self.default_model_cloze = default_model_cloze
        self.max_retries = max_retries

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CARD_MAPPING_SYSTEM_PROMPT),
                ("human", "{note_context_json}"),
            ]
        )

        logger.info(
            "card_mapper_initialized",
            llm_type=type(llm).__name__,
            default_deck_prefix=default_deck_prefix,
        )

    def map(
        self,
        note_context: NoteContext,
        feedback: Optional[str] = None,
    ) -> ProposedCard:
        """Map a NoteContext to a ProposedCard.

        Args:
            note_context: The note to map
            feedback: Optional feedback from previous validation attempts

        Returns:
            ProposedCard with mapped fields

        Raises:
            ValueError: If mapping fails after retries
        """
        logger.info(
            "card_mapping_start",
            slug=note_context.slug,
            note_path=note_context.note_path,
        )

        # Serialize note context to JSON
        note_context_json = note_context.model_dump_json(indent=2)

        # Add feedback if provided
        if feedback:
            note_context_json += f"\n\n## Previous Validation Feedback:\n{feedback}\n\nPlease address the feedback above and regenerate the card."

        # Generate with retries
        for attempt in range(self.max_retries):
            try:
                result = self._generate_card(note_context_json)

                logger.info(
                    "card_mapping_success",
                    slug=note_context.slug,
                    card_type=result.card_type,
                    model=result.model_name,
                    attempt=attempt + 1,
                )

                return result

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "card_mapping_parse_error",
                    slug=note_context.slug,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt == self.max_retries - 1:
                    raise ValueError(
                        f"Failed to map card after {self.max_retries} attempts: {e}"
                    )

        raise ValueError("Unexpected error in card mapping")

    def _generate_card(self, note_context_json: str) -> ProposedCard:
        """Generate card with LLM call.

        Args:
            note_context_json: JSON-serialized NoteContext

        Returns:
            ProposedCard parsed from LLM response
        """
        # Create messages
        messages = [
            SystemMessage(content=CARD_MAPPING_SYSTEM_PROMPT),
            HumanMessage(content=note_context_json),
        ]

        # Call LLM
        response = self.llm.invoke(messages)

        # Extract content
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        try:
            # Handle potential markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            card_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(
                "json_parse_failed",
                response=response_text[:500],
                error=str(e),
            )
            raise

        # Convert to ProposedCard
        try:
            proposed_card = ProposedCard(**card_dict)
            return proposed_card
        except Exception as e:
            logger.error(
                "pydantic_validation_failed",
                card_dict=card_dict,
                error=str(e),
            )
            raise ValueError(f"Failed to create ProposedCard: {e}")

    def map_with_fallback(
        self,
        note_context: NoteContext,
        feedback: Optional[str] = None,
    ) -> ProposedCard:
        """Map with automatic fallback to rule-based mapping on failure.

        Args:
            note_context: The note to map
            feedback: Optional feedback from validation

        Returns:
            ProposedCard (from LLM or fallback)
        """
        try:
            return self.map(note_context, feedback)
        except Exception as e:
            logger.warning(
                "card_mapping_failed_fallback",
                slug=note_context.slug,
                error=str(e),
            )
            return self._fallback_mapping(note_context)

    def _fallback_mapping(self, note_context: NoteContext) -> ProposedCard:
        """Rule-based fallback mapping when LLM fails.

        Args:
            note_context: The note to map

        Returns:
            ProposedCard with simple rule-based mapping
        """
        logger.info("using_fallback_mapping", slug=note_context.slug)

        # Determine card type
        card_type = note_context.frontmatter.card_type_hint or CardType.BASIC

        # Determine model
        if card_type == CardType.CLOZE:
            model_name = self.default_model_cloze
        else:
            model_name = self.default_model_basic

        # Determine deck
        deck_name = note_context.frontmatter.deck_hint or (
            f"{self.default_deck_prefix}::{note_context.frontmatter.topic}"
        )

        # Map fields
        fields: dict[str, str] = {
            "Front": note_context.sections.question,
            "Back": note_context.sections.answer,
        }

        if note_context.sections.extra:
            fields["Extra"] = note_context.sections.extra

        # Generate tags
        tags = list(note_context.frontmatter.tags)
        tags.append(note_context.frontmatter.topic.lower().replace(" ", "-"))
        tags.append(note_context.frontmatter.lang.value)
        tags.append(note_context.frontmatter.difficulty.value)

        # Bilingual mode
        bilingual_mode = BilingualMode.NONE

        return ProposedCard(
            card_type=card_type,
            model_name=model_name,
            deck_name=deck_name,
            fields=fields,
            tags=tags,
            language=note_context.frontmatter.lang,
            bilingual_mode=bilingual_mode,
            slug=note_context.slug,
            origin=NoteContextOrigin(
                note_path=note_context.note_path,
                source_note_lang=note_context.frontmatter.lang,
            ),
            confidence=0.5,
            notes="Generated using fallback rule-based mapping due to LLM failure",
        )
