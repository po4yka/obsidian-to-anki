"""JSON card specification schema for structured LLM output.

This module defines Pydantic models for JSON card output from the LLM.
The JSON schema is converted to APF HTML by the APFRenderer.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CardSection(BaseModel):
    """Content section of a card (front or back)."""

    title: str = Field(description="Question/title text for the card")
    key_point_code: str | None = Field(
        default=None, description="Code block content if any"
    )
    key_point_code_lang: str = Field(
        default="plaintext", description="Programming language for code block"
    )
    key_point_notes: list[str] = Field(
        default_factory=list,
        description="Bullet point notes (5-7 detailed points recommended)",
    )
    other_notes: str = Field(
        default="", description="Additional notes, references, and links"
    )
    extra: str = Field(default="", description="Extra information and context")

    @field_validator("key_point_code_lang", mode="before")
    @classmethod
    def normalize_lang(cls, v: str | None) -> str:
        """Normalize code language identifier."""
        if not v:
            return "plaintext"
        return v.lower().strip()


class CardSpec(BaseModel):
    """JSON specification for a single flashcard.

    This is the structured output from the LLM that gets converted
    to APF HTML by the APFRenderer.
    """

    card_index: int = Field(ge=1, description="1-based card index within the note")
    slug: str = Field(min_length=1, description="Unique card identifier")
    lang: str = Field(pattern=r"^(en|ru)$", description="Language code (en or ru)")
    card_type: Literal["Simple", "Missing", "Draw"] = Field(
        default="Simple", description="Card type: Simple, Missing (cloze), or Draw"
    )
    tags: list[str] = Field(
        default_factory=list, description="3-6 snake_case tags for the card"
    )

    # Card content
    front: CardSection = Field(description="Front/primary content section")
    back: CardSection | None = Field(
        default=None, description="Back content section (if different from front)"
    )

    # Manifest data (filled in by caller, not LLM)
    source_path: str = Field(default="", description="Source file path")
    source_anchor: str = Field(default="", description="Anchor in source file")
    guid: str = Field(default="", description="Anki GUID for the card")
    slug_base: str = Field(default="", description="Base slug without index/lang")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: list | None) -> list[str]:
        """Normalize tags to snake_case."""
        if not v:
            return []
        normalized = []
        for tag in v:
            if isinstance(tag, str):
                # Convert to snake_case
                tag = tag.strip().lower().replace(" ", "_").replace("-", "_")
                if tag:
                    normalized.append(tag)
        return normalized

    @field_validator("card_type", mode="before")
    @classmethod
    def normalize_card_type(cls, v: str | None) -> str:
        """Normalize card type to valid values."""
        if not v:
            return "Simple"
        v_lower = v.lower().strip()
        if v_lower in ("missing", "cloze", "missing (cloze)"):
            return "Missing"
        if v_lower == "draw":
            return "Draw"
        return "Simple"


class CardGenerationSpec(BaseModel):
    """Complete JSON output from card generation LLM.

    Contains all generated cards for a single note.
    """

    cards: list[CardSpec] = Field(
        default_factory=list, description="All generated cards"
    )
    generation_notes: str = Field(
        default="", description="Notes about the generation process"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Overall generation confidence"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float | None) -> float:
        """Clamp confidence to valid range."""
        if v is None:
            return 0.8
        return max(0.0, min(1.0, float(v)))
