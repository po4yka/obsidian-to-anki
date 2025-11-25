"""Semantic validation for APF cards using LLM."""

from ...models import NoteMetadata
from ...providers.base import BaseLLMProvider
from ...utils.logging import get_logger
from ..json_schemas import get_post_validation_schema
from ..models import GeneratedCard, PostValidationResult
from .prompts import SEMANTIC_VALIDATION_SYSTEM_PROMPT, build_semantic_prompt

logger = get_logger(__name__)


def semantic_validation(
    cards: list[GeneratedCard],
    metadata: NoteMetadata,
    strict_mode: bool,
    ollama_client: BaseLLMProvider,
    model: str,
    temperature: float,
) -> PostValidationResult:
    """Perform semantic validation using LLM.

    Args:
        cards: Generated cards
        metadata: Note metadata for context
        strict_mode: Enable strict validation
        ollama_client: LLM provider instance
        model: Model to use for validation
        temperature: Sampling temperature

    Returns:
        PostValidationResult
    """
    # Build validation prompt
    prompt = build_semantic_prompt(cards, metadata, strict_mode)

    # Get JSON schema for structured output
    json_schema = get_post_validation_schema()

    # Call LLM
    result = ollama_client.generate_json(
        model=model,
        prompt=prompt,
        system=SEMANTIC_VALIDATION_SYSTEM_PROMPT,
        temperature=temperature,
        json_schema=json_schema,
    )

    # Parse LLM response
    is_valid = result.get("is_valid", False)
    error_type = result.get("error_type", "none")
    error_details = result.get("error_details", "")
    corrected_cards_data = result.get("corrected_cards")

    # Convert corrected cards if provided
    corrected_cards = None
    if corrected_cards_data:
        try:
            corrected_cards = [
                GeneratedCard(**card_data) for card_data in corrected_cards_data
            ]
        except Exception as e:
            logger.warning("failed_to_parse_corrected_cards", error=str(e))

    return PostValidationResult(
        is_valid=is_valid,
        error_type=error_type,
        error_details=error_details,
        corrected_cards=corrected_cards,
        validation_time=0.0,  # Will be set by caller
    )
