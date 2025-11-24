"""Quality check utilities with customizable prompts."""

import json
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


def run_quality_check(
    card_fields: dict[str, str],
    quality_config: dict[str, Any],
    llm_client: Any,
    field_to_check: str | None = None,
) -> dict[str, Any]:
    """
    Run quality check on a card using customizable prompt.

    Args:
        card_fields: Dictionary of card fields
        quality_config: Quality check configuration from template
        llm_client: LLM client instance
        field_to_check: Specific field to check (from config if not provided)

    Returns:
        Dictionary with 'is_valid' (bool), 'reason' (str), and 'score' (float, 0-1)
    """
    field = field_to_check or quality_config.get("field", "Front")
    model = quality_config.get("model")  # Can use different model for checks
    prompt_template = quality_config.get("prompt", "")

    if not prompt_template:
        # Default quality check prompt
        prompt_template = """You are an expert native speaker. Evaluate if the following text sounds natural and well-written in its language.

Text: {text}

Consider grammar, syntax, word choice, and common phrasing.

Respond with JSON only, with no additional text or explanations outside the JSON structure.
Your response must be a JSON object with two keys:
- "is_valid": a boolean (true if the text passes your criteria, false otherwise).
- "reason": a brief, one-sentence explanation for your decision.
- "score": a float between 0.0 and 1.0 indicating quality (1.0 = perfect, 0.0 = poor).
"""

    # Substitute field value
    text_to_check = card_fields.get(field, "")
    prompt = prompt_template.replace("{text}", text_to_check)

    # Use specified model or default
    check_model = model or "gpt-4o-mini"

    try:
        if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            # OpenAI-style client
            response = llm_client.chat.completions.create(
                model=check_model,
                temperature=0.0,  # Deterministic for quality checks
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            result_text = response.choices[0].message.content or "{}"
        else:
            # Generic interface
            result_text = llm_client.generate(
                prompt, model=check_model, temperature=0.0)

        # Parse JSON response
        result = json.loads(result_text)
        if not isinstance(result, dict):
            raise ValueError("Quality check response is not a JSON object")

        return {
            "is_valid": result.get("is_valid", True),
            "reason": result.get("reason", "No reason provided"),
            "score": float(result.get("score", 1.0)),
        }

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("quality_check_parse_error", error=str(e))
        # Conservative fallback: assume valid
        return {
            "is_valid": True,
            "reason": f"Quality check failed to parse: {e}",
            "score": 0.5,
        }
    except Exception as e:
        logger.error("quality_check_failed", error=str(e))
        # Conservative fallback
        return {
            "is_valid": True,
            "reason": f"Quality check error: {e}",
            "score": 0.5,
        }
