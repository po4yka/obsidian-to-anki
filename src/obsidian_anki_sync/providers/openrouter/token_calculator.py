"""Token calculation utilities for OpenRouter requests."""

import json
from typing import Any

from .models import (
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_CONTEXT_WINDOWS,
    MODEL_MAX_OUTPUT_TOKENS,
)


def calculate_prompt_tokens_estimate(prompt: str, system: str) -> int:
    """Estimate the number of tokens in the prompt.

    Args:
        prompt: User prompt text
        system: System prompt text

    Returns:
        Estimated token count (rough estimate: 1 token ≈ 4 chars)
    """
    return (len(prompt) + len(system)) // 4


def calculate_schema_overhead(json_schema: dict[str, Any] | None) -> int:
    """Calculate the token overhead from JSON schema.

    Args:
        json_schema: JSON schema dictionary

    Returns:
        Estimated schema token overhead
    """
    if not json_schema:
        return 0
    return len(json.dumps(json_schema.get("schema", {}))) // 4


def get_model_context_window(model: str) -> int:
    """Get the context window size for a model.

    Args:
        model: Model identifier

    Returns:
        Context window size in tokens
    """
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


def get_model_max_output(model: str) -> int:
    """Get the maximum output token limit for a model.

    Args:
        model: Model identifier

    Returns:
        Maximum output tokens
    """
    return MODEL_MAX_OUTPUT_TOKENS.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


def calculate_effective_max_tokens(
    model: str,
    prompt_tokens_estimate: int,
    schema_overhead: int,
    json_schema: dict[str, Any] | None,
    configured_max_tokens: int,
) -> int:
    """Calculate the effective max_tokens for a request.

    Args:
        model: Model identifier
        prompt_tokens_estimate: Estimated prompt token count
        schema_overhead: Schema token overhead
        json_schema: JSON schema dictionary (if any)
        configured_max_tokens: Configured max_tokens value

    Returns:
        Effective max_tokens to use
    """
    context_window = get_model_context_window(model)
    model_max_output = get_model_max_output(model)

    if json_schema:
        # For structured outputs, be more generous with tokens
        # Use 4-5x multiplier for complex bilingual content with structured output
        # This accounts for:
        # - Bilingual responses (2x)
        # - Structured JSON overhead (1x)
        # - Code examples and formatting (1x)
        # - Safety margin (0.5-1x)
        multiplier = 4.5 if prompt_tokens_estimate > 3000 else 4.0
        estimated_needed = int(prompt_tokens_estimate * multiplier) + schema_overhead

        # For structured outputs, ensure minimum floor to prevent truncation
        # Complex schemas (like QA extraction) need more tokens
        schema_name = json_schema.get("name", "")
        # Set reasonable minimums that respect model output limits
        # These will be capped by model_max_output anyway
        if (
            "qa_extraction" in schema_name.lower()
            or "extraction" in schema_name.lower()
        ):
            min_tokens_for_schema = 4096  # QA extraction needs reasonable tokens
        elif "validation" in schema_name.lower():
            min_tokens_for_schema = 2048  # Validation schemas are simpler
        else:
            min_tokens_for_schema = 3072  # Default for other structured outputs

        # Use the larger of: configured max, estimated needed, or minimum floor
        desired_max_tokens = max(
            configured_max_tokens, estimated_needed, min_tokens_for_schema
        )

        # But ensure we don't exceed the model's context window
        # Reserve space for input tokens and safety margin
        max_allowed_by_context = (
            context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
        )

        # Use the minimum of: desired, context limit, and model output limit
        effective_max_tokens = min(
            desired_max_tokens, max_allowed_by_context, model_max_output
        )

        return effective_max_tokens
    else:
        # For non-structured outputs, still respect context window and model limits
        max_allowed_by_context = (
            context_window - prompt_tokens_estimate - CONTEXT_SAFETY_MARGIN
        )
        effective_max_tokens = min(
            configured_max_tokens, max_allowed_by_context, model_max_output
        )
        return effective_max_tokens
