"""Payload construction utilities for OpenRouter API requests."""

from typing import Any

from .models import MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS


def build_messages(prompt: str, system: str) -> list[dict[str, str]]:
    """Build messages array in OpenAI format.

    Args:
        prompt: User prompt text
        system: System prompt text

    Returns:
        List of message dictionaries
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def optimize_schema_for_request(schema: dict[str, Any]) -> dict[str, Any]:
    """Optimize JSON schema for token efficiency.

    Removes verbose descriptions and metadata that don't affect validation
    but consume tokens. Keeps essential validation rules.

    Args:
        schema: Original JSON schema dictionary

    Returns:
        Optimized schema dictionary
    """
    if not isinstance(schema, dict):
        return schema

    optimized = {}

    # Keep essential fields
    essential_fields = {
        "type",
        "properties",
        "items",
        "required",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "additionalProperties",
        "anyOf",
        "oneOf",
        "allOf",
    }

    for key, value in schema.items():
        if key in essential_fields:
            if key == "properties" and isinstance(value, dict):
                # Recursively optimize nested properties
                optimized[key] = {
                    k: optimize_schema_for_request(v) for k, v in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Optimize array items schema
                optimized[key] = optimize_schema_for_request(value)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                # Optimize union types
                optimized_list = [
                    (
                        optimize_schema_for_request(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
                optimized[key] = optimized_list  # type: ignore[assignment]
            else:
                optimized[key] = value

    return optimized


def build_payload(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    effective_max_tokens: int,
    json_schema: dict[str, Any] | None = None,
    format: str = "",
    reasoning_enabled: bool = False,
) -> dict[str, Any]:
    """Build request payload for OpenRouter API.

    Args:
        model: Model identifier
        messages: Messages array
        temperature: Sampling temperature
        effective_max_tokens: Calculated max_tokens value
        json_schema: Optional JSON schema for structured output
        format: Response format (e.g., "json")
        reasoning_enabled: Enable reasoning mode

    Returns:
        Request payload dictionary
    """
    # For Grok 4.1 Fast with JSON schema: use temperature=0 for deterministic output
    effective_temperature = temperature
    if json_schema and model in MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS:
        if "grok" in model.lower() and temperature > 0:
            effective_temperature = 0.0

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": effective_temperature,
        "max_tokens": effective_max_tokens,
        "stream": False,
    }

    # For Grok models: explicitly disable reasoning when using JSON schema
    if "grok" in model.lower() and json_schema:
        payload["reasoning"] = {"enabled": False}
    elif reasoning_enabled and not json_schema:
        payload["reasoning_enabled"] = True

    # Handle structured output
    if json_schema:
        schema_dict = json_schema.get("schema", {})
        schema_name = json_schema.get("name", "response")

        # Validate schema structure
        if not schema_dict or not isinstance(schema_dict, dict):
            # Fallback to basic JSON mode if schema is invalid
            payload["response_format"] = {"type": "json_object"}
        else:
            # Determine strict mode based on model compatibility
            default_strict = json_schema.get("strict", True)

            # Optimize schema: remove unnecessary metadata for token efficiency
            optimized_schema = optimize_schema_for_request(schema_dict)

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": default_strict,
                    "schema": optimized_schema,
                },
            }
    elif format == "json":
        payload["response_format"] = {"type": "json_object"}

    return payload
