"""Payload construction utilities for OpenRouter API requests."""

from typing import Any

from obsidian_anki_sync.utils.reasoning import normalize_reasoning_effort

from .models import (
    MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS,
    MODELS_WITH_REASONING_SUPPORT,
)


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


def _should_use_reasoning_with_schema(json_schema: dict[str, Any]) -> bool:
    """Determine if reasoning should be enabled with JSON schema based on complexity.

    Args:
        json_schema: JSON schema dictionary

    Returns:
        True if reasoning should be enabled for this schema
    """
    schema_dict = json_schema.get("schema", {})

    # Enable reasoning for complex schemas that would benefit from step-by-step thinking
    if not isinstance(schema_dict, dict):
        return False

    # Check for complex nested structures
    properties = schema_dict.get("properties", {})
    if len(properties) > 5:  # Many properties suggest complex output
        return True

    # Check for deeply nested objects or arrays
    def _count_nested_complexity(
        obj: dict[str, Any],
        depth: int = 0,
        max_depth: int = 10,
        visited: set | None = None,
    ) -> int:
        """Count nested complexity in schema with recursion protection."""
        # Protect against infinite recursion
        if depth > max_depth:
            return 0

        # Skip empty or non-object schemas
        if not obj or not isinstance(obj, dict):
            return 0

        # Track visited objects to prevent circular reference loops
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return 0
        visited.add(obj_id)

        complexity = 0

        if depth > 2:  # Deep nesting
            complexity += 1

        # Check properties
        props = obj.get("properties")
        if props and isinstance(props, dict) and len(props) > 3:
            complexity += len(props) // 2

        # Check for arrays with complex items
        items = obj.get("items")
        if items and isinstance(items, dict) and items.get("type"):
            if items.get("type") == "object" and items.get("properties"):
                complexity += 2
            complexity += _count_nested_complexity(items, depth + 1, max_depth, visited)

        # Check for nested objects in properties
        if props and isinstance(props, dict):
            for prop_schema in props.values():
                if isinstance(prop_schema, dict) and prop_schema.get("type"):
                    if prop_schema.get("type") == "object" and prop_schema.get(
                        "properties"
                    ):
                        complexity += _count_nested_complexity(
                            prop_schema, depth + 1, max_depth, visited
                        )
                    elif prop_schema.get("type") == "array":
                        complexity += 1
                        # Also check items of array properties
                        array_items = prop_schema.get("items")
                        if array_items and isinstance(array_items, dict):
                            if array_items.get("type") == "object" and array_items.get(
                                "properties"
                            ):
                                complexity += 2  # Array of objects is complex
                            complexity += _count_nested_complexity(
                                array_items, depth + 1, max_depth, visited
                            )

        return complexity

    complexity_score = _count_nested_complexity(schema_dict)

    # Enable reasoning for schemas with moderate to high complexity
    return complexity_score >= 3


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
    reasoning_effort: str | None = None,
    stream: bool = False,
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
        "stream": stream,
    }

    effort_value = _resolve_reasoning_effort(
        model=model,
        requested_effort=reasoning_effort,
        reasoning_enabled=reasoning_enabled,
        json_schema=json_schema,
    )
    if effort_value:
        payload["reasoning"] = {"effort": effort_value}
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


def _resolve_reasoning_effort(
    model: str,
    requested_effort: str | None,
    reasoning_enabled: bool,
    json_schema: dict[str, Any] | None,
) -> str | None:
    """Determine reasoning effort for the current request."""
    normalized = (
        normalize_reasoning_effort(requested_effort)
        if requested_effort is not None
        else "auto"
    )

    supports_reasoning = model in MODELS_WITH_REASONING_SUPPORT

    if normalized == "none":
        return None

    if normalized == "auto":
        if not supports_reasoning:
            return None
        should_enable = reasoning_enabled or (
            json_schema and _should_use_reasoning_with_schema(json_schema)
        )
        if should_enable:
            return MODELS_WITH_REASONING_SUPPORT[model]
        return None

    if supports_reasoning:
        return normalized

    return None
