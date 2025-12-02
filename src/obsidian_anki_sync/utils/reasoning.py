"""Utilities for normalizing reasoning-effort configuration."""

from __future__ import annotations

from typing import Final

VALID_REASONING_EFFORTS: Final[set[str]] = {
    "minimal",
    "low",
    "medium",
    "high",
    "auto",
    "none",
}


def normalize_reasoning_effort(value: str | None) -> str:
    """Normalize configuration strings into canonical reasoning-effort values."""
    if value is None:
        return "auto"

    normalized = value.strip().lower()
    if normalized not in VALID_REASONING_EFFORTS:
        msg = (
            "Reasoning effort must be one of "
            f"{', '.join(sorted(VALID_REASONING_EFFORTS))}: {value}"
        )
        raise ValueError(msg)
    return normalized


def effort_for_openrouter(value: str | None) -> str | None:
    """Convert config effort values into OpenRouter-compatible payload hints."""
    normalized = normalize_reasoning_effort(value)
    if normalized in {"auto", "none"}:
        return None
    return normalized


def effort_for_openai(value: str | None) -> str | None:
    """Convert config effort values into OpenAI-compatible reasoning hints."""
    normalized = normalize_reasoning_effort(value)
    if normalized in {"auto", "none"}:
        return None
    if normalized == "minimal":
        return "low"
    return normalized
