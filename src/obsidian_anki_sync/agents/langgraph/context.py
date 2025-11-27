"""Shared context for LangGraph pipeline execution.

Stores non-serializable resources (config and model instances) outside of
pipeline state so checkpoints remain JSON/pickle friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import Config


@dataclass
class PipelineContext:
    """Container for objects shared across LangGraph nodes."""

    config: Config
    resources: dict[str, Any]

    def get_resource(self, key: str) -> Any:
        return self.resources.get(key)

    def get_model(self, key: str) -> Any:
        """Backwards-compatible helper for stored model instances."""

        return self.get_resource(key)


_pipeline_context: PipelineContext | None = None


def set_pipeline_context(config: Config, resources: dict[str, Any]) -> None:
    """Initialize shared context for LangGraph nodes."""

    global _pipeline_context
    _pipeline_context = PipelineContext(config=config, resources=resources)


def get_pipeline_context() -> PipelineContext:
    """Fetch the initialized pipeline context."""

    if _pipeline_context is None:
        raise RuntimeError("Pipeline context has not been initialized")
    return _pipeline_context
