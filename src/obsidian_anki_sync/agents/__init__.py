"""Multi-agent AI system for obsidian-to-anki conversion."""

from .langgraph import LangGraphOrchestrator
from .models import (
    AgentPipelineResult,
    GeneratedCard,
    GenerationResult,
    PostValidationResult,
    PreValidationResult,
)

try:
    from .pydantic import GeneratorAgentAI, PostValidatorAgentAI, PreValidatorAgentAI

    _PYDANTIC_AGENTS_AVAILABLE = True
except ModuleNotFoundError:
    GeneratorAgentAI = None  # type: ignore[assignment, misc]
    PostValidatorAgentAI = None  # type: ignore[assignment, misc]
    PreValidatorAgentAI = None  # type: ignore[assignment, misc]
    _PYDANTIC_AGENTS_AVAILABLE = False

__all__ = [
    "AgentPipelineResult",
    "GeneratedCard",
    "GenerationResult",
    "LangGraphOrchestrator",
    "PostValidationResult",
    "PreValidationResult",
]

if _PYDANTIC_AGENTS_AVAILABLE:
    __all__ += [
        "GeneratorAgentAI",
        "PostValidatorAgentAI",
        "PreValidatorAgentAI",
    ]
