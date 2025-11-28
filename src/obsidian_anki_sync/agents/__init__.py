"""Multi-agent AI system for obsidian-to-anki conversion."""

from .models import (
    AgentPipelineResult,
    GeneratedCard,
    GenerationResult,
    PostValidationResult,
    PreValidationResult,
)
from .orchestrator import AgentOrchestrator

try:
    from .pydantic import GeneratorAgentAI, PostValidatorAgentAI, PreValidatorAgentAI

    _PYDANTIC_AGENTS_AVAILABLE = True
except ModuleNotFoundError:
    GeneratorAgentAI = None  # type: ignore[assignment, misc]
    PostValidatorAgentAI = None  # type: ignore[assignment, misc]
    PreValidatorAgentAI = None  # type: ignore[assignment, misc]
    _PYDANTIC_AGENTS_AVAILABLE = False

try:
    from .langgraph import LangGraphOrchestrator

    _LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:
    LangGraphOrchestrator = None  # type: ignore[assignment, misc]
    _LANGGRAPH_AVAILABLE = False

__all__ = [
    # Result models
    "PreValidationResult",
    "GeneratedCard",
    "GenerationResult",
    "PostValidationResult",
    "AgentPipelineResult",
    # Orchestrators
    "AgentOrchestrator",  # Legacy orchestrator
]

if _LANGGRAPH_AVAILABLE:
    __all__.append("LangGraphOrchestrator")

if _PYDANTIC_AGENTS_AVAILABLE:
    __all__.extend(
        [
            "PreValidatorAgentAI",
            "GeneratorAgentAI",
            "PostValidatorAgentAI",
        ]
    )
