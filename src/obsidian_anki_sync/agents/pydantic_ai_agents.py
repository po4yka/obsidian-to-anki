"""PydanticAI-based agent implementations for card generation pipeline.

Re-export facade. Implementations live in agents/pydantic_ai/ subpackage.
"""

from .pydantic_ai.enhancement_agents import (
    CardSplittingAgentAI,
    ContextEnrichmentAgentAI,
    ContextEnrichmentDeps,
    DuplicateDetectionAgentAI,
    DuplicateDetectionDeps,
    MemorizationQualityAgentAI,
)
from .pydantic_ai.generator_agent import GeneratorAgentAI
from .pydantic_ai.outputs import CardGenerationOutput
from .pydantic_ai.post_validator_agent import PostValidatorAgentAI
from .pydantic_ai.pre_validator_agent import PreValidatorAgentAI

__all__ = [
    "CardGenerationOutput",
    "CardSplittingAgentAI",
    "ContextEnrichmentAgentAI",
    "ContextEnrichmentDeps",
    "DuplicateDetectionAgentAI",
    "DuplicateDetectionDeps",
    "GeneratorAgentAI",
    "MemorizationQualityAgentAI",
    "PostValidatorAgentAI",
    "PreValidatorAgentAI",
]
