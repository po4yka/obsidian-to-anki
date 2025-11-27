"""PydanticAI-based agent implementations.

This package provides type-safe agents using PydanticAI for card generation pipeline.
"""

from .card_splitting import CardSplittingAgentAI
from .context_enrichment import ContextEnrichmentAgentAI
from .duplicate_detection import DuplicateDetectionAgentAI
from .generator import GeneratorAgentAI
from .memorization import MemorizationQualityAgentAI
from .models import (
    CardSplitPlanOutput,
    CardSplittingOutput,
    ContextEnrichmentDeps,
    ContextEnrichmentOutput,
    DuplicateDetectionDeps,
    DuplicateDetectionOutput,
    DuplicateMatchOutput,
    EnrichmentAdditionOutput,
    GenerationDeps,
    MemorizationIssue,
    MemorizationQualityOutput,
    PostValidationDeps,
    PostValidationOutput,
    PreValidationDeps,
    PreValidationOutput,
)
from .post_validator import PostValidatorAgentAI
from .pre_validator import PreValidatorAgentAI
from .split_validator import SplitValidatorAgentAI

__all__ = [
    # Agents
    "PreValidatorAgentAI",
    "GeneratorAgentAI",
    "PostValidatorAgentAI",
    "MemorizationQualityAgentAI",
    "CardSplittingAgentAI",
    "DuplicateDetectionAgentAI",
    "ContextEnrichmentAgentAI",
    "SplitValidatorAgentAI",
    # Models
    "PreValidationOutput",
    "PreValidationDeps",
    "CardGenerationOutput",
    "GenerationDeps",
    "PostValidationOutput",
    "PostValidationDeps",
    "MemorizationIssue",
    "MemorizationQualityOutput",
    "CardSplitPlanOutput",
    "CardSplittingOutput",
    "DuplicateMatchOutput",
    "DuplicateDetectionOutput",
    "DuplicateDetectionDeps",
    "EnrichmentAdditionOutput",
    "ContextEnrichmentOutput",
    "ContextEnrichmentDeps",
]
