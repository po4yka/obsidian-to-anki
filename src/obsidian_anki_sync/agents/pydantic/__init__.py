"""PydanticAI-based agent implementations.

This package provides type-safe agents using PydanticAI for card generation pipeline.
"""

from .card_splitting import CardSplittingAgentAI
from .context_enrichment import ContextEnrichmentAgentAI
from .duplicate_detection import DuplicateDetectionAgentAI
from .highlight import HighlightAgentAI
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
    HighlightCandidateOutput,
    HighlightDeps,
    HighlightOutput,
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
    "CardGenerationOutput",
    "CardSplitPlanOutput",
    "CardSplittingAgentAI",
    "CardSplittingOutput",
    "ContextEnrichmentAgentAI",
    "ContextEnrichmentDeps",
    "ContextEnrichmentOutput",
    "DuplicateDetectionAgentAI",
    "DuplicateDetectionDeps",
    "DuplicateDetectionOutput",
    "DuplicateMatchOutput",
    "EnrichmentAdditionOutput",
    "GenerationDeps",
    "GeneratorAgentAI",
    "HighlightAgentAI",
    "MemorizationIssue",
    "MemorizationQualityAgentAI",
    "MemorizationQualityOutput",
    "PostValidationDeps",
    "PostValidationOutput",
    "PostValidatorAgentAI",
    "PreValidationDeps",
    # Models
    "PreValidationOutput",
    # Agents
    "PreValidatorAgentAI",
    "HighlightCandidateOutput",
    "HighlightDeps",
    "HighlightOutput",
    "SplitValidatorAgentAI",
]
