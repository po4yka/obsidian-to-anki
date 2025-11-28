"""OpenRouter provider package.

This package provides a modular implementation of the OpenRouter LLM provider.
Components are organized into separate modules for better maintainability.
"""

from .models import (
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    HTTP_STATUS_RETRYABLE,
    MODEL_CONTEXT_WINDOWS,
    MODEL_MAX_OUTPUT_TOKENS,
    MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS,
    MODELS_WITH_STRUCTURED_OUTPUT_ISSUES,
)
from .provider import OpenRouterProvider

__all__ = [
    "CONTEXT_SAFETY_MARGIN",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "HTTP_STATUS_RETRYABLE",
    "MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS",
    "MODELS_WITH_STRUCTURED_OUTPUT_ISSUES",
    "MODEL_CONTEXT_WINDOWS",
    "MODEL_MAX_OUTPUT_TOKENS",
    "OpenRouterProvider",
]
