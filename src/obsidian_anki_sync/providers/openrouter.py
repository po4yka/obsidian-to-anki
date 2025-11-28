"""OpenRouter provider implementation.

This module provides backward compatibility by re-exporting from the
refactored openrouter package. The actual implementation is now split
across multiple modules in the openrouter/ directory for better
maintainability.

For new code, import directly from openrouter package:
    from .openrouter import OpenRouterProvider

This file will be maintained for backward compatibility.
"""

# Re-export everything from the refactored package
from .openrouter import (
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    HTTP_STATUS_RETRYABLE,
    MODEL_CONTEXT_WINDOWS,
    MODEL_MAX_OUTPUT_TOKENS,
    MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS,
    MODELS_WITH_STRUCTURED_OUTPUT_ISSUES,
    OpenRouterProvider,
)

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
