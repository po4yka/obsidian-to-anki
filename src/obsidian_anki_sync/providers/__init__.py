"""LLM provider abstractions and implementations."""

from .base import BaseLLMProvider
from .factory import ProviderFactory
from .lm_studio import LMStudioProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider

try:
    from .pydantic_ai_models import (
        PydanticAIModelFactory,
        create_openrouter_model_from_env,
    )

    _PYDANTIC_MODELS_AVAILABLE = True
except ModuleNotFoundError:
    PydanticAIModelFactory = None  # type: ignore[assignment, misc]
    create_openrouter_model_from_env = None  # type: ignore[assignment, misc]
    _PYDANTIC_MODELS_AVAILABLE = False

__all__ = [
    # Core providers
    "BaseLLMProvider",
    "LMStudioProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    "ProviderFactory",
]

if _PYDANTIC_MODELS_AVAILABLE:
    __all__ += [
        "PydanticAIModelFactory",
        "create_openrouter_model_from_env",
    ]
