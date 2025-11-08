"""LLM provider abstractions and implementations."""

from .base import BaseLLMProvider
from .factory import ProviderFactory
from .lm_studio import LMStudioProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider
from .pydantic_ai_models import PydanticAIModelFactory, create_openrouter_model_from_env

__all__ = [
    # Legacy providers
    "BaseLLMProvider",
    "ProviderFactory",
    "LMStudioProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    # PydanticAI providers
    "PydanticAIModelFactory",
    "create_openrouter_model_from_env",
]
