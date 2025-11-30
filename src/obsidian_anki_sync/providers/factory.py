"""Provider factory for creating LLM provider instances."""

from typing import Any, cast

from obsidian_anki_sync.utils.logging import get_logger

from .anthropic import AnthropicProvider
from .base import BaseLLMProvider
from .lm_studio import LMStudioProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating LLM provider instances.

    This factory creates the appropriate provider based on configuration,
    supporting Ollama (local/cloud), LM Studio, OpenRouter, OpenAI, and Anthropic.
    """

    PROVIDER_MAP = {
        "ollama": OllamaProvider,
        "lm_studio": LMStudioProvider,
        "lmstudio": LMStudioProvider,  # Alias
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,  # Alias for Anthropic
    }

    @classmethod
    def create_provider(
        cls, provider_type: str, verbose_logging: bool = False, **kwargs: Any
    ) -> BaseLLMProvider:
        """Create a provider instance based on type.

        Args:
            provider_type: Provider type ("ollama", "lm_studio", "openrouter", "openai", "anthropic", "claude")
            verbose_logging: Whether to log detailed initialization info
            **kwargs: Provider-specific configuration parameters

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider_type is not supported

        Examples:
            # Create Ollama local provider
            >>> provider = ProviderFactory.create_provider(
            ...     "ollama",
            ...     base_url="http://localhost:11434"
            ... )

            # Create OpenAI provider
            >>> provider = ProviderFactory.create_provider(
            ...     "openai",
            ...     api_key="sk-..."
            ... )

            # Create Anthropic (Claude) provider
            >>> provider = ProviderFactory.create_provider(
            ...     "anthropic",
            ...     api_key="sk-ant-..."
            ... )

            # Create OpenRouter provider
            >>> provider = ProviderFactory.create_provider(
            ...     "openrouter",
            ...     api_key="sk-or-..."
            ... )

            # Create LM Studio provider
            >>> provider = ProviderFactory.create_provider(
            ...     "lm_studio",
            ...     base_url="http://localhost:1234/v1"
            ... )
        """
        provider_type_lower = provider_type.lower()

        if provider_type_lower not in cls.PROVIDER_MAP:
            available = ", ".join(sorted(cls.PROVIDER_MAP.keys()))
            msg = (
                f"Unsupported provider type: {provider_type}. "
                f"Available providers: {available}"
            )
            raise ValueError(msg)

        provider_class = cls.PROVIDER_MAP[provider_type_lower]

        if verbose_logging:
            logger.info(
                "creating_provider",
                provider_type=provider_type,
                provider_class=provider_class.__name__,
            )

        try:
            provider = cast(
                "BaseLLMProvider",
                provider_class(verbose_logging=verbose_logging, **kwargs),
            )
            if verbose_logging:
                logger.info(
                    "provider_created_successfully", provider_type=provider_type
                )
            return provider
        except Exception as e:
            logger.error(
                "provider_creation_failed",
                provider_type=provider_type,
                error=str(e),
            )
            raise

    @classmethod
    def create_from_config(
        cls, config: Any, verbose_logging: bool = False
    ) -> BaseLLMProvider:
        """Create a provider instance from a Config object.

        Args:
            config: Configuration object with provider settings
            verbose_logging: Whether to log detailed initialization info

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider configuration is invalid
            AttributeError: If config doesn't have required attributes

        Examples:
            >>> from obsidian_anki_sync.config import Config
            >>> config = Config(
            ...     llm_provider="ollama",
            ...     ollama_base_url="http://localhost:11434"
            ... )
            >>> provider = ProviderFactory.create_from_config(config)
        """
        provider_type = getattr(config, "llm_provider", "ollama").lower()

        # Build provider-specific kwargs
        kwargs: dict[str, Any] = {}

        if provider_type == "ollama":
            kwargs = {
                "base_url": getattr(
                    config, "ollama_base_url", "http://localhost:11434"
                ),
                "api_key": getattr(config, "ollama_api_key", None),
                "timeout": getattr(config, "llm_timeout", 120.0),
            }

        elif provider_type in ("lm_studio", "lmstudio"):
            # Use 'or' to handle both missing attribute AND explicit None value
            max_tokens = getattr(config, "llm_max_tokens", None) or 2048
            kwargs = {
                "base_url": getattr(
                    config, "lm_studio_base_url", "http://localhost:1234/v1"
                ),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_tokens": max_tokens,
            }

        elif provider_type == "openrouter":
            # Use 'or' to handle both missing attribute AND explicit None value
            max_tokens = getattr(config, "llm_max_tokens", None) or 2048
            kwargs = {
                "api_key": getattr(config, "openrouter_api_key", None),
                "base_url": getattr(
                    config, "openrouter_base_url", "https://openrouter.ai/api/v1"
                ),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_tokens": max_tokens,
                "site_url": getattr(config, "openrouter_site_url", None),
                "site_name": getattr(config, "openrouter_site_name", None),
            }

        elif provider_type == "openai":
            kwargs = {
                "api_key": getattr(config, "openai_api_key", None),
                "base_url": getattr(
                    config, "openai_base_url", "https://api.openai.com/v1"
                ),
                "organization": getattr(config, "openai_organization", None),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_retries": getattr(config, "openai_max_retries", 3),
            }

        elif provider_type in ("anthropic", "claude"):
            # Use 'or' to handle both missing attribute AND explicit None value
            max_tokens = getattr(config, "llm_max_tokens", None) or 4096
            kwargs = {
                "api_key": getattr(config, "anthropic_api_key", None),
                "base_url": getattr(
                    config, "anthropic_base_url", "https://api.anthropic.com"
                ),
                "api_version": getattr(config, "anthropic_api_version", "2023-06-01"),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_tokens": max_tokens,
                "max_retries": getattr(config, "anthropic_max_retries", 3),
            }

        else:
            msg = f"Unsupported provider type in config: {provider_type}"
            raise ValueError(msg)

        if verbose_logging:
            logger.info(
                "creating_provider_from_config",
                provider_type=provider_type,
                config_attributes=list(kwargs.keys()),
            )

        return cls.create_provider(
            provider_type, verbose_logging=verbose_logging, **kwargs
        )

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """List all supported provider types.

        Returns:
            List of supported provider type identifiers
        """
        return sorted(set(cls.PROVIDER_MAP.keys()))
