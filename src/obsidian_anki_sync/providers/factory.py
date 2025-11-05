"""Provider factory for creating LLM provider instances."""

from typing import Any, cast

from ..utils.logging import get_logger
from .base import BaseLLMProvider
from .lm_studio import LMStudioProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider

logger = get_logger(__name__)


class ProviderFactory:
    """Factory for creating LLM provider instances.

    This factory creates the appropriate provider based on configuration,
    supporting Ollama (local/cloud), LM Studio, and OpenRouter.
    """

    PROVIDER_MAP = {
        "ollama": OllamaProvider,
        "lm_studio": LMStudioProvider,
        "lmstudio": LMStudioProvider,  # Alias
        "openrouter": OpenRouterProvider,
    }

    @classmethod
    def create_provider(cls, provider_type: str, **kwargs: Any) -> BaseLLMProvider:
        """Create a provider instance based on type.

        Args:
            provider_type: Provider type ("ollama", "lm_studio", "openrouter")
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

            # Create Ollama cloud provider
            >>> provider = ProviderFactory.create_provider(
            ...     "ollama",
            ...     base_url="https://api.ollama.com",
            ...     api_key="your-api-key"
            ... )

            # Create LM Studio provider
            >>> provider = ProviderFactory.create_provider(
            ...     "lm_studio",
            ...     base_url="http://localhost:1234/v1"
            ... )

            # Create OpenRouter provider
            >>> provider = ProviderFactory.create_provider(
            ...     "openrouter",
            ...     api_key="your-api-key"
            ... )
        """
        provider_type_lower = provider_type.lower()

        if provider_type_lower not in cls.PROVIDER_MAP:
            available = ", ".join(sorted(cls.PROVIDER_MAP.keys()))
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Available providers: {available}"
            )

        provider_class = cls.PROVIDER_MAP[provider_type_lower]

        logger.info(
            "creating_provider",
            provider_type=provider_type,
            provider_class=provider_class.__name__,
        )

        try:
            provider = cast(BaseLLMProvider, provider_class(**kwargs))
            logger.info("provider_created_successfully", provider_type=provider_type)
            return provider
        except Exception as e:
            logger.error(
                "provider_creation_failed",
                provider_type=provider_type,
                error=str(e),
            )
            raise

    @classmethod
    def create_from_config(cls, config: Any) -> BaseLLMProvider:
        """Create a provider instance from a Config object.

        Args:
            config: Configuration object with provider settings

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
            kwargs = {
                "base_url": getattr(
                    config, "lm_studio_base_url", "http://localhost:1234/v1"
                ),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_tokens": getattr(config, "llm_max_tokens", 2048),
            }

        elif provider_type == "openrouter":
            kwargs = {
                "api_key": getattr(config, "openrouter_api_key", None),
                "base_url": getattr(
                    config, "openrouter_base_url", "https://openrouter.ai/api/v1"
                ),
                "timeout": getattr(config, "llm_timeout", 120.0),
                "max_tokens": getattr(config, "llm_max_tokens", 2048),
                "site_url": getattr(config, "openrouter_site_url", None),
                "site_name": getattr(config, "openrouter_site_name", None),
            }

        else:
            raise ValueError(f"Unsupported provider type in config: {provider_type}")

        logger.info(
            "creating_provider_from_config",
            provider_type=provider_type,
            config_attributes=list(kwargs.keys()),
        )

        return cls.create_provider(provider_type, **kwargs)

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """List all supported provider types.

        Returns:
            List of supported provider type identifiers
        """
        return sorted(set(cls.PROVIDER_MAP.keys()))
