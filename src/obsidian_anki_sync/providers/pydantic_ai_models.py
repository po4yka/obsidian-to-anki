"""PydanticAI model providers for the agent system.

This module provides adapters between our existing LLM providers and PydanticAI's
model interface, enabling type-safe structured outputs and better agent capabilities.
"""

from typing import Any

import httpx
from pydantic import Field
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers.openrouter.retry_handler import RetryTransport
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class OpenRouterSettings(BaseSettings):
    """OpenRouter configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str | None = Field(
        default=None, description="OpenRouter API key"
    )
    openrouter_site_url: str | None = Field(
        default=None, description="Site URL for rankings"
    )
    openrouter_site_name: str | None = Field(
        default=None, description="Site name for rankings"
    )


class EnhancedOpenRouterModel(OpenAIChatModel):
    """Enhanced OpenRouter model with proper PydanticAI integration.

    This model extends PydanticAI's OpenAIChatModel and properly configures
    the OpenAI provider for OpenRouter compatibility. It does NOT override
    the request method to avoid API compatibility issues with PydanticAI.

    For reasoning support, use OpenRouter's native reasoning models or
    configure reasoning at the prompt/system level.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        site_name: str | None = None,
        max_tokens: int | None = None,
        reasoning_enabled: bool = False,
        **kwargs: Any,
    ):
        """Initialize enhanced OpenRouter model.

        Args:
            model_name: Model identifier
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            site_url: Site URL for rankings
            site_name: Site name for rankings
            max_tokens: Maximum tokens per request (stored for reference)
            reasoning_enabled: Enable reasoning for this model (stored for reference)
            **kwargs: Additional OpenAIChatModel arguments
        """
        # Build HTTP headers for OpenRouter
        http_headers: dict[str, str] = {}
        if site_url:
            http_headers["HTTP-Referer"] = site_url
        if site_name:
            http_headers["X-Title"] = site_name

        # Create retry transport with connection pooling
        retry_transport = RetryTransport(
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            max_retries=5,
            initial_delay=2.0,  # Start with 2s delay for rate limits
        )

        # Create explicitly configured async HTTP client
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=30.0),
            transport=retry_transport,
            headers=http_headers if http_headers else None,
        )

        # Initialize with OpenAIProvider for custom base_url support
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "",
            http_client=http_client,
        )
        super().__init__(
            model_name,
            provider=provider,
            **kwargs,
        )

        # Store configuration for reference (not used in request override)
        self.reasoning_enabled = reasoning_enabled
        self._max_tokens = max_tokens

        logger.info(
            "enhanced_openrouter_model_created",
            model=model_name,
            reasoning_enabled=reasoning_enabled,
            max_tokens=max_tokens,
        )

    # NOTE: We intentionally do NOT override the request() method.
    # PydanticAI's internal API uses ModelMessage, ModelSettings, and
    # ModelRequestParameters types that are not simple dicts. Overriding
    # this method incorrectly causes "'ModelRequest' object is not subscriptable"
    # errors. The parent OpenAIChatModel handles all request processing correctly.


class PydanticAIModelFactory:
    """Factory for creating PydanticAI model instances from configuration.

    Supports multiple providers through PydanticAI's unified interface:
    - OpenRouter (via OpenAI-compatible API)
    - OpenAI (direct)
    - Anthropic (future support)
    - Local models via Ollama (future support)
    """

    @staticmethod
    def create_enhanced_openrouter_model(
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        site_name: str | None = None,
        max_tokens: int | None = None,
        reasoning_enabled: bool = False,
        **kwargs: Any,
    ) -> EnhancedOpenRouterModel:
        """Create an enhanced OpenRouter model with reasoning support.

        This model uses the OpenRouterProvider internally to enable reasoning
        configuration and other enhancements not available in standard OpenAIModel.

        Args:
            model_name: Model identifier (e.g., "x-ai/grok-4.1-fast")
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            site_url: Site URL for rankings
            site_name: Site name for rankings
            max_tokens: Maximum tokens per request
            reasoning_enabled: Enable reasoning for this model
            **kwargs: Additional configuration

        Returns:
            EnhancedOpenRouterModel instance
        """
        return EnhancedOpenRouterModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            site_url=site_url,
            site_name=site_name,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            **kwargs,
        )

    @staticmethod
    def create_openrouter_model(
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ) -> OpenAIChatModel:
        """Create a PydanticAI model using OpenRouter.

        OpenRouter provides access to multiple LLM providers through an OpenAI-compatible API.

        Args:
            model_name: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            api_key: OpenRouter API key (if not provided, uses OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            site_url: Your site URL for OpenRouter rankings (optional)
            site_name: Your site name for OpenRouter rankings (optional)
            **kwargs: Additional configuration passed to OpenAIChatModel

        Returns:
            Configured OpenAIChatModel instance pointing to OpenRouter

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Get API key from environment if not provided
        if api_key is None:
            settings = OpenRouterSettings()
            api_key = settings.openrouter_api_key

        if not api_key:
            msg = (
                "OpenRouter API key is required. Provide it via the api_key parameter "
                "or set the OPENROUTER_API_KEY environment variable."
            )
            raise ValueError(msg)

        # Build HTTP headers for OpenRouter
        http_headers: dict[str, str] = {}
        if site_url:
            http_headers["HTTP-Referer"] = site_url
        if site_name:
            http_headers["X-Title"] = site_name

        # Create retry transport with connection pooling
        retry_transport = RetryTransport(
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            max_retries=5,
            initial_delay=2.0,
        )

        # Create explicitly configured async HTTP client
        # This ensures proper timeout and connection pooling configuration
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=30.0),
            transport=retry_transport,
            headers=http_headers if http_headers else None,
        )

        # Create OpenAI-compatible model pointing to OpenRouter
        # Use OpenAIProvider for custom base_url (PydanticAI API change)
        openai_provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )
        model = OpenAIChatModel(
            model_name,
            provider=openai_provider,
            **kwargs,
        )

        logger.info(
            "pydantic_ai_openrouter_model_created",
            model=model_name,
            base_url=base_url,
            has_site_info=bool(site_url and site_name),
        )

        return model  # type: ignore[no-any-return]

    @staticmethod
    def create_from_config(
        config: Config,
        model_name: str | None = None,
        provider: str | None = None,
        reasoning_enabled: bool | None = None,
        max_tokens: int | None = None,
    ) -> OpenAIChatModel:
        """Create a PydanticAI model from service configuration.

        Args:
            config: Service configuration
            model_name: Override model name (uses generator_model from config if not provided)
            provider: Override provider (uses llm_provider from config if not provided)
            reasoning_enabled: Override reasoning setting
            max_tokens: Override max tokens setting

        Returns:
            Configured PydanticAI model instance

        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider = provider or config.llm_provider
        model_name = model_name or config.generator_model

        if provider.lower() == "openrouter":
            # Use enhanced OpenRouter model for reasoning support
            return PydanticAIModelFactory.create_enhanced_openrouter_model(
                model_name=model_name,
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                site_url=config.openrouter_site_url,
                site_name=config.openrouter_site_name,
                max_tokens=max_tokens,
                reasoning_enabled=reasoning_enabled or False,
            )
        elif provider.lower() in ("ollama", "lm_studio", "lmstudio"):
            # For now, Ollama and LM Studio can use OpenAI-compatible interface
            # PydanticAI's OpenAIChatModel works with any OpenAI-compatible endpoint
            base_url = (
                config.ollama_base_url
                if provider.lower() == "ollama"
                else config.lm_studio_base_url
            )

            # Ollama and LM Studio don't require API keys for local usage
            # Use placeholder key if needed
            api_key = config.ollama_api_key or "local"

            logger.info(
                "pydantic_ai_local_model_created",
                provider=provider,
                model=model_name,
                base_url=base_url,
            )

            # Create explicitly configured async HTTP client for local providers
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=30.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                ),
            )

            # Use OpenAIProvider for custom base_url (PydanticAI API change)
            local_provider = OpenAIProvider(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client,
            )
            return OpenAIChatModel(  # type: ignore[no-any-return]
                model_name,
                provider=local_provider,
            )
        else:
            msg = (
                f"Unsupported provider for PydanticAI: {provider}. "
                f"Supported providers: openrouter, ollama, lm_studio"
            )
            raise ValueError(msg)


def create_openrouter_model_from_env(
    model_name: str = "anthropic/claude-3-5-sonnet",
) -> OpenAIChatModel:
    """Convenience function to create OpenRouter model from environment variables.

    Expects:
    - OPENROUTER_API_KEY: Your OpenRouter API key
    - OPENROUTER_SITE_URL (optional): Your site URL for rankings
    - OPENROUTER_SITE_NAME (optional): Your site name for rankings

    Args:
        model_name: Model to use (default: anthropic/claude-3-5-sonnet)

    Returns:
        Configured OpenAIChatModel instance
    """
    settings = OpenRouterSettings()
    return PydanticAIModelFactory.create_openrouter_model(
        model_name=model_name,
        api_key=settings.openrouter_api_key,
        site_url=settings.openrouter_site_url,
        site_name=settings.openrouter_site_name,
    )
