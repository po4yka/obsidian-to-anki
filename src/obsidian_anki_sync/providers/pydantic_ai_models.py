"""PydanticAI model providers for the agent system.

This module provides adapters between our existing LLM providers and PydanticAI's
model interface, enabling type-safe structured outputs and better agent capabilities.
"""

import os
from typing import Any

import httpx
from pydantic import Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import Config
from ..utils.logging import get_logger
from .openrouter import OpenRouterProvider

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
        default=None, description="OpenRouter API key")
    openrouter_site_url: str | None = Field(
        default=None, description="Site URL for rankings")
    openrouter_site_name: str | None = Field(
        default=None, description="Site name for rankings")


class EnhancedOpenRouterModel(OpenAIModel):
    """Enhanced OpenRouter model that uses OpenRouterProvider for reasoning support.

    This model extends PydanticAI's OpenAIModel but routes requests through
    the OpenRouterProvider to enable reasoning configuration and other enhancements.
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
            max_tokens: Maximum tokens per request
            reasoning_enabled: Enable reasoning for this model
            **kwargs: Additional OpenAIModel arguments
        """
        # Initialize with basic config first
        super().__init__(
            model_name,
            base_url=base_url,
            api_key=api_key or "dummy",  # Will be overridden
            **kwargs,
        )

        # Create OpenRouter provider instance
        self.provider = OpenRouterProvider(
            api_key=api_key,
            base_url=base_url,
            site_url=site_url,
            site_name=site_name,
            max_tokens=max_tokens,
        )

        # Store reasoning configuration
        self.reasoning_enabled = reasoning_enabled

        logger.info(
            "enhanced_openrouter_model_created",
            model=model_name,
            reasoning_enabled=reasoning_enabled,
            max_tokens=max_tokens,
        )

    async def request(
        self,
        messages: list[dict[str, Any]],
        model_settings: dict[str, Any] | None = None,
        model_request_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a request using the OpenRouterProvider with reasoning support.

        This overrides PydanticAI's default request method to use our enhanced
        OpenRouter provider which supports reasoning configuration.

        Args:
            messages: Chat messages in OpenAI format
            model_settings: Model settings (temperature, etc.)
            model_request_parameters: Additional request parameters

        Returns:
            Response dictionary in OpenAI format
        """
        # Extract parameters
        temperature = model_settings.get(
            "temperature", 0.7) if model_settings else 0.7
        max_tokens = model_settings.get(
            "max_tokens") if model_settings else None

        # Build system message and user prompt
        system_message = ""
        user_messages = []

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            elif message["role"] == "user":
                user_messages.append(message["content"])

        # Combine user messages (PydanticAI sometimes splits them)
        prompt = " ".join(user_messages)

        # Make request through OpenRouterProvider
        response = self.provider.generate(
            model=self.model_name,
            prompt=prompt,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_enabled=self.reasoning_enabled,
        )

        # Convert response to OpenAI format expected by PydanticAI
        return {
            "choices": [
                {
                    "finish_reason": response.get("finish_reason", "stop"),
                    "index": 0,
                    "message": {
                        "content": response["response"],
                        "role": "assistant",
                    },
                }
            ],
            "model": response.get("model", self.model_name),
            "usage": response.get("usage", {}),
        }


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
    ) -> OpenAIModel:
        """Create a PydanticAI model using OpenRouter.

        OpenRouter provides access to multiple LLM providers through an OpenAI-compatible API.

        Args:
            model_name: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            api_key: OpenRouter API key (if not provided, uses OPENROUTER_API_KEY env var)
            base_url: OpenRouter API base URL
            site_url: Your site URL for OpenRouter rankings (optional)
            site_name: Your site name for OpenRouter rankings (optional)
            **kwargs: Additional configuration passed to OpenAIModel

        Returns:
            Configured OpenAIModel instance pointing to OpenRouter

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Get API key from environment if not provided
        if api_key is None:
            settings = OpenRouterSettings()
            api_key = settings.openrouter_api_key

        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it via the api_key parameter "
                "or set the OPENROUTER_API_KEY environment variable."
            )

        # Build HTTP headers for OpenRouter
        http_headers: dict[str, str] = {}
        if site_url:
            http_headers["HTTP-Referer"] = site_url
        if site_name:
            http_headers["X-Title"] = site_name

        # Create explicitly configured async HTTP client
        # This ensures proper timeout and connection pooling configuration
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            headers=http_headers if http_headers else None,
        )

        # Create OpenAI-compatible model pointing to OpenRouter
        model = OpenAIModel(  # type: ignore[call-overload]
            model_name,
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
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
    ) -> OpenAIModel:
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
            # PydanticAI's OpenAIModel works with any OpenAI-compatible endpoint
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
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                ),
            )

            return OpenAIModel(  # type: ignore[call-overload,no-any-return]
                model_name,
                base_url=base_url,
                api_key=api_key,
                http_client=http_client,
            )
        else:
            raise ValueError(
                f"Unsupported provider for PydanticAI: {provider}. "
                f"Supported providers: openrouter, ollama, lm_studio"
            )


def create_openrouter_model_from_env(
    model_name: str = "anthropic/claude-3-5-sonnet",
) -> OpenAIModel:
    """Convenience function to create OpenRouter model from environment variables.

    Expects:
    - OPENROUTER_API_KEY: Your OpenRouter API key
    - OPENROUTER_SITE_URL (optional): Your site URL for rankings
    - OPENROUTER_SITE_NAME (optional): Your site name for rankings

    Args:
        model_name: Model to use (default: anthropic/claude-3-5-sonnet)

    Returns:
        Configured OpenAIModel instance
    """
    settings = OpenRouterSettings()
    return PydanticAIModelFactory.create_openrouter_model(
        model_name=model_name,
        api_key=settings.openrouter_api_key,
        site_url=settings.openrouter_site_url,
        site_name=settings.openrouter_site_name,
    )
