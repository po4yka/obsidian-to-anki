"""PydanticAI model providers for the agent system.

This module provides adapters between our existing LLM providers and PydanticAI's
model interface, enabling type-safe structured outputs and better agent capabilities.
"""

import os
from typing import Any

from pydantic_ai.models.openai import OpenAIModel

from ..config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class PydanticAIModelFactory:
    """Factory for creating PydanticAI model instances from configuration.

    Supports multiple providers through PydanticAI's unified interface:
    - OpenRouter (via OpenAI-compatible API)
    - OpenAI (direct)
    - Anthropic (future support)
    - Local models via Ollama (future support)
    """

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
            api_key = os.environ.get("OPENROUTER_API_KEY")

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

        # Create OpenAI-compatible model pointing to OpenRouter
        model = OpenAIModel(  # type: ignore[call-overload]
            model_name,
            base_url=base_url,
            api_key=api_key,
            http_client=None,  # Will use default httpx client
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
    ) -> OpenAIModel:
        """Create a PydanticAI model from service configuration.

        Args:
            config: Service configuration
            model_name: Override model name (uses generator_model from config if not provided)
            provider: Override provider (uses llm_provider from config if not provided)

        Returns:
            Configured PydanticAI model instance

        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider = provider or config.llm_provider
        model_name = model_name or config.generator_model

        if provider.lower() == "openrouter":
            return PydanticAIModelFactory.create_openrouter_model(
                model_name=model_name,
                api_key=config.openrouter_api_key,
                base_url=config.openrouter_base_url,
                site_url=config.openrouter_site_url,
                site_name=config.openrouter_site_name,
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

            return OpenAIModel(  # type: ignore[call-overload,no-any-return]
                model_name,
                base_url=base_url,
                api_key=api_key,
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
    return PydanticAIModelFactory.create_openrouter_model(
        model_name=model_name,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        site_url=os.environ.get("OPENROUTER_SITE_URL"),
        site_name=os.environ.get("OPENROUTER_SITE_NAME"),
    )
