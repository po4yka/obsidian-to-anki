"""Provider configuration models.

Type-safe configuration models for all LLM providers.
"""

from pydantic import BaseModel, Field, SecretStr


class BaseProviderConfig(BaseModel):
    """Base configuration for all LLM providers."""

    timeout: float = Field(default=240.0, ge=1.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")


class OllamaConfig(BaseProviderConfig):
    """Configuration for Ollama provider."""

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    timeout: float = Field(
        default=900.0, ge=1.0, description="Request timeout (longer for local models)"
    )


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""

    api_key: SecretStr = Field(description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI API base URL"
    )
    organization: str | None = Field(default=None, description="OpenAI organization ID")
    timeout: float = Field(default=240.0, ge=1.0, description="Request timeout")


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic provider."""

    api_key: SecretStr = Field(description="Anthropic API key")
    base_url: str = Field(
        default="https://api.anthropic.com", description="Anthropic API base URL"
    )
    api_version: str = Field(default="2023-06-01", description="Anthropic API version")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum response tokens")
    timeout: float = Field(default=240.0, ge=1.0, description="Request timeout")


class OpenRouterConfig(BaseProviderConfig):
    """Configuration for OpenRouter provider."""

    api_key: SecretStr = Field(description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    site_url: str | None = Field(
        default=None, description="Your site URL for OpenRouter attribution"
    )
    site_name: str | None = Field(
        default=None, description="Your site name for OpenRouter attribution"
    )
    max_tokens: int | None = Field(
        default=2048, ge=1, description="Maximum response tokens"
    )
    timeout: float = Field(
        default=300.0, ge=1.0, description="Request timeout (longer for routing)"
    )


class LMStudioConfig(BaseProviderConfig):
    """Configuration for LM Studio provider."""

    base_url: str = Field(
        default="http://localhost:1234/v1", description="LM Studio API base URL"
    )
    timeout: float = Field(
        default=600.0, ge=1.0, description="Request timeout (longer for local models)"
    )
