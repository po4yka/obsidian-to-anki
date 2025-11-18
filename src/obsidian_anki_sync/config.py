"""Configuration management for the sync service."""

import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from .exceptions import ConfigurationError
from .utils.path_validator import (
    validate_db_path,
    validate_source_dir,
    validate_vault_path,
)


class Config(BaseSettings):
    """Service configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    # Required fields
    # Obsidian paths - vault_path can be empty string from env, will be validated
    vault_path: Path | str = Field(default="", description="Path to Obsidian vault")
    source_dir: Path = Field(default=Path("."), description="Source directory within vault")

    @field_validator("vault_path", mode="before")
    @classmethod
    def parse_vault_path(cls, v: Any) -> Path:
        """Convert string to Path for vault_path."""
        if isinstance(v, str):
            if not v:
                # Empty string means not set - will be caught by validation
                return Path("")
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @field_validator("source_dir", "db_path", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @field_validator("source_subdirs", mode="before")
    @classmethod
    def parse_source_subdirs(cls, v: Any) -> list[Path] | None:
        """Convert source_subdirs to list of Paths."""
        if v is None:
            return None
        if isinstance(v, str):
            return [Path(v)]
        if isinstance(v, list):
            return [Path(str(d)) for d in v]
        return v

    # Anki settings
    anki_connect_url: str = Field(
        default="http://127.0.0.1:8765", description="AnkiConnect URL"
    )
    anki_deck_name: str = Field(
        default="Interview Questions", description="Anki deck name"
    )
    anki_note_type: str = Field(
        default="APF::Simple", description="Anki note type"
    )

    # Runtime settings
    run_mode: str = Field(
        default="apply", description="Run mode: 'apply' or 'dry-run'"
    )
    delete_mode: str = Field(
        default="delete", description="Delete mode: 'delete' or 'archive'"
    )

    # Database
    db_path: Path = Field(
        default=Path(".sync_state.db"), description="Path to sync state database"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    # Optional fields (with defaults)
    # Obsidian source directories (optional - overrides source_dir if provided)
    # List of relative paths from vault_path to search for Q&A notes
    # Example: [".", "Interviews", "CS/Algorithms"]
    source_subdirs: list[Path] | None = Field(
        default=None, description="List of source subdirectories"
    )
    # LLM Provider Configuration
    # Unified provider system - choose one: 'ollama', 'lm_studio', 'openrouter'
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 900.0  # 15 minutes default for large models
    llm_max_tokens: int = 128000  # Increased to 128k to prevent truncation for large bilingual content
    llm_reasoning_enabled: bool = False  # Enable reasoning mode for models that support it (e.g., DeepSeek)

    # Ollama provider settings (local or cloud)
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None  # Only for Ollama Cloud

    # LM Studio provider settings
    lm_studio_base_url: str = "http://localhost:1234/v1"

    # OpenAI provider settings
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_organization: str | None = None
    openai_max_retries: int = 3

    # Anthropic provider settings
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_version: str = "2023-06-01"
    anthropic_max_retries: int = 3

    # OpenRouter provider settings
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_site_name: str | None = None

    # Legacy OpenRouter settings (for backward compatibility)
    openrouter_model: str = "openai/gpt-4"

    # Deck export settings (for .apkg generation) - optional with defaults
    export_deck_name: str | None = None
    export_deck_description: str = ""
    export_output_path: Path | None = None

    # Agent system settings (optional, defaults provided)
    use_agent_system: bool = False

    # Parser-Repair Agent
    parser_repair_enabled: bool = True
    parser_repair_model: str = "qwen3:8b"
    parser_repair_temperature: float = 0.0

    # Q&A Extractor Agent (flexible LLM-based Q&A extraction)
    qa_extractor_model: str = "qwen3:8b"
    qa_extractor_temperature: float = 0.0

    # Pre-Validator Agent
    pre_validator_model: str = "qwen3:8b"
    pre_validator_temperature: float = 0.0
    pre_validation_enabled: bool = True

    # Generator Agent
    generator_model: str = "qwen3:32b"
    generator_temperature: float = 0.3

    # Post-Validator Agent
    post_validator_model: str = "qwen3:14b"
    post_validator_temperature: float = 0.0
    post_validation_max_retries: int = 3
    post_validation_auto_fix: bool = True
    post_validation_strict_mode: bool = True

    # LangGraph + PydanticAI Agent System (new!)
    use_langgraph: bool = False  # Enable LangGraph-based orchestration
    use_pydantic_ai: bool = False  # Enable PydanticAI for structured outputs

    # ============================================================================
    # Unified Model Configuration (OpenRouter)
    # ============================================================================
    # Default model used by ALL agents unless specifically overridden
    # Optimized 2025 models: MiniMax M2, Kimi K2, DeepSeek V3, Qwen 2.5
    default_llm_model: str = "qwen/qwen-2.5-72b-instruct"

    # Individual agent model overrides (optimized for each task)
    # Pre-validator: Fast, efficient validation
    pydantic_ai_pre_validator_model: str = "qwen/qwen-2.5-32b-instruct"
    # Generator: Powerful content creation
    pydantic_ai_generator_model: str = "qwen/qwen-2.5-72b-instruct"
    # Post-validator: Strong reasoning and quality checks
    pydantic_ai_post_validator_model: str = "deepseek/deepseek-chat"
    # Context Enrichment: Excellent for code generation and creative examples
    context_enrichment_model: str = "minimax/minimax-m2"
    # Memorization Quality: Strong analytical capabilities
    memorization_quality_model: str = "moonshotai/kimi-k2"
    # Card Splitting: Advanced reasoning for decision making
    card_splitting_model: str = "moonshotai/kimi-k2-thinking"
    # Duplicate Detection: Efficient comparison
    duplicate_detection_model: str = "qwen/qwen-2.5-32b-instruct"

    # LangGraph Workflow Configuration
    langgraph_max_retries: int = 3
    langgraph_auto_fix: bool = True
    langgraph_strict_mode: bool = True
    langgraph_checkpoint_enabled: bool = True  # Enable state persistence

    # Enhancement Agents (optional quality improvements)
    enable_card_splitting: bool = True  # Analyze if note should be split
    enable_context_enrichment: bool = True  # Add examples and mnemonics
    enable_memorization_quality: bool = True  # Check SRS effectiveness
    enable_duplicate_detection: bool = (
        False  # Check against existing cards (requires existing_cards)
    )

    # Performance Optimization Settings
    enable_batch_operations: bool = True  # Enable batch Anki and DB operations
    batch_size: int = 50  # Cards per batch for Anki and DB operations
    max_concurrent_generations: int = 5  # Max parallel card generations

    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model name for a specific agent.

        Args:
            agent_type: Agent type (e.g., "pre_validator", "generator", "context_enrichment")

        Returns:
            Model name (uses default_llm_model if agent-specific model is not set)
        """
        agent_model_map = {
            "pre_validator": self.pydantic_ai_pre_validator_model,
            "generator": self.pydantic_ai_generator_model,
            "post_validator": self.pydantic_ai_post_validator_model,
            "context_enrichment": self.context_enrichment_model,
            "memorization_quality": self.memorization_quality_model,
            "card_splitting": self.card_splitting_model,
            "duplicate_detection": self.duplicate_detection_model,
        }

        # Get agent-specific model, fall back to default if empty
        agent_model = agent_model_map.get(agent_type, "")
        return agent_model if agent_model else self.default_llm_model

    def validate(self) -> None:
        """Validate configuration values."""
        # Ensure vault_path is a Path object
        if isinstance(self.vault_path, str):
            if not self.vault_path:
                raise ConfigurationError(
                    "vault_path is required",
                    suggestion="Set VAULT_PATH environment variable or vault_path in config.yaml",
                )
            self.vault_path = Path(self.vault_path).expanduser().resolve()

        # Validate vault path with security checks
        validated_vault = validate_vault_path(self.vault_path, allow_symlinks=False)
        self.vault_path = validated_vault  # Update with validated path

        # Validate source directory with path traversal protection
        _ = validate_source_dir(validated_vault, self.source_dir)
        # Keep source_dir as relative for consistency
        # but we've verified it exists and is safe

        # Validate database path
        validated_db = validate_db_path(self.db_path, vault_path=validated_vault)
        self.db_path = validated_db  # Update with validated path

        # Validate LLM provider
        valid_providers = [
            "ollama",
            "lm_studio",
            "lmstudio",
            "openrouter",
            "openai",
            "anthropic",
            "claude",
        ]
        if self.llm_provider.lower() not in valid_providers:
            raise ConfigurationError(
                f"Invalid llm_provider: {self.llm_provider}. "
                f"Must be one of: {', '.join(valid_providers)}",
                suggestion=f"Set llm_provider to one of: {', '.join(valid_providers)}",
            )

        # Provider-specific API key validation
        provider_lower = self.llm_provider.lower()

        if provider_lower == "openrouter" and not self.openrouter_api_key:
            raise ConfigurationError(
                "OpenRouter API key is required when using OpenRouter provider.",
                suggestion="Set OPENROUTER_API_KEY environment variable or openrouter_api_key in config.yaml",
            )

        if provider_lower == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required when using OpenAI provider.",
                suggestion="Set OPENAI_API_KEY environment variable or openai_api_key in config.yaml. "
                "Get your API key from https://platform.openai.com/api-keys",
            )

        if provider_lower in ("anthropic", "claude") and not self.anthropic_api_key:
            raise ConfigurationError(
                "Anthropic API key is required when using Anthropic/Claude provider.",
                suggestion="Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config.yaml. "
                "Get your API key from https://console.anthropic.com/",
            )

        if self.run_mode not in ("apply", "dry-run"):
            raise ConfigurationError(
                f"Invalid run_mode: {self.run_mode}",
                suggestion="Set run_mode to either 'apply' or 'dry-run'",
            )

        if self.delete_mode not in ("delete", "archive"):
            raise ConfigurationError(
                f"Invalid delete_mode: {self.delete_mode}",
                suggestion="Set delete_mode to either 'delete' or 'archive'",
            )

        if not (0 <= self.llm_temperature <= 1):
            raise ConfigurationError(
                f"LLM temperature must be 0-1: {self.llm_temperature}",
                suggestion="Set llm_temperature to a value between 0.0 and 1.0",
            )

        if not (0 <= self.llm_top_p <= 1):
            raise ConfigurationError(
                f"LLM top_p must be 0-1: {self.llm_top_p}",
                suggestion="Set llm_top_p to a value between 0.0 and 1.0",
            )


_config: Config | None = None


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from .env and config.yaml files using pydantic-settings."""
    import yaml  # type: ignore[import-untyped]

    # Find config.yaml file
    candidate_paths: list[Path] = []

    if config_path:
        candidate_paths.append(config_path.expanduser())
    else:
        env_path = os.getenv("OBSIDIAN_ANKI_CONFIG")
        if env_path:
            candidate_paths.append(Path(env_path).expanduser())
        candidate_paths.append(Path.cwd() / "config.yaml")
        default_repo_config = Path(__file__).resolve().parents[2] / "config.yaml"
        candidate_paths.append(default_repo_config)

    resolved_config_path: Path | None = None
    for candidate in candidate_paths:
        if candidate.exists():
            resolved_config_path = candidate
            break

    # Load YAML data if file exists
    yaml_data: dict[str, Any] = {}
    if resolved_config_path:
        try:
            with open(resolved_config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            from ..utils.logging import get_logger

            logger = get_logger(__name__)
            logger.warning("yaml_config_load_failed", path=str(resolved_config_path), error=str(e))

    # Convert YAML data to environment variable format for pydantic-settings
    # pydantic-settings will automatically load from .env file via model_config
    # We'll merge YAML data by setting environment variables temporarily
    import contextlib

    @contextlib.contextmanager
    def yaml_as_env():
        """Temporarily set environment variables from YAML data."""
        # Store original env values
        original_env: dict[str, str] = {}
        try:
            # Set env vars from YAML (YAML takes precedence over .env)
            for key, value in yaml_data.items():
                env_key = key.upper()
                original_env[env_key] = os.environ.get(env_key, "")
                if value is not None:
                    # Handle special types
                    if isinstance(value, (list, dict)):
                        # For complex types, we'll handle them in the Config model
                        continue
                    elif isinstance(value, Path):
                        os.environ[env_key] = str(value)
                    else:
                        os.environ[env_key] = str(value)
            yield
        finally:
            # Restore original env values
            for key, value in original_env.items():
                if value:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    # Load config using pydantic-settings
    # Handle special cases for Path and list types from YAML
    with yaml_as_env():
        # Process source_subdirs from YAML if present
        source_subdirs: list[Path] | None = None
        if "source_subdirs" in yaml_data:
            source_subdirs_raw = yaml_data["source_subdirs"]
            if isinstance(source_subdirs_raw, list):
                source_subdirs = [Path(str(d)) for d in source_subdirs_raw]
            elif isinstance(source_subdirs_raw, str):
                source_subdirs = [Path(source_subdirs_raw)]

        # Process export_output_path from YAML if present
        export_output_path: Path | None = None
        if "export_output_path" in yaml_data:
            export_output_str = yaml_data["export_output_path"]
            if export_output_str:
                export_output_path = Path(str(export_output_str))

        # Create config instance - pydantic-settings will load from env vars
        # We pass YAML-specific values directly
        config_kwargs: dict[str, Any] = {}
        if source_subdirs is not None:
            config_kwargs["source_subdirs"] = source_subdirs
        if export_output_path is not None:
            config_kwargs["export_output_path"] = export_output_path
        if "vault_path" in yaml_data:
            config_kwargs["vault_path"] = Path(str(yaml_data["vault_path"])).expanduser().resolve()
        if "source_dir" in yaml_data:
            config_kwargs["source_dir"] = Path(str(yaml_data["source_dir"]))
        if "db_path" in yaml_data:
            config_kwargs["db_path"] = Path(str(yaml_data["db_path"]))

        config = Config(**config_kwargs)

    # Validate configuration
    config.validate()
    return config


def get_config() -> Config:
    """Get singleton config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set singleton config instance (for testing)."""
    global _config
    _config = config
