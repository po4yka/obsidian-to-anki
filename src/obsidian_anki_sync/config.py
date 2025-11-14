"""Configuration management for the sync service."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

from .exceptions import ConfigurationError
from .utils.path_validator import (
    validate_db_path,
    validate_source_dir,
    validate_vault_path,
)


@dataclass
class Config:
    """Service configuration."""

    # Required fields (no defaults) - MUST come first in dataclasses
    # Obsidian paths
    vault_path: Path
    source_dir: Path

    # Anki settings
    anki_connect_url: str
    anki_deck_name: str
    anki_note_type: str

    # Runtime settings
    run_mode: str  # 'apply' or 'dry-run'
    delete_mode: str  # 'delete' or 'archive'

    # Database
    db_path: Path

    # Logging
    log_level: str

    # Optional fields (with defaults) - MUST come after required fields
    # Obsidian source directories (optional - overrides source_dir if provided)
    # List of relative paths from vault_path to search for Q&A notes
    # Example: [".", "Interviews", "CS/Algorithms"]
    source_subdirs: list[Path] | None = None
    # LLM Provider Configuration
    # Unified provider system - choose one: 'ollama', 'lm_studio', 'openrouter'
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 900.0  # 15 minutes default for large models
    llm_max_tokens: int = 2048
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
    # Examples: "openrouter/polaris-alpha", "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"
    default_llm_model: str = "openrouter/polaris-alpha"

    # Individual agent model overrides (optional - leave empty to use default_llm_model)
    # Only set these if you want a specific agent to use a different model
    pydantic_ai_pre_validator_model: str = ""  # Empty = use default_llm_model
    pydantic_ai_generator_model: str = ""  # Empty = use default_llm_model
    pydantic_ai_post_validator_model: str = ""  # Empty = use default_llm_model
    context_enrichment_model: str = ""  # Empty = use default_llm_model
    memorization_quality_model: str = ""  # Empty = use default_llm_model
    card_splitting_model: str = ""  # Empty = use default_llm_model
    duplicate_detection_model: str = ""  # Empty = use default_llm_model

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
    """Load configuration from .env and config.yaml files."""
    # Load .env
    load_dotenv()

    # Load config.yaml if exists
    config_data: dict[str, Any] = {}
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

    if resolved_config_path:
        with open(resolved_config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # Helper to get string from config/env
    def get_str(key: str, default: str) -> str:
        val = config_data.get(key) or os.getenv(key.upper())
        return str(val) if val is not None else default

    # Helper to get bool from config/env
    def get_bool(key: str, default: bool) -> bool:
        val = config_data.get(key) or os.getenv(key.upper())
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes", "on")

    # Helper to get int from config/env
    def get_int(key: str, default: int) -> int:
        val = config_data.get(key) or os.getenv(key.upper())
        return int(val) if val is not None else default

    # Helper to get float from config/env
    def get_float(key: str, default: float) -> float:
        val = config_data.get(key) or os.getenv(key.upper())
        return float(val) if val is not None else default

    # Build config from environment and file
    deck_name = get_str("anki_deck_name", "Interview Questions")

    # Get vault and source paths
    vault_path_str = get_str("vault_path", "")
    source_dir_str = get_str("source_dir", ".")

    # Get source subdirs if configured (overrides source_dir)
    source_subdirs_raw = config_data.get("source_subdirs")
    source_subdirs: list[Path] | None = None
    if source_subdirs_raw is not None:
        if isinstance(source_subdirs_raw, list):
            source_subdirs = [Path(str(d)) for d in source_subdirs_raw]
        elif isinstance(source_subdirs_raw, str):
            # Support single string (convert to list)
            source_subdirs = [Path(source_subdirs_raw)]

    # Get export output path if configured
    export_output_str = config_data.get("export_output_path") or os.getenv(
        "EXPORT_OUTPUT_PATH"
    )
    export_output_path: Path | None = (
        Path(export_output_str) if export_output_str else None
    )

    config = Config(
        vault_path=Path(vault_path_str).expanduser().resolve(),
        source_dir=Path(source_dir_str),
        source_subdirs=source_subdirs,
        anki_connect_url=get_str("anki_connect_url", "http://127.0.0.1:8765"),
        anki_deck_name=deck_name,
        anki_note_type=get_str("anki_note_type", "APF::Simple"),
        # Deck export settings
        export_deck_name=get_str("export_deck_name", deck_name),
        export_deck_description=get_str("export_deck_description", ""),
        export_output_path=export_output_path,
        # LLM Provider Configuration
        llm_provider=get_str("llm_provider", "ollama"),
        # Common LLM settings
        llm_temperature=get_float("llm_temperature", 0.2),
        llm_top_p=get_float("llm_top_p", 0.3),
        llm_timeout=get_float(
            "llm_timeout", 900.0
        ),  # 15 minutes default for large models
        llm_max_tokens=get_int("llm_max_tokens", 2048),
        # Ollama settings
        ollama_base_url=get_str("ollama_base_url", "http://localhost:11434"),
        ollama_api_key=config_data.get("ollama_api_key") or os.getenv("OLLAMA_API_KEY"),
        # LM Studio settings
        lm_studio_base_url=get_str("lm_studio_base_url", "http://localhost:1234/v1"),
        # OpenAI settings
        openai_api_key=str(
            config_data.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or ""
        ),
        openai_base_url=get_str("openai_base_url", "https://api.openai.com/v1"),
        openai_organization=config_data.get("openai_organization")
        or os.getenv("OPENAI_ORGANIZATION"),
        openai_max_retries=get_int("openai_max_retries", 3),
        # Anthropic settings
        anthropic_api_key=str(
            config_data.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY") or ""
        ),
        anthropic_base_url=get_str("anthropic_base_url", "https://api.anthropic.com"),
        anthropic_api_version=get_str("anthropic_api_version", "2023-06-01"),
        anthropic_max_retries=get_int("anthropic_max_retries", 3),
        # OpenRouter settings
        openrouter_api_key=str(
            config_data.get("openrouter_api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or ""
        ),
        openrouter_base_url=get_str(
            "openrouter_base_url", "https://openrouter.ai/api/v1"
        ),
        openrouter_site_url=config_data.get("openrouter_site_url")
        or os.getenv("OPENROUTER_SITE_URL"),
        openrouter_site_name=config_data.get("openrouter_site_name")
        or os.getenv("OPENROUTER_SITE_NAME"),
        openrouter_model=get_str("openrouter_model", "openai/gpt-4"),
        run_mode=get_str("run_mode", "apply"),
        delete_mode=get_str("delete_mode", "delete"),
        db_path=Path(get_str("db_path", ".sync_state.db")),
        log_level=get_str("log_level", "INFO"),
        # Agent system settings
        use_agent_system=get_bool("use_agent_system", False),
        qa_extractor_model=get_str("qa_extractor_model", "qwen3:8b"),
        qa_extractor_temperature=get_float("qa_extractor_temperature", 0.0),
        pre_validator_model=get_str("pre_validator_model", "qwen3:8b"),
        pre_validator_temperature=get_float("pre_validator_temperature", 0.0),
        pre_validation_enabled=get_bool("pre_validation_enabled", True),
        generator_model=get_str("generator_model", "qwen3:32b"),
        generator_temperature=get_float("generator_temperature", 0.3),
        post_validator_model=get_str("post_validator_model", "qwen3:14b"),
        post_validator_temperature=get_float("post_validator_temperature", 0.0),
        post_validation_max_retries=get_int("post_validation_max_retries", 3),
        post_validation_auto_fix=get_bool("post_validation_auto_fix", True),
        post_validation_strict_mode=get_bool("post_validation_strict_mode", True),
        # LangGraph + PydanticAI settings
        use_langgraph=get_bool("use_langgraph", False),
        use_pydantic_ai=get_bool("use_pydantic_ai", False),
        pydantic_ai_pre_validator_model=get_str(
            "pydantic_ai_pre_validator_model", "openai/gpt-4o-mini"
        ),
        pydantic_ai_generator_model=get_str(
            "pydantic_ai_generator_model", "anthropic/claude-3-5-sonnet"
        ),
        pydantic_ai_post_validator_model=get_str(
            "pydantic_ai_post_validator_model", "openai/gpt-4o-mini"
        ),
        langgraph_max_retries=get_int("langgraph_max_retries", 3),
        langgraph_auto_fix=get_bool("langgraph_auto_fix", True),
        langgraph_strict_mode=get_bool("langgraph_strict_mode", True),
        langgraph_checkpoint_enabled=get_bool("langgraph_checkpoint_enabled", True),
    )

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
