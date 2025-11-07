"""Configuration management for the sync service."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigurationError


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
    # LLM Provider Configuration
    # Unified provider system - choose one: 'ollama', 'lm_studio', 'openrouter'
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 900.0  # 15 minutes for large models
    llm_max_tokens: int = 2048

    # Ollama provider settings (local or cloud)
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None  # Only for Ollama Cloud

    # LM Studio provider settings
    lm_studio_base_url: str = "http://localhost:1234/v1"

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

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.vault_path.exists():
            raise ConfigurationError(
                f"Vault path does not exist: {self.vault_path}",
                suggestion="Set VAULT_PATH environment variable or vault_path in config.yaml to a valid directory",
            )

        full_source = self.vault_path / self.source_dir
        if not full_source.exists():
            raise ConfigurationError(
                f"Source directory does not exist: {full_source}",
                suggestion=f"Create the directory '{self.source_dir}' in your vault or update source_dir in config.yaml",
            )

        # Validate LLM provider
        valid_providers = ["ollama", "lm_studio", "lmstudio", "openrouter"]
        if self.llm_provider.lower() not in valid_providers:
            raise ConfigurationError(
                f"Invalid llm_provider: {self.llm_provider}. "
                f"Must be one of: {', '.join(valid_providers)}",
                suggestion=f"Set llm_provider to one of: {', '.join(valid_providers)}",
            )

        # Provider-specific validation
        if self.llm_provider.lower() == "openrouter" and not self.openrouter_api_key:
            raise ConfigurationError(
                "OpenRouter API key is required when using OpenRouter provider.",
                suggestion="Set OPENROUTER_API_KEY environment variable or openrouter_api_key in config.yaml",
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
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
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
    source_dir_str = get_str("source_dir", "interview_questions/InterviewQuestions")

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
        llm_timeout=get_float("llm_timeout", 120.0),
        llm_max_tokens=get_int("llm_max_tokens", 2048),
        # Ollama settings
        ollama_base_url=get_str("ollama_base_url", "http://localhost:11434"),
        ollama_api_key=config_data.get("ollama_api_key") or os.getenv("OLLAMA_API_KEY"),
        # LM Studio settings
        lm_studio_base_url=get_str("lm_studio_base_url", "http://localhost:1234/v1"),
        # OpenRouter settings
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
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
