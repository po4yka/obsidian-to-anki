"""Configuration management for the sync service."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """Service configuration."""

    # Obsidian paths
    vault_path: Path
    source_dir: Path

    # Anki settings
    anki_connect_url: str
    anki_deck_name: str
    anki_note_type: str

    # LLM Provider Configuration
    # Unified provider system - choose one: 'ollama', 'lm_studio', 'openrouter'
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 120.0
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

    # Runtime settings
    run_mode: str  # 'apply' or 'dry-run'
    delete_mode: str  # 'delete' or 'archive'

    # Database
    db_path: Path

    # Logging
    log_level: str

    # Deck export settings (for .apkg generation) - optional with defaults
    export_deck_name: str | None = None
    export_deck_description: str = ""
    export_output_path: Path | None = None

    # Agent system settings (optional, defaults provided)
    use_agent_system: bool = False
    agent_execution_mode: str = "parallel"  # 'parallel' or 'sequential'

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
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        full_source = self.vault_path / self.source_dir
        if not full_source.exists():
            raise ValueError(f"Source directory does not exist: {full_source}")

        # Validate LLM provider
        valid_providers = ["ollama", "lm_studio", "lmstudio", "openrouter"]
        if self.llm_provider.lower() not in valid_providers:
            raise ValueError(
                f"Invalid llm_provider: {self.llm_provider}. "
                f"Must be one of: {', '.join(valid_providers)}"
            )

        # Provider-specific validation
        if self.llm_provider.lower() == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required when using OpenRouter provider. "
                "Set OPENROUTER_API_KEY environment variable or openrouter_api_key in config."
            )

        # Legacy: OpenRouter API key only required if NOT using agent system (for backward compatibility)
        if not self.use_agent_system and self.llm_provider.lower() == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when not using agent system"
            )

        if self.run_mode not in ("apply", "dry-run"):
            raise ValueError(f"Invalid run_mode: {self.run_mode}")

        if self.delete_mode not in ("delete", "archive"):
            raise ValueError(f"Invalid delete_mode: {self.delete_mode}")

        if not (0 <= self.llm_temperature <= 1):
            raise ValueError(f"LLM temperature must be 0-1: {self.llm_temperature}")

        if not (0 <= self.llm_top_p <= 1):
            raise ValueError(f"LLM top_p must be 0-1: {self.llm_top_p}")


_config: Config | None = None


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from .env and config.yaml files."""
    # Load .env
    load_dotenv()

    # Load config.yaml if exists
    config_data = {}
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

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
    deck_name = config_data.get("anki_deck_name") or os.getenv(
        "ANKI_DECK_NAME", "Interview Questions"
    )

    config = Config(
        vault_path=Path(config_data.get("vault_path") or os.getenv("VAULT_PATH", ""))
        .expanduser()
        .resolve(),
        source_dir=Path(
            config_data.get("source_dir")
            or os.getenv("SOURCE_DIR", "interview_questions/InterviewQuestions")
        ),
        anki_connect_url=config_data.get("anki_connect_url")
        or os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765"),
        anki_deck_name=deck_name,
        anki_note_type=config_data.get("anki_note_type")
        or os.getenv("ANKI_NOTE_TYPE", "APF::Simple"),
        # Deck export settings
        export_deck_name=config_data.get("export_deck_name")
        or os.getenv("EXPORT_DECK_NAME")
        or deck_name,
        export_deck_description=config_data.get("export_deck_description")
        or os.getenv("EXPORT_DECK_DESCRIPTION", ""),
        export_output_path=(
            Path(
                config_data.get("export_output_path")
                or os.getenv("EXPORT_OUTPUT_PATH", "output.apkg")
            )
            if config_data.get("export_output_path") or os.getenv("EXPORT_OUTPUT_PATH")
            else None
        ),
        # LLM Provider Configuration
        llm_provider=config_data.get("llm_provider")
        or os.getenv("LLM_PROVIDER", "ollama"),
        # Common LLM settings
        llm_temperature=get_float("llm_temperature", 0.2),
        llm_top_p=get_float("llm_top_p", 0.3),
        llm_timeout=get_float("llm_timeout", 120.0),
        llm_max_tokens=get_int("llm_max_tokens", 2048),
        # Ollama settings
        ollama_base_url=config_data.get("ollama_base_url")
        or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_api_key=config_data.get("ollama_api_key")
        or os.getenv("OLLAMA_API_KEY"),
        # LM Studio settings
        lm_studio_base_url=config_data.get("lm_studio_base_url")
        or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        # OpenRouter settings
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=config_data.get("openrouter_base_url")
        or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_site_url=config_data.get("openrouter_site_url")
        or os.getenv("OPENROUTER_SITE_URL"),
        openrouter_site_name=config_data.get("openrouter_site_name")
        or os.getenv("OPENROUTER_SITE_NAME"),
        openrouter_model=config_data.get("openrouter_model")
        or os.getenv("OPENROUTER_MODEL", "openai/gpt-4"),
        run_mode=config_data.get("run_mode") or os.getenv("RUN_MODE", "apply"),
        delete_mode=config_data.get("delete_mode")
        or os.getenv("DELETE_MODE", "delete"),
        db_path=Path(
            config_data.get("db_path") or os.getenv("DB_PATH", ".sync_state.db")
        ),
        log_level=config_data.get("log_level") or os.getenv("LOG_LEVEL", "INFO"),
        # Agent system settings
        use_agent_system=get_bool("use_agent_system", False),
        agent_execution_mode=config_data.get("agent_execution_mode")
        or os.getenv("AGENT_EXECUTION_MODE", "parallel"),
        pre_validator_model=config_data.get("pre_validator_model")
        or os.getenv("PRE_VALIDATOR_MODEL", "qwen3:8b"),
        pre_validator_temperature=get_float("pre_validator_temperature", 0.0),
        pre_validation_enabled=get_bool("pre_validation_enabled", True),
        generator_model=config_data.get("generator_model")
        or os.getenv("GENERATOR_MODEL", "qwen3:32b"),
        generator_temperature=get_float("generator_temperature", 0.3),
        post_validator_model=config_data.get("post_validator_model")
        or os.getenv("POST_VALIDATOR_MODEL", "qwen3:14b"),
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
