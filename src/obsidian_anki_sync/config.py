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

    # LLM settings
    openrouter_api_key: str
    openrouter_model: str
    llm_temperature: float
    llm_top_p: float

    # Runtime settings
    run_mode: str  # 'apply' or 'dry-run'
    delete_mode: str  # 'delete' or 'archive'

    # Database
    db_path: Path

    # Logging
    log_level: str

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        full_source = self.vault_path / self.source_dir
        if not full_source.exists():
            raise ValueError(f"Source directory does not exist: {full_source}")

        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

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

    # Build config from environment and file
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
        anki_deck_name=config_data.get("anki_deck_name")
        or os.getenv("ANKI_DECK_NAME", "Interview Questions"),
        anki_note_type=config_data.get("anki_note_type")
        or os.getenv("ANKI_NOTE_TYPE", "APF::Simple"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=config_data.get("openrouter_model")
        or os.getenv("OPENROUTER_MODEL", "openai/gpt-4"),
        llm_temperature=float(
            config_data.get("llm_temperature") or os.getenv("LLM_TEMPERATURE", "0.2")
        ),
        llm_top_p=float(config_data.get("llm_top_p") or os.getenv("LLM_TOP_P", "0.3")),
        run_mode=config_data.get("run_mode") or os.getenv("RUN_MODE", "apply"),
        delete_mode=config_data.get("delete_mode")
        or os.getenv("DELETE_MODE", "delete"),
        db_path=Path(
            config_data.get("db_path") or os.getenv("DB_PATH", ".sync_state.db")
        ),
        log_level=config_data.get("log_level") or os.getenv("LOG_LEVEL", "INFO"),
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
