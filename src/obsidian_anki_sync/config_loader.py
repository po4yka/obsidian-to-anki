"""Config loader utilities (split from config.py)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from .config_settings import Config
from .exceptions import ConfigurationError
from .utils.logging import get_logger

_config: Config | None = None


def load_config(
    config_path: Path | None = None, *, strict_config: bool = True
) -> Config:
    """Load configuration from .env and config.yaml files using pydantic-settings."""
    import yaml

    logger = get_logger(__name__)

    candidate_paths: list[Path] = []

    if config_path:
        candidate_paths.append(config_path.expanduser())
        logger.info(
            "config_loading", config_path=str(config_path), source="cli_argument"
        )
    else:
        env_path = os.getenv("OBSIDIAN_ANKI_CONFIG")
        if env_path:
            candidate_paths.append(Path(env_path).expanduser())
            logger.debug(
                "config_searching", source="environment_variable", path=env_path
            )
        candidate_paths.append(Path.cwd() / "config.yaml")
        default_repo_config = Path(__file__).resolve().parents[2] / "config.yaml"
        candidate_paths.append(default_repo_config)
        logger.debug(
            "config_searching",
            source="default_locations",
            paths=[str(p) for p in candidate_paths],
        )

    resolved_config_path: Path | None = None
    for candidate in candidate_paths:
        if candidate.exists():
            resolved_config_path = candidate
            logger.info("config_file_found", config_path=str(resolved_config_path))
            break

    if not resolved_config_path:
        logger.warning(
            "config_file_not_found", searched_paths=[str(p) for p in candidate_paths]
        )

    yaml_data: dict[str, Any] = {}
    if resolved_config_path:
        try:
            with open(resolved_config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            logger.debug(
                "config_yaml_loaded",
                config_path=str(resolved_config_path),
                keys_count=len(yaml_data),
            )
        except Exception as e:
            logger.error(
                "config_yaml_load_error",
                config_path=str(resolved_config_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            if strict_config:
                msg = f"Failed to parse config file: {resolved_config_path}"
                suggestion = (
                    "Check YAML syntax (indentation, colons, quotes). "
                    "Validate file encoding is UTF-8. "
                    f"Original error: {e}"
                )
                raise ConfigurationError(msg, suggestion=suggestion) from e

    @contextlib.contextmanager
    def yaml_as_env():
        """Temporarily set environment variables from YAML data."""
        original_env: dict[str, str] = {}
        try:
            for key, value in yaml_data.items():
                env_key = key.upper()
                original_env[env_key] = os.environ.get(env_key, "")
                if value is not None:
                    if isinstance(value, (list, dict)):
                        continue
                    if isinstance(value, Path):
                        os.environ[env_key] = str(value)
                    else:
                        os.environ[env_key] = str(value)
            yield
        finally:
            for key, value in original_env.items():
                if value:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    with yaml_as_env():
        source_subdirs: list[Path] | None = None
        if "source_subdirs" in yaml_data:
            source_subdirs_raw = yaml_data["source_subdirs"]
            if isinstance(source_subdirs_raw, list):
                source_subdirs = [Path(str(d)) for d in source_subdirs_raw]
            elif isinstance(source_subdirs_raw, str):
                source_subdirs = [Path(source_subdirs_raw)]

        export_output_path: Path | None = None
        if "export_output_path" in yaml_data:
            export_output_str = yaml_data["export_output_path"]
            if export_output_str:
                export_output_path = Path(str(export_output_str))

        config_kwargs: dict[str, Any] = {}
        if "model_overrides" in yaml_data:
            config_kwargs["model_overrides"] = yaml_data.get("model_overrides", {})
        if source_subdirs is not None:
            config_kwargs["source_subdirs"] = source_subdirs
        if export_output_path is not None:
            config_kwargs["export_output_path"] = export_output_path
        if "vault_path" in yaml_data:
            config_kwargs["vault_path"] = (
                Path(str(yaml_data["vault_path"])).expanduser().resolve()
            )
        if "source_dir" in yaml_data:
            config_kwargs["source_dir"] = Path(str(yaml_data["source_dir"]))
        if "db_path" in yaml_data:
            config_kwargs["db_path"] = Path(str(yaml_data["db_path"]))
        if "data_dir" in yaml_data:
            config_kwargs["data_dir"] = (
                Path(str(yaml_data["data_dir"])).expanduser().resolve()
            )

        try:
            config = Config(**config_kwargs)
            logger.info(
                "config_loaded",
                vault_path=str(config.vault_path) if config.vault_path else None,
                llm_provider=getattr(config, "llm_provider", None),
            )
        except Exception as e:
            logger.error(
                "config_validation_error",
                error=str(e),
                error_type=type(e).__name__,
                config_path=str(resolved_config_path) if resolved_config_path else None,
            )
            raise

    try:
        config.validate_config()
        logger.debug("config_validation_passed")
    except Exception as e:
        effective_strict = strict_config and config.strict_mode
        if effective_strict:
            logger.error(
                "config_validation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        logger.warning(
            "config_validation_warning",
            error=str(e),
            error_type=type(e).__name__,
        )

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


def reset_config() -> None:
    """Reset global config instance (for testing only)."""
    global _config
    _config = None


__all__ = ["Config", "get_config", "load_config", "reset_config", "set_config"]

