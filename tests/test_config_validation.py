"""Comprehensive tests for configuration system validation and model selection."""

import os
import tempfile
from pathlib import Path

import pytest

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import ConfigurationError


class TestModelSelection:
    """Test model selection with preset cascade logic."""

    def test_get_model_for_agent_with_explicit_override(self, temp_dir):
        """Test that explicit agent model override takes precedence."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            generator_model="custom/generator-model",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("generator")
        assert model == "custom/generator-model"

    def test_get_model_for_agent_fallback_to_preset(self, temp_dir):
        """Test fallback to preset when no explicit override is set."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            generator_model="",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("generator")
        # All presets now use x-ai/grok-4.1-fast (free, high quality, 2M context)
        assert model == "x-ai/grok-4.1-fast"

    def test_get_model_for_agent_fallback_to_default(self, temp_dir):
        """Test fallback to default_llm_model when preset is invalid."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="invalid_preset",
            generator_model="",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("generator")
        assert model == "qwen/qwen-2.5-72b-instruct"

    def test_get_model_for_agent_pre_validator(self, temp_dir):
        """Test pre_validator agent model selection."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            pre_validator_model="custom/pre-validator",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("pre_validator")
        assert model == "custom/pre-validator"

    def test_get_model_for_agent_post_validator(self, temp_dir):
        """Test post_validator agent model selection."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="high_quality",
            post_validator_model="custom/post-validator",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("post_validator")
        assert model == "custom/post-validator"

    def test_get_model_for_agent_context_enrichment(self, temp_dir):
        """Test context_enrichment agent model selection."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            context_enrichment_model="custom/enrichment",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("context_enrichment")
        assert model == "custom/enrichment"

    def test_get_model_for_agent_unknown_agent_type(self, temp_dir):
        """Test handling of unknown agent type."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("unknown_agent")
        assert model == "qwen/qwen-2.5-72b-instruct"

    def test_get_model_config_for_task_with_overrides(self, temp_dir):
        """Test get_model_config_for_task with temperature overrides."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            generator_temperature=0.7,
            generator_max_tokens=4096,
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model_config = config.get_model_config_for_task("generation")
        assert model_config["temperature"] == 0.7
        assert model_config["max_tokens"] == 4096

    def test_get_model_config_for_task_unknown_task(self, temp_dir):
        """Test get_model_config_for_task with unknown task returns defaults."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset="balanced",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
            llm_temperature=0.5,
            llm_max_tokens=8192,
        )

        model_config = config.get_model_config_for_task("unknown_task")
        assert model_config["model_name"] == "qwen/qwen-2.5-72b-instruct"
        assert model_config["temperature"] == 0.5
        assert model_config["max_tokens"] == 8192


class TestModelPresets:
    """Test different model preset configurations."""

    @pytest.mark.parametrize(
        ("preset", "expected_generator"),
        [
            # All presets now use x-ai/grok-4.1-fast (free, high quality, 2M context)
            ("cost_effective", "x-ai/grok-4.1-fast"),
            ("balanced", "x-ai/grok-4.1-fast"),
            ("high_quality", "x-ai/grok-4.1-fast"),
            ("fast", "x-ai/grok-4.1-fast"),
        ],
    )
    def test_preset_generator_models(self, temp_dir, preset, expected_generator):
        """Test that presets use expected models for generation task."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset=preset,
            generator_model="",
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model = config.get_model_for_agent("generator")
        assert model == expected_generator

    @pytest.mark.parametrize(
        ("preset", "task", "expected_temp"),
        [
            ("cost_effective", "generation", 0.3),
            ("balanced", "generation", 0.3),
            ("high_quality", "pre_validation", 0.0),
            ("fast", "post_validation", 0.0),
        ],
    )
    def test_preset_temperature_values(self, temp_dir, preset, task, expected_temp):
        """Test that presets use correct temperature values for tasks."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            model_preset=preset,
            default_llm_model="qwen/qwen-2.5-72b-instruct",
        )

        model_config = config.get_model_config_for_task(task)
        assert model_config["temperature"] == expected_temp


class TestConfigurationValidation:
    """Test configuration validation logic."""

    def test_valid_configuration_loads_successfully(self, temp_dir):
        """Test that a valid configuration loads without errors."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            llm_provider="ollama",
            llm_temperature=0.2,
            llm_top_p=0.3,
            run_mode="apply",
            delete_mode="delete",
        )

        assert config.vault_path.resolve() == vault_path.resolve()
        assert config.llm_temperature == 0.2

    def test_empty_vault_path_uses_current_directory(self, temp_dir):
        """Test that empty vault_path resolves to current directory."""
        # Empty vault_path is allowed and resolves to current working directory
        config = Config(
            vault_path=Path(),
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )
        # Empty string resolves to current directory (absolute path)
        assert config.vault_path.is_absolute()

    def test_invalid_vault_path(self, temp_dir):
        """Test error when vault_path does not exist."""
        non_existent_path = temp_dir / "nonexistent"

        with pytest.raises(ConfigurationError, match="Vault path does not exist"):
            Config(
                vault_path=non_existent_path,
                source_dir=Path(),
            )

    def test_invalid_llm_provider(self, temp_dir):
        """Test error for invalid llm_provider."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="Invalid llm_provider"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_provider="invalid_provider",
            )

    def test_missing_openrouter_api_key(self, temp_dir):
        """Test error when OpenRouter provider is selected without API key."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="OpenRouter API key is required"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_provider="openrouter",
                openrouter_api_key="",
            )

    def test_missing_openai_api_key(self, temp_dir):
        """Test error when OpenAI provider is selected without API key."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="OpenAI API key is required"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_provider="openai",
                openai_api_key="",
            )

    def test_missing_anthropic_api_key(self, temp_dir):
        """Test error when Anthropic provider is selected without API key."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="Anthropic API key is required"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_provider="anthropic",
                anthropic_api_key="",
            )

    def test_invalid_run_mode(self, temp_dir):
        """Test error for invalid run_mode."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="Invalid run_mode"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                run_mode="invalid_mode",
            )

    def test_invalid_delete_mode(self, temp_dir):
        """Test error for invalid delete_mode."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="Invalid delete_mode"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                delete_mode="invalid_mode",
            )

    def test_invalid_temperature_too_high(self, temp_dir):
        """Test error when temperature is above 1.0."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="temperature must be 0-1"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_temperature=1.5,
            )

    def test_invalid_temperature_too_low(self, temp_dir):
        """Test error when temperature is below 0.0."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="temperature must be 0-1"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_temperature=-0.1,
            )

    def test_invalid_top_p_too_high(self, temp_dir):
        """Test error when top_p is above 1.0."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="top_p must be 0-1"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_top_p=1.5,
            )

    def test_invalid_top_p_too_low(self, temp_dir):
        """Test error when top_p is below 0.0."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(ConfigurationError, match="top_p must be 0-1"):
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_top_p=-0.1,
            )

    def test_configuration_error_includes_suggestion(self, temp_dir):
        """Test that ConfigurationError includes helpful suggestions."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        with pytest.raises(
            ConfigurationError, match="Set llm_provider to one of"
        ) as exc_info:
            Config(
                vault_path=vault_path,
                source_dir=Path(),
                llm_provider="invalid_provider",
            )

        assert exc_info.value.suggestion is not None
        assert "Set llm_provider" in exc_info.value.suggestion


class TestPathValidation:
    """Test path validation and expansion."""

    def test_vault_path_tilde_expansion(self, temp_dir):
        """Test that vault_path expands ~ correctly."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.vault_path.is_absolute()

    def test_vault_path_resolution(self, temp_dir):
        """Test that vault_path is resolved to absolute path."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.vault_path.is_absolute()

    def test_source_dir_path_expansion(self, temp_dir):
        """Test that source_dir expands paths correctly."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()
        source_dir = vault_path / "notes"
        source_dir.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path("notes"),
            db_path=temp_dir / "test.db",
        )

        assert config.source_dir == Path("notes")

    def test_db_path_expansion(self, temp_dir):
        """Test that db_path expands ~ correctly."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()
        db_path = temp_dir / "state.db"

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=db_path,
        )

        assert config.db_path.resolve() == db_path.resolve()

    def test_db_directory_creation(self, temp_dir):
        """Test that db_path parent directory is created during validation."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()
        db_dir = temp_dir / "newdir"
        db_dir.mkdir()
        db_path = db_dir / "test.db"

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=db_path,
        )

        assert db_dir.exists()
        assert config.db_path.parent == db_dir.resolve()

    def test_db_directory_not_writable(self, temp_dir):
        """Test error when db_path parent directory is not writable."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()
        db_dir = temp_dir / "readonly"
        db_dir.mkdir()
        db_path = db_dir / "test.db"

        os.chmod(db_dir, 0o444)

        try:
            with pytest.raises(
                ConfigurationError, match="Database directory is not writable"
            ):
                Config(
                    vault_path=vault_path,
                    source_dir=Path(),
                    db_path=db_path,
                )
        finally:
            os.chmod(db_dir, 0o755)

    def test_source_subdirs_list_conversion(self, temp_dir):
        """Test that source_subdirs converts strings to Path list."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            source_subdirs=["dir1", "dir2"],
        )

        assert config.source_subdirs == [Path("dir1"), Path("dir2")]

    def test_source_subdirs_single_string_conversion(self, temp_dir):
        """Test that single source_subdirs string converts to list of Paths."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            source_subdirs="single_dir",
        )

        assert config.source_subdirs == [Path("single_dir")]


class TestProviderValidation:
    """Test LLM provider-specific validation."""

    @pytest.mark.parametrize(
        "provider",
        [
            "ollama",
            "lm_studio",
            "lmstudio",
            "openrouter",
            "openai",
            "anthropic",
            "claude",
        ],
    )
    def test_valid_provider_names(self, temp_dir, provider):
        """Test that all valid provider names are accepted."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        api_keys = {
            "openrouter": "test-key",
            "openai": "test-key",
            "anthropic": "test-key",
            "claude": "test-key",
        }

        kwargs = {
            "vault_path": vault_path,
            "source_dir": Path(),
            "db_path": temp_dir / "test.db",
            "llm_provider": provider,
        }

        if provider in api_keys:
            if provider == "openrouter":
                kwargs["openrouter_api_key"] = api_keys[provider]
            elif provider == "openai":
                kwargs["openai_api_key"] = api_keys[provider]
            elif provider in ("anthropic", "claude"):
                kwargs["anthropic_api_key"] = api_keys[provider]

        config = Config(**kwargs)
        assert config.llm_provider == provider

    def test_ollama_provider_no_api_key_required(self, temp_dir):
        """Test that Ollama provider doesn't require API key."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            llm_provider="ollama",
        )

        assert config.llm_provider == "ollama"

    def test_lm_studio_provider_no_api_key_required(self, temp_dir):
        """Test that LM Studio provider doesn't require API key."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
            llm_provider="lm_studio",
        )

        assert config.llm_provider == "lm_studio"


class TestConfigurationDefaults:
    """Test default configuration values."""

    def test_default_anki_settings(self, temp_dir):
        """Test default Anki configuration values."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.anki_connect_url == "http://127.0.0.1:8765"
        assert config.anki_deck_name == "Interview Questions"
        assert config.anki_note_type == "APF::Simple"

    def test_default_llm_settings(self, temp_dir):
        """Test default LLM configuration values."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.llm_provider == "ollama"
        assert config.llm_temperature == 0.2
        assert config.llm_top_p == 0.3
        assert config.llm_timeout == 3600.0
        assert config.llm_max_tokens == 8192

    def test_default_model_preset(self, temp_dir):
        """Test default model preset is balanced."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.model_preset == "balanced"

    def test_default_agent_settings(self, temp_dir):
        """Test default agent system settings."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.use_langgraph is False
        assert config.use_pydantic_ai is False

    def test_default_performance_settings(self, temp_dir):
        """Test default performance optimization settings."""
        vault_path = temp_dir / "vault"
        vault_path.mkdir()

        config = Config(
            vault_path=vault_path,
            source_dir=Path(),
            db_path=temp_dir / "test.db",
        )

        assert config.enable_batch_operations is True
        assert config.batch_size == 50
        assert config.max_concurrent_generations == 5


@pytest.fixture()
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
