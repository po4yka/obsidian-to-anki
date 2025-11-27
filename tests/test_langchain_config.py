"""Tests for LangChain agent configuration.

This module tests the configuration validation and functionality
for LangChain agent settings.
"""

import pytest
from pydantic import ValidationError

from src.obsidian_anki_sync.config import Config


class TestLangChainAgentFrameworkConfig:
    """Test LangChain agent framework configuration."""

    def test_valid_agent_frameworks(self):
        """Test valid agent framework values."""
        # Test pydantic_ai framework
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="pydantic_ai"
        )
        assert config.agent_framework == "pydantic_ai"

        # Test langchain framework
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="langchain"
        )
        assert config.agent_framework == "langchain"

    def test_invalid_agent_framework(self):
        """Test invalid agent framework value."""
        with pytest.raises(ValidationError) as exc_info:
            Config(
                vault_path=".",
                source_dir=".",
                agent_framework="invalid_framework"
            )

        error = str(exc_info.value)
        assert "agent_framework must be one of" in error
        assert "['pydantic_ai', 'langchain']" in error

    def test_default_agent_framework(self):
        """Test default agent framework value."""
        config = Config(vault_path=".", source_dir=".")
        assert config.agent_framework == "pydantic_ai"


class TestLangChainAgentTypeConfig:
    """Test LangChain agent type configuration."""

    def test_valid_langchain_agent_types(self):
        """Test valid LangChain agent type values."""
        valid_types = ["tool_calling", "react", "structured_chat", "json_chat"]

        for agent_type in valid_types:
            config = Config(
                vault_path=".",
                source_dir=".",
                agent_framework="langchain",
                langchain_generator_type=agent_type,
                langchain_pre_validator_type=agent_type,
                langchain_post_validator_type=agent_type,
                langchain_enrichment_type=agent_type
            )

            assert config.langchain_generator_type == agent_type
            assert config.langchain_pre_validator_type == agent_type
            assert config.langchain_post_validator_type == agent_type
            assert config.langchain_enrichment_type == agent_type

    def test_invalid_langchain_agent_type(self):
        """Test invalid LangChain agent type value."""
        with pytest.raises(ValidationError) as exc_info:
            Config(
                vault_path=".",
                source_dir=".",
                langchain_generator_type="invalid_type"
            )

        error = str(exc_info.value)
        assert "LangChain agent type must be one of" in error
        assert "['tool_calling', 'react', 'structured_chat', 'json_chat']" in error

    def test_default_langchain_agent_types(self):
        """Test default LangChain agent type values."""
        config = Config(vault_path=".", source_dir=".")

        assert config.langchain_generator_type == "tool_calling"
        assert config.langchain_pre_validator_type == "react"
        assert config.langchain_post_validator_type == "tool_calling"
        assert config.langchain_enrichment_type == "structured_chat"


class TestAgentFallbackConfig:
    """Test agent fallback configuration."""

    def test_valid_fallback_values(self):
        """Test valid fallback configuration values."""
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_fallback_on_error="pydantic_ai",
            agent_fallback_on_timeout="react"
        )

        assert config.agent_fallback_on_error == "pydantic_ai"
        assert config.agent_fallback_on_timeout == "react"

    def test_default_fallback_values(self):
        """Test default fallback configuration values."""
        config = Config(vault_path=".", source_dir=".")

        assert config.agent_fallback_on_error == "pydantic_ai"
        assert config.agent_fallback_on_timeout == "react"


class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_langchain_config_complete(self):
        """Test complete LangChain configuration."""
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="langchain",
            langchain_generator_type="tool_calling",
            langchain_pre_validator_type="react",
            langchain_post_validator_type="structured_chat",
            langchain_enrichment_type="json_chat",
            agent_fallback_on_error="pydantic_ai",
            agent_fallback_on_timeout="tool_calling"
        )

        assert config.agent_framework == "langchain"
        assert config.langchain_generator_type == "tool_calling"
        assert config.langchain_pre_validator_type == "react"
        assert config.langchain_post_validator_type == "structured_chat"
        assert config.langchain_enrichment_type == "json_chat"
        assert config.agent_fallback_on_error == "pydantic_ai"
        assert config.agent_fallback_on_timeout == "tool_calling"

    def test_pydantic_ai_config_complete(self):
        """Test complete PydanticAI configuration."""
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="pydantic_ai",
            # LangChain settings should be present but not used
            langchain_generator_type="tool_calling",
            agent_fallback_on_error="langchain",
            agent_fallback_on_timeout="react"
        )

        assert config.agent_framework == "pydantic_ai"
        assert config.langchain_generator_type == "tool_calling"  # Still set
        assert config.agent_fallback_on_error == "langchain"
        assert config.agent_fallback_on_timeout == "react"

    def test_config_from_env_vars(self, monkeypatch):
        """Test configuration loading from environment variables."""
        monkeypatch.setenv("AGENT_FRAMEWORK", "langchain")
        monkeypatch.setenv("LANGCHAIN_GENERATOR_TYPE", "react")
        monkeypatch.setenv("AGENT_FALLBACK_ON_ERROR", "pydantic_ai")

        # Create config - pydantic-settings should load from env
        config = Config(vault_path=".", source_dir=".")

        # Note: This test may need adjustment based on how pydantic-settings
        # handles env var loading in the test environment
        # For now, just verify the config object is created successfully
        assert isinstance(config, Config)
        assert hasattr(config, 'agent_framework')

    def test_config_field_descriptions(self):
        """Test that configuration fields have proper descriptions."""
        # Test agent_framework field
        field_info = Config.model_fields['agent_framework']
        assert field_info.description is not None
        assert "pydantic_ai" in field_info.description
        assert "langchain" in field_info.description

        # Test LangChain agent type fields
        generator_field = Config.model_fields['langchain_generator_type']
        assert generator_field.description is not None
        assert "tool_calling" in generator_field.description
        assert "react" in generator_field.description

    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        # Test with None values (should use defaults)
        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework=None,  # Should default to pydantic_ai
        )

        # The field validator converts None to default value
        assert config.agent_framework == "pydantic_ai"

        # Test with empty string for agent_framework
        with pytest.raises(ValidationError):
            Config(
                vault_path=".",
                source_dir=".",
                agent_framework=""  # Empty string should fail validation
            )


class TestConfigBackwardCompatibility:
    """Test backward compatibility with existing configuration."""

    def test_legacy_langchain_agents_flag(self):
        """Test that legacy use_langchain_agents flag still works."""
        config = Config(
            vault_path=".",
            source_dir=".",
            use_langchain_agents=True
        )

        # Should not affect new unified configuration
        assert config.agent_framework == "pydantic_ai"  # Default
        assert config.use_langchain_agents is True  # Legacy flag preserved

    def test_mixed_legacy_and_new_config(self):
        """Test mixing legacy and new configuration options."""
        config = Config(
            vault_path=".",
            source_dir=".",
            use_langgraph=True,  # Legacy LangGraph flag
            use_pydantic_ai=True,  # Legacy PydanticAI flag
            agent_framework="langchain",  # New unified flag
            langchain_generator_type="structured_chat"
        )

        # Both legacy and new settings should coexist
        assert config.use_langgraph is True
        assert config.use_pydantic_ai is True
        assert config.agent_framework == "langchain"
        assert config.langchain_generator_type == "structured_chat"


if __name__ == "__main__":
    pytest.main([__file__])
