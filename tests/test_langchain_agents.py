"""Tests for LangChain-based agents.

This module contains comprehensive tests for the LangChain agent implementations,
including Tool Calling, ReAct, Structured Chat, and JSON Chat agents.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.obsidian_anki_sync.agents.langchain.base import BaseLangChainAgent, LangChainAgentResult
from src.obsidian_anki_sync.agents.langchain.tools import (
    APFValidatorTool,
    HTMLFormatterTool,
    SlugGeneratorTool,
    ContentHashTool,
    MetadataExtractorTool,
    CardTemplateTool,
    QAExtractorTool,
)


class TestLangChainAgentResult:
    """Test LangChainAgentResult dataclass."""

    def test_initialization(self):
        """Test LangChainAgentResult initialization."""
        result = LangChainAgentResult(
            success=True,
            reasoning="Test reasoning",
            data={"test": "data"},
            warnings=["warning1", "warning2"],
            confidence=0.85,
            metadata={"key": "value"}
        )

        assert result.success is True
        assert result.reasoning == "Test reasoning"
        assert result.data == {"test": "data"}
        assert result.warnings == ["warning1", "warning2"]
        assert result.confidence == 0.85
        assert result.metadata == {"key": "value"}

    def test_default_values(self):
        """Test default values in LangChainAgentResult."""
        result = LangChainAgentResult(success=False, reasoning="Failed")

        assert result.success is False
        assert result.reasoning == "Failed"
        assert result.data is None
        assert result.warnings == []
        assert result.confidence == 1.0
        assert result.metadata == {}


class TestAPFValidatorTool:
    """Test APFValidatorTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = APFValidatorTool()
        assert tool.name == "apf_validator"
        assert "Validate APF card format" in tool.description

    def test_valid_apf(self):
        """Test validation of valid APF content."""
        tool = APFValidatorTool()
        valid_apf = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: test-card | CardType: Simple | Tags: test -->
<!-- Title -->
Question
<!-- Key point -->
Answer
<!-- Key point notes -->
<ul><li>Note</li></ul>

<!-- END_CARDS -->
END_OF_CARDS"""

        result = tool._run(valid_apf)
        assert "APF validation passed" in result

    def test_invalid_apf(self):
        """Test validation of invalid APF content."""
        tool = APFValidatorTool()
        invalid_apf = "Invalid content"

        result = tool._run(invalid_apf)
        assert "APF validation failed" in result


class TestHTMLFormatterTool:
    """Test HTMLFormatterTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = HTMLFormatterTool()
        assert tool.name == "html_formatter"
        assert "Format and validate HTML content" in tool.description

    def test_valid_html(self):
        """Test formatting of valid HTML."""
        tool = HTMLFormatterTool()
        html_content = "<p>Valid HTML</p>"

        result = tool._run(html_content)
        assert "HTML validation passed" in result

    def test_invalid_html(self):
        """Test formatting of invalid HTML."""
        tool = HTMLFormatterTool()
        html_content = "<p>Unclosed paragraph"

        result = tool._run(html_content)
        assert "HTML validation failed" in result


class TestSlugGeneratorTool:
    """Test SlugGeneratorTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = SlugGeneratorTool()
        assert tool.name == "slug_generator"
        assert "Generate unique slugs" in tool.description

    def test_slug_generation(self):
        """Test slug generation."""
        tool = SlugGeneratorTool()

        result = tool._run("Test Question", "base-slug")
        assert "Generated slug:" in result
        assert "test-question--base-slug" in result

    def test_slug_with_existing(self):
        """Test slug generation with existing slugs."""
        tool = SlugGeneratorTool()

        result = tool._run("Test Question", "base-slug",
                           ["test-question--base-slug"])
        assert "Generated slug:" in result
        assert "test-question--base-slug-1" in result


class TestContentHashTool:
    """Test ContentHashTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = ContentHashTool()
        assert tool.name == "content_hash"
        assert "Compute content hash" in tool.description

    def test_hash_computation(self):
        """Test content hash computation."""
        tool = ContentHashTool()
        content = "Test content for hashing"

        result = tool._run(content)
        assert "Content hash:" in result
        assert len(result) > 20  # Should contain hash


class TestMetadataExtractorTool:
    """Test MetadataExtractorTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = MetadataExtractorTool()
        assert tool.name == "metadata_extractor"
        assert "Extract metadata from note content" in tool.description

    def test_valid_frontmatter(self):
        """Test extraction from valid YAML frontmatter."""
        tool = MetadataExtractorTool()
        content = """---
title: Test Note
topic: algorithms
difficulty: easy
---

# Note content
Test question and answer."""

        result = tool._run(content)
        assert "Extracted metadata:" in result

    def test_no_frontmatter(self):
        """Test extraction when no frontmatter exists."""
        tool = MetadataExtractorTool()
        content = "# Just content\nNo frontmatter here."

        result = tool._run(content)
        assert "No YAML frontmatter found" in result


class TestCardTemplateTool:
    """Test CardTemplateTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = CardTemplateTool()
        assert tool.name == "card_template"
        assert "Generate APF card templates" in tool.description

    def test_simple_card_template(self):
        """Test generation of simple card template."""
        tool = CardTemplateTool()

        result = tool._run(
            card_type="Simple",
            question="What is Python?",
            answer="A programming language",
            slug="python-definition",
            language="en",
            tags=["python", "programming"]
        )

        assert "Generated Simple card template:" in result
        assert "python-definition" in result
        assert "What is Python?" in result
        assert "A programming language" in result


class TestQAExtractorTool:
    """Test QAExtractorTool."""

    def test_name_and_description(self):
        """Test tool name and description."""
        tool = QAExtractorTool()
        assert tool.name == "qa_extractor"
        assert "Extract Q&A pairs from note content" in tool.description

    def test_extraction_with_qa_pairs(self):
        """Test Q&A extraction from content with pairs."""
        tool = QAExtractorTool()
        content = """# Question 1
What is Python?
**Answer:** A programming language

# Question 2
What is a variable?
**Answer:** A storage location"""

        result = tool._run(content)
        assert "Extracted" in result and "Q&A pairs" in result

    def test_no_qa_pairs(self):
        """Test extraction when no Q&A pairs found."""
        tool = QAExtractorTool()
        content = "Just regular content without Q&A structure."

        result = tool._run(content)
        assert "No Q&A pairs found" in result


class TestToolRegistry:
    """Test tool registry functionality."""

    def test_get_tool(self):
        """Test getting tool by name."""
        from src.obsidian_anki_sync.agents.langchain.tools import get_tool

        tool = get_tool("apf_validator")
        assert isinstance(tool, APFValidatorTool)

        tool = get_tool("html_formatter")
        assert isinstance(tool, HTMLFormatterTool)

    def test_get_tool_invalid_name(self):
        """Test getting tool with invalid name."""
        from src.obsidian_anki_sync.agents.langchain.tools import get_tool

        with pytest.raises(ValueError, match="Unknown tool"):
            get_tool("invalid_tool")

    def test_get_tools_for_agent(self):
        """Test getting tools for specific agent types."""
        from src.obsidian_anki_sync.agents.langchain.tools import get_tools_for_agent

        generator_tools = get_tools_for_agent("generator")
        assert len(generator_tools) > 0

        validator_tools = get_tools_for_agent("validator")
        assert len(validator_tools) > 0

        # Check that tools are instances of BaseTool
        for tool in generator_tools + validator_tools:
            from langchain_core.tools import BaseTool
            assert isinstance(tool, BaseTool)


# Mock LangChain classes for testing
@pytest.fixture
def mock_langchain_model():
    """Mock LangChain language model."""
    model = Mock()
    model.__class__.__name__ = "MockLanguageModel"
    return model


@pytest.fixture
def mock_tool():
    """Mock LangChain tool."""
    tool = Mock()
    tool.name = "mock_tool"
    return tool


class TestBaseLangChainAgent:
    """Test BaseLangChainAgent functionality."""

    def test_initialization(self, mock_langchain_model, mock_tool):
        """Test base agent initialization."""
        # Create a concrete implementation for testing
        class TestAgent(BaseLangChainAgent):
            def _create_agent(self):
                return Mock()

        agent = TestAgent(
            model=mock_langchain_model,
            tools=[mock_tool],
            agent_type="test",
            temperature=0.5,
            max_tokens=100
        )

        assert agent.model == mock_langchain_model
        assert agent.tools == [mock_tool]
        assert agent.agent_type == "test"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 100

    def test_get_agent_info(self, mock_langchain_model, mock_tool):
        """Test getting agent information."""
        class TestAgent(BaseLangChainAgent):
            def _create_agent(self):
                return Mock()

        agent = TestAgent(
            model=mock_langchain_model,
            tools=[mock_tool],
            agent_type="test"
        )

        info = agent.get_agent_info()
        assert info["agent_type"] == "test"
        assert "MockLanguageModel" in str(info["model"])
        assert info["tools"] == ["mock_tool"]
        assert info["temperature"] == 0.0

    def test_validate_result_success(self, mock_langchain_model, mock_tool):
        """Test result validation for successful results."""
        class TestAgent(BaseLangChainAgent):
            def _create_agent(self):
                return Mock()

        agent = TestAgent(model=mock_langchain_model, tools=[mock_tool])

        # Test with LangChainAgentResult
        result = LangChainAgentResult(success=True, reasoning="Success")
        validated = agent._validate_result(result)
        assert validated.success is True
        assert validated.reasoning == "Success"

        # Test with dict
        dict_result = {
            "success": False,
            "reasoning": "Failed",
            "confidence": 0.5,
            "warnings": ["Warning"]
        }
        validated = agent._validate_result(dict_result)
        assert validated.success is False
        assert validated.reasoning == "Failed"
        assert validated.confidence == 0.5
        assert validated.warnings == ["Warning"]

    def test_extract_confidence(self, mock_langchain_model, mock_tool):
        """Test confidence extraction from output."""
        class TestAgent(BaseLangChainAgent):
            def _create_agent(self):
                return Mock()

        agent = TestAgent(model=mock_langchain_model, tools=[mock_tool])

        # Test high confidence
        output = "I am highly confident in this result"
        confidence = agent._extract_confidence(output)
        assert confidence >= 0.8

        # Test percentage
        output = "Confidence: 75%"
        confidence = agent._extract_confidence(output)
        assert confidence == 0.75

        # Test default
        output = "Regular output without confidence indicators"
        confidence = agent._extract_confidence(output)
        assert confidence == 0.8  # Default

    def test_extract_warnings(self, mock_langchain_model, mock_tool):
        """Test warning extraction from output."""
        class TestAgent(BaseLangChainAgent):
            def _create_agent(self):
                return Mock()

        agent = TestAgent(model=mock_langchain_model, tools=[mock_tool])

        output = """This is a warning: be careful with this approach.
Note: This might cause issues.
Regular text here.
Suggestion: Consider alternative approach."""

        warnings = agent._extract_warnings(output)
        assert len(warnings) >= 2
        assert any("warning:" in w.lower() for w in warnings)
        assert any("note:" in w.lower() for w in warnings)


class TestLangChainFactory:
    """Test LangChain agent factory."""

    def test_factory_initialization(self):
        """Test factory initialization."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.langchain.factory import LangChainAgentFactory

        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="langchain"
        )
        factory = LangChainAgentFactory(config)

        assert factory.config == config

    def test_available_agent_types(self):
        """Test getting available agent types."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.langchain.factory import LangChainAgentFactory

        config = Config(vault_path=".", source_dir=".")
        factory = LangChainAgentFactory(config)

        types = factory.get_available_agent_types()
        expected_types = ["tool_calling", "react",
                          "structured_chat", "json_chat"]
        assert set(types) == set(expected_types)

    def test_cache_management(self):
        """Test agent caching functionality."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.langchain.factory import LangChainAgentFactory

        config = Config(vault_path=".", source_dir=".")
        factory = LangChainAgentFactory(config)

        # Initially empty cache
        info = factory.get_cache_info()
        assert info["cached_agents"] == 0
        assert info["cache_keys"] == []

        # Clear cache (should not error)
        factory.clear_cache()
        info = factory.get_cache_info()
        assert info["cached_agents"] == 0


# Integration-style tests that may require mocking
class TestUnifiedAgentSelector:
    """Test unified agent selector."""

    def test_selector_initialization(self):
        """Test unified agent selector initialization."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.unified_agent import UnifiedAgentSelector

        config = Config(vault_path=".", source_dir=".")
        selector = UnifiedAgentSelector(config)

        assert selector.config == config
        assert selector._agents == {}

    def test_get_agent_pydantic_ai(self):
        """Test getting PydanticAI agent."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.unified_agent import UnifiedAgentSelector

        config = Config(
            vault_path=".",
            source_dir=".",
            agent_framework="pydantic_ai"
        )
        selector = UnifiedAgentSelector(config)

        agent = selector.get_agent("pydantic_ai", "generator")
        assert agent.agent_framework == "pydantic_ai"

    def test_get_agent_with_fallback(self):
        """Test agent selection with fallback."""
        from src.obsidian_anki_sync.config import Config
        from src.obsidian_anki_sync.agents.unified_agent import UnifiedAgentSelector

        config = Config(vault_path=".", source_dir=".")
        selector = UnifiedAgentSelector(config)

        # Test fallback functionality
        agent = selector.get_agent_with_fallback(
            "pydantic_ai", "langchain", "generator")
        assert agent.agent_framework == "pydantic_ai"


if __name__ == "__main__":
    pytest.main([__file__])
