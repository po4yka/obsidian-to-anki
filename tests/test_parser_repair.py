"""Tests for ParserRepairAgent."""

import json
from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.agents.parser_repair import ParserRepairAgent, attempt_repair
from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.obsidian.parser import parse_note_with_repair


@pytest.fixture
def mock_ollama_provider():
    """Create a mock Ollama provider for testing."""
    mock_provider = MagicMock()
    return mock_provider


@pytest.fixture
def parser_repair_agent(mock_ollama_provider):
    """Create a ParserRepairAgent instance for testing."""
    return ParserRepairAgent(
        ollama_client=mock_ollama_provider, model="qwen3:8b", temperature=0.0
    )


@pytest.fixture
def malformed_note_empty_language_tags():
    """Create a malformed note with empty language_tags."""
    content = """---
id: test-001
title: Test Question
topic: testing
language_tags: []
created: 2024-01-01
updated: 2024-01-02
---

# Вопрос (RU)
> What is testing?

# Question (EN)
> What is testing?

## Ответ (RU)
Testing is verification.

## Answer (EN)
Testing is verification.
"""
    return content


@pytest.fixture
def repaired_note_content():
    """Create repaired note content."""
    content = """---
id: test-001
title: Test Question
topic: testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Вопрос (RU)
> What is testing?

# Question (EN)
> What is testing?

## Ответ (RU)
Testing is verification.

## Answer (EN)
Testing is verification.
"""
    return content


class TestParserRepairAgent:
    """Tests for ParserRepairAgent class."""

    def test_initialization(self, parser_repair_agent):
        """Test agent initialization."""
        assert parser_repair_agent.model == "qwen3:8b"
        assert parser_repair_agent.temperature == 0.0

    def test_build_repair_prompt(
        self, parser_repair_agent, malformed_note_empty_language_tags
    ):
        """Test repair prompt generation."""
        error = "Missing required fields: language_tags"
        prompt = parser_repair_agent._build_repair_prompt(
            malformed_note_empty_language_tags, error
        )

        assert "PARSING ERROR:" in prompt
        assert error in prompt
        assert "language_tags" in prompt.lower()
        assert "RESPOND IN JSON FORMAT:" in prompt

    def test_repair_and_parse_success(
        self,
        parser_repair_agent,
        malformed_note_empty_language_tags,
        repaired_note_content,
        tmp_path,
    ):
        """Test successful repair and parse."""
        # Create temp file with malformed content
        test_file = tmp_path / "test-note.md"
        test_file.write_text(malformed_note_empty_language_tags)

        # Mock LLM response
        repair_response = {
            "is_repairable": True,
            "diagnosis": "Empty language_tags field",
            "repairs": [
                {
                    "issue": "language_tags is empty",
                    "fix": "Set language_tags to [en, ru] based on content",
                }
            ],
            "repaired_content": repaired_note_content,
        }

        parser_repair_agent.ollama_client.generate_json.return_value = repair_response

        # Attempt repair
        result = parser_repair_agent.repair_and_parse(
            test_file, ParserError("language_tags is empty")
        )

        assert result is not None
        metadata, qa_pairs = result
        assert metadata.language_tags == ["en", "ru"]
        assert len(qa_pairs) == 1

    def test_repair_and_parse_unrepairable(self, parser_repair_agent, tmp_path):
        """Test unrepairable note."""
        # Create temp file with fundamentally broken content
        test_file = tmp_path / "broken-note.md"
        test_file.write_text("This is not even a valid note")

        # Mock LLM response indicating unrepairable
        repair_response = {
            "is_repairable": False,
            "diagnosis": "File contains no valid frontmatter or structure",
            "repairs": [],
            "repaired_content": None,
        }

        parser_repair_agent.ollama_client.generate_json.return_value = repair_response

        # Attempt repair
        result = parser_repair_agent.repair_and_parse(
            test_file, ParserError("No frontmatter found")
        )

        assert result is None

    def test_repair_and_parse_llm_failure(self, parser_repair_agent, tmp_path):
        """Test LLM call failure."""
        test_file = tmp_path / "test-note.md"
        test_file.write_text("---\ntest\n---")

        # Mock LLM exception
        parser_repair_agent.ollama_client.generate_json.side_effect = Exception(
            "LLM error"
        )

        # Attempt repair
        result = parser_repair_agent.repair_and_parse(
            test_file, ParserError("Parse failed")
        )

        assert result is None

    def test_repair_and_parse_invalid_json(self, parser_repair_agent, tmp_path):
        """Test invalid JSON response from LLM."""
        test_file = tmp_path / "test-note.md"
        test_file.write_text("---\ntest\n---")

        # Mock invalid JSON response - generate_json raises JSONDecodeError
        parser_repair_agent.ollama_client.generate_json.side_effect = (
            json.JSONDecodeError("Invalid JSON", "This is not JSON", 0)
        )

        # Attempt repair
        result = parser_repair_agent.repair_and_parse(
            test_file, ParserError("Parse failed")
        )

        assert result is None


class TestAttemptRepairHelper:
    """Tests for attempt_repair helper function."""

    def test_attempt_repair_calls_agent(
        self, mock_ollama_provider, tmp_path, repaired_note_content
    ):
        """Test that attempt_repair creates agent and calls repair_and_parse."""
        test_file = tmp_path / "test-note.md"
        test_file.write_text("malformed content")

        # Mock successful repair
        repair_response = {
            "is_repairable": True,
            "diagnosis": "Fixed language_tags",
            "repairs": [{"issue": "empty tags", "fix": "added [en, ru]"}],
            "repaired_content": repaired_note_content,
        }

        mock_ollama_provider.generate_json.return_value = repair_response

        result = attempt_repair(
            file_path=test_file,
            original_error=ParserError("Parse failed"),
            ollama_client=mock_ollama_provider,
            model="qwen3:8b",
        )

        assert result is not None
        metadata, qa_pairs = result
        assert metadata.language_tags == ["en", "ru"]


class TestParseNoteWithRepair:
    """Tests for parse_note_with_repair function."""

    def test_parse_note_with_repair_success_no_repair_needed(self, tmp_path):
        """Test parsing a valid note that doesn't need repair."""
        # Create valid note
        valid_note = """---
id: test-001
title: Test Question
topic: testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Вопрос (RU)
> What is testing?

# Question (EN)
> What is testing?

## Ответ (RU)
Testing is verification.

## Answer (EN)
Testing is verification.
"""
        test_file = tmp_path / "valid-note.md"
        test_file.write_text(valid_note)

        # Should parse without needing repair
        metadata, qa_pairs = parse_note_with_repair(
            test_file, ollama_client=None, enable_repair=False
        )

        assert metadata.language_tags == ["en", "ru"]
        assert len(qa_pairs) == 1

    def test_parse_note_with_repair_disabled(self, tmp_path):
        """Test that repair is skipped when disabled."""
        # Create malformed note with missing frontmatter
        malformed_note = """# Question
Test without frontmatter
"""
        test_file = tmp_path / "malformed-note.md"
        test_file.write_text(malformed_note)

        # Should raise error when repair is disabled
        with pytest.raises(ParserError):
            parse_note_with_repair(test_file, enable_repair=False)

    def test_parse_note_with_repair_success(
        self, tmp_path, mock_ollama_provider, repaired_note_content
    ):
        """Test successful parsing with repair."""
        # Create malformed note without frontmatter
        malformed_note = """# Question
Some content without frontmatter
"""
        test_file = tmp_path / "malformed-note.md"
        test_file.write_text(malformed_note)

        # Mock successful repair
        repair_response = {
            "is_repairable": True,
            "diagnosis": "Fixed structure",
            "repairs": [
                {
                    "issue": "missing frontmatter",
                    "fix": "added frontmatter and sections",
                }
            ],
            "repaired_content": repaired_note_content,
        }

        mock_ollama_provider.generate_json.return_value = repair_response

        # Should succeed with repair
        metadata, qa_pairs = parse_note_with_repair(
            test_file,
            ollama_client=mock_ollama_provider,
            enable_repair=True,
            repair_model="qwen3:8b",
        )

        assert metadata.language_tags == ["en", "ru"]
        assert len(qa_pairs) == 1

    def test_parse_note_with_repair_failure(self, tmp_path, mock_ollama_provider):
        """Test that error is raised when repair fails."""
        # Create malformed note
        malformed_note = "Completely broken"
        test_file = tmp_path / "broken-note.md"
        test_file.write_text(malformed_note)

        # Mock unrepairable response
        repair_response = {
            "is_repairable": False,
            "diagnosis": "Fundamentally broken",
            "repairs": [],
            "repaired_content": None,
        }

        mock_ollama_provider.generate_json.return_value = repair_response

        # Should raise error when repair fails
        with pytest.raises(ParserError, match="Parse failed and repair unsuccessful"):
            parse_note_with_repair(
                test_file, ollama_client=mock_ollama_provider, enable_repair=True
            )
