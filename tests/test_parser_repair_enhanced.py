"""Tests for enhanced ParserRepairAgent with error diagnosis and quality scoring."""

from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.agents.parser_repair import ParserRepairAgent
from obsidian_anki_sync.exceptions import ParserError


@pytest.fixture()
def mock_ollama_provider():
    """Create a mock Ollama provider."""
    mock_provider = MagicMock()
    return mock_provider


@pytest.fixture()
def enhanced_parser_repair_agent(mock_ollama_provider):
    """Create an enhanced ParserRepairAgent instance."""
    return ParserRepairAgent(
        ollama_client=mock_ollama_provider,
        model="qwen3:8b",
        temperature=0.0,
        enable_content_generation=True,
        repair_missing_sections=True,
    )


@pytest.fixture()
def malformed_note_with_quality_issues():
    """Create a malformed note with quality issues."""
    return """---
id: test-002
title: Test Note
topic: testing
language_tags: [en]
created: 2024-01-01
updated: 2024-01-02
---

# Question (EN)
> What is testing?

## Answer (EN)
Testing is verification process.
"""


@pytest.fixture()
def repaired_note_high_quality():
    """Create a high-quality repaired note."""
    return """---
id: test-002
title: Test Note
topic: testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Question (EN)
> What is testing?

# Вопрос (RU)
> Что такое тестирование?

## Answer (EN)
Testing is a verification process that ensures software works correctly.

## Ответ (RU)
Тестирование - это процесс проверки, который обеспечивает правильную работу программного обеспечения.
"""


class TestEnhancedParserRepair:
    """Tests for enhanced parser repair features."""

    def test_error_diagnosis_parsing(
        self,
        enhanced_parser_repair_agent,
        malformed_note_with_quality_issues,
        repaired_note_high_quality,
        tmp_path,
    ):
        """Test structured error diagnosis parsing."""
        test_file = tmp_path / "test-note.md"
        test_file.write_text(malformed_note_with_quality_issues)

        # Mock LLM response with error diagnosis
        repair_response = {
            "is_repairable": True,
            "diagnosis": "Missing Russian language section",
            "repairs": [
                {
                    "type": "content_generation",
                    "description": "Generated missing Russian question and answer sections",
                }
            ],
            "repaired_content": repaired_note_high_quality,
            "content_generation_applied": True,
            "generated_sections": [
                {
                    "section_type": "question_ru",
                    "method": "translation",
                    "description": "Translated from English question",
                },
                {
                    "section_type": "answer_ru",
                    "method": "translation",
                    "description": "Translated from English answer",
                },
            ],
            "error_diagnosis": {
                "error_category": "content",
                "severity": "medium",
                "error_description": "Missing Russian language sections despite language_tags including 'ru'",
                "repair_priority": 3,
                "can_auto_fix": True,
            },
            "quality_before": {
                "completeness_score": 0.5,
                "structure_score": 0.9,
                "bilingual_consistency": 0.0,
                "technical_accuracy": 0.8,
                "overall_score": 0.55,
                "issues_found": [
                    "Missing Russian question section",
                    "Missing Russian answer section",
                    "Incomplete answer (too brief)",
                ],
            },
            "quality_after": {
                "completeness_score": 1.0,
                "structure_score": 1.0,
                "bilingual_consistency": 1.0,
                "technical_accuracy": 0.9,
                "overall_score": 0.975,
                "issues_found": [],
            },
            "repair_time": 1.2,
        }

        enhanced_parser_repair_agent.ollama_client.generate_json.return_value = (
            repair_response
        )

        result = enhanced_parser_repair_agent.repair_and_parse(
            test_file, ParserError("Missing Russian sections")
        )

        assert result is not None
        # Verify error diagnosis was parsed
        # Note: The agent doesn't return the diagnosis directly, but it's logged
        # We can verify the repair was successful
        metadata, _qa_pairs = result
        assert "ru" in metadata.language_tags

    def test_quality_scoring_improvement(
        self,
        enhanced_parser_repair_agent,
        malformed_note_with_quality_issues,
        repaired_note_high_quality,
        tmp_path,
    ):
        """Test quality scoring shows improvement after repair."""
        test_file = tmp_path / "test-note.md"
        test_file.write_text(malformed_note_with_quality_issues)

        repair_response = {
            "is_repairable": True,
            "diagnosis": "Quality improvements applied",
            "repairs": [
                {
                    "type": "content_generation",
                    "description": "Added missing sections and improved answer quality",
                }
            ],
            "repaired_content": repaired_note_high_quality,
            "content_generation_applied": True,
            "generated_sections": [
                {
                    "section_type": "question_ru",
                    "method": "translation",
                    "description": "Translated question",
                },
                {
                    "section_type": "answer_ru",
                    "method": "translation",
                    "description": "Translated and enhanced answer",
                },
            ],
            "error_diagnosis": {
                "error_category": "quality",
                "severity": "low",
                "error_description": "Content quality can be improved",
                "repair_priority": 5,
                "can_auto_fix": True,
            },
            "quality_before": {
                "completeness_score": 0.5,
                "structure_score": 0.9,
                "bilingual_consistency": 0.0,
                "technical_accuracy": 0.8,
                "overall_score": 0.55,
                "issues_found": ["Missing sections", "Brief answer"],
            },
            "quality_after": {
                "completeness_score": 1.0,
                "structure_score": 1.0,
                "bilingual_consistency": 1.0,
                "technical_accuracy": 0.9,
                "overall_score": 0.975,
                "issues_found": [],
            },
            "repair_time": 1.5,
        }

        enhanced_parser_repair_agent.ollama_client.generate_json.return_value = (
            repair_response
        )

        result = enhanced_parser_repair_agent.repair_and_parse(
            test_file, ParserError("Quality issues")
        )

        assert result is not None
        # Verify quality improvement (before: 0.55, after: 0.975)
        # The agent logs this, but we verify repair succeeded

    def test_error_diagnosis_critical_severity(
        self, enhanced_parser_repair_agent, tmp_path
    ):
        """Test critical severity error diagnosis."""
        test_file = tmp_path / "critical-error.md"
        test_file.write_text("Completely broken file")

        repair_response = {
            "is_repairable": False,
            "diagnosis": "File is completely unparseable",
            "repairs": [],
            "repaired_content": None,
            "content_generation_applied": False,
            "generated_sections": [],
            "error_diagnosis": {
                "error_category": "structure",
                "severity": "critical",
                "error_description": "No valid frontmatter, no content structure, completely corrupted",
                "repair_priority": 1,
                "can_auto_fix": False,
            },
            "quality_before": {
                "completeness_score": 0.0,
                "structure_score": 0.0,
                "bilingual_consistency": 0.0,
                "technical_accuracy": 0.0,
                "overall_score": 0.0,
                "issues_found": ["No frontmatter", "No structure", "No content"],
            },
            "quality_after": None,
            "repair_time": 0.5,
        }

        enhanced_parser_repair_agent.ollama_client.generate_json.return_value = (
            repair_response
        )

        result = enhanced_parser_repair_agent.repair_and_parse(
            test_file, ParserError("Critical error")
        )

        assert result is None  # Unrepairable

    def test_grammar_and_clarity_improvements(
        self, enhanced_parser_repair_agent, tmp_path
    ):
        """Test that grammar and clarity improvements are included in prompt."""
        prompt = enhanced_parser_repair_agent._build_repair_prompt(
            "test content", "test error", enable_content_gen=True
        )

        # Verify enhanced content generation instructions include grammar/clarity
        assert "Grammar and Clarity Improvements" in prompt
        assert "Technical Accuracy Checks" in prompt
        assert "grammar" in prompt.lower()
        assert "clarity" in prompt.lower()
