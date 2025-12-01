
from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.agents.autofix.handlers import AutoFixIssue, UnknownErrorHandler
from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.agents.post_validation.error_categories import ErrorCategory
from obsidian_anki_sync.agents.post_validation.validation_models import ValidationError
from obsidian_anki_sync.agents.post_validation.validator import PostValidatorAgent


@pytest.fixture
def mock_ollama_client():
    return MagicMock()

@pytest.fixture
def agent(mock_ollama_client):
    return PostValidatorAgent(ollama_client=mock_ollama_client)

def test_validate_returns_structured_errors(agent):
    cards = [GeneratedCard(card_index=1, slug="test", lang="en", apf_html="content", confidence=1.0)]
    metadata = MagicMock()

    # Mock syntax_validation to return structured errors
    with patch("obsidian_anki_sync.agents.post_validation.validator.syntax_validation") as mock_syntax:
        mock_syntax.return_value = [
            ValidationError(
                category=ErrorCategory.SYNTAX,
                message="Syntax error",
                code="syntax_error",
                context={}
            )
        ]

        result = agent.validate(cards, metadata)

        assert result.is_valid is False
        assert result.error_type == "syntax"
        assert result.structured_errors is not None
        assert len(result.structured_errors) == 1
        assert result.structured_errors[0]["category"] == ErrorCategory.SYNTAX
        assert result.structured_errors[0]["message"] == "Syntax error"

def test_attempt_auto_fix_uses_repair_learning(agent):
    cards = [GeneratedCard(card_index=1, slug="test", lang="en", apf_html="content", confidence=1.0)]
    error_details = "Syntax error"
    structured_errors = [
        {
            "category": "syntax",
            "message": "Syntax error",
            "code": "syntax_error",
            "context": {}
        }
    ]

    # Mock RepairLearningSystem
    agent.repair_learning = MagicMock()
    agent.repair_learning.suggest_strategy.return_value = "deterministic"

    # Mock deterministic fixer
    with patch("obsidian_anki_sync.agents.post_validation.validator.DeterministicFixer.apply_fixes") as mock_fix:
        mock_fix.return_value = cards

        result = agent.attempt_auto_fix(cards, error_details, structured_errors)

        assert result is not None
        agent.repair_learning.suggest_strategy.assert_called_once()
        agent.repair_learning.learn_from_success.assert_called_once()

def test_unknown_error_handler():
    handler = UnknownErrorHandler(log_file="test_unknown_errors.log")
    content = "test content"
    issues = [
        AutoFixIssue(
            issue_type="unknown_error",
            description="Unknown error",
            location="unknown"
        )
    ]

    # Mock open to avoid writing to file
    with patch("builtins.open", MagicMock()):
        fixed_content, updated_issues = handler.fix(content, issues)

        assert fixed_content == content
        assert updated_issues[0].auto_fixed is False
        assert "Logged to" in updated_issues[0].fix_description
