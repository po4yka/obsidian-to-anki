"""Tests for validation output model auto-correction validators."""

import pytest

from obsidian_anki_sync.agents.models import CardCorrection
from obsidian_anki_sync.agents.pydantic.models import CardIssue, PostValidationOutput


class TestCardIssueValidators:
    """Tests for CardIssue auto-correction."""

    def test_empty_issue_description_gets_default(self):
        """Empty issue_description should get default value."""
        issue = CardIssue(card_index=1, issue_description="")
        assert issue.issue_description == "Unspecified issue"

    def test_none_issue_description_gets_default(self):
        """None issue_description should get default value."""
        issue = CardIssue.model_validate({"card_index": 1, "issue_description": None})
        assert issue.issue_description == "Unspecified issue"

    def test_whitespace_only_issue_description_gets_default(self):
        """Whitespace-only issue_description should get default value."""
        issue = CardIssue(card_index=1, issue_description="   ")
        assert issue.issue_description == "Unspecified issue"

    def test_valid_issue_description_preserved(self):
        """Valid issue_description should be preserved."""
        issue = CardIssue(card_index=1, issue_description="Missing closing tag")
        assert issue.issue_description == "Missing closing tag"

    def test_zero_card_index_corrected(self):
        """Zero card_index should be corrected to 1."""
        issue = CardIssue.model_validate({"card_index": 0, "issue_description": "test"})
        assert issue.card_index == 1

    def test_negative_card_index_corrected(self):
        """Negative card_index should be corrected to 1."""
        issue = CardIssue.model_validate(
            {"card_index": -5, "issue_description": "test"}
        )
        assert issue.card_index == 1

    def test_none_card_index_defaults_to_1(self):
        """None card_index should default to 1."""
        issue = CardIssue.model_validate(
            {"card_index": None, "issue_description": "test"}
        )
        assert issue.card_index == 1


class TestCardCorrectionValidators:
    """Tests for CardCorrection auto-correction."""

    def test_none_card_index_defaults_to_1(self):
        """None card_index should default to 1."""
        correction = CardCorrection.model_validate(
            {"card_index": None, "field_name": "apf_html", "suggested_value": "test"}
        )
        assert correction.card_index == 1

    def test_empty_field_name_gets_default(self):
        """Empty field_name should get default value."""
        correction = CardCorrection(card_index=1, field_name="", suggested_value="test")
        assert correction.field_name == "apf_html"

    def test_none_suggested_value_becomes_empty_string(self):
        """None suggested_value should become empty string."""
        correction = CardCorrection.model_validate(
            {"card_index": 1, "field_name": "apf_html", "suggested_value": None}
        )
        assert correction.suggested_value == ""

    def test_valid_values_preserved(self):
        """Valid values should be preserved."""
        correction = CardCorrection(
            card_index=2,
            field_name="slug",
            suggested_value="new-slug",
            rationale="Fix naming",
        )
        assert correction.card_index == 2
        assert correction.field_name == "slug"
        assert correction.suggested_value == "new-slug"
        assert correction.rationale == "Fix naming"


class TestPostValidationOutputValidators:
    """Tests for PostValidationOutput auto-correction."""

    def test_none_is_valid_defaults_to_true(self):
        """None is_valid should default to True."""
        output = PostValidationOutput.model_validate(
            {"is_valid": None, "error_type": "none"}
        )
        assert output.is_valid is True

    def test_string_is_valid_coerced(self):
        """String is_valid should be coerced to bool."""
        output = PostValidationOutput.model_validate(
            {"is_valid": "true", "error_type": "none"}
        )
        assert output.is_valid is True

        output2 = PostValidationOutput.model_validate(
            {"is_valid": "false", "error_type": "syntax"}
        )
        assert output2.is_valid is False

    def test_none_error_type_defaults_to_none(self):
        """None error_type should default to 'none'."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": None}
        )
        assert output.error_type == "none"

    def test_empty_error_type_defaults_to_none(self):
        """Empty error_type should default to 'none'."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": ""}
        )
        assert output.error_type == "none"

    def test_error_type_aliases_mapped(self):
        """Error type aliases should be mapped to valid types."""
        # no_error -> none
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "no_error"}
        )
        assert output.error_type == "none"

        # format -> syntax
        output2 = PostValidationOutput.model_validate(
            {"is_valid": False, "error_type": "format"}
        )
        assert output2.error_type == "syntax"

        # accuracy -> factual
        output3 = PostValidationOutput.model_validate(
            {"is_valid": False, "error_type": "accuracy"}
        )
        assert output3.error_type == "factual"

    def test_unknown_error_type_defaults_to_none(self):
        """Unknown error_type should default to 'none'."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "unknown_error_xyz"}
        )
        assert output.error_type == "none"

    def test_consistency_valid_with_error_type_corrected(self):
        """If is_valid=True but error_type is not 'none', correct error_type."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "syntax"}
        )
        assert output.is_valid is True
        assert output.error_type == "none"

    def test_consistency_invalid_with_none_error_type_inferred(self):
        """If is_valid=False but error_type='none' with issues, infer error_type."""
        output = PostValidationOutput.model_validate(
            {
                "is_valid": False,
                "error_type": "none",
                "card_issues": [{"card_index": 1, "issue_description": "test issue"}],
            }
        )
        assert output.is_valid is False
        assert output.error_type == "template"

    def test_confidence_out_of_range_clamped(self):
        """Confidence values out of range should be clamped."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "none", "confidence": 1.5}
        )
        assert output.confidence == 1.0

        output2 = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "none", "confidence": -0.5}
        )
        assert output2.confidence == 0.0

    def test_none_confidence_defaults(self):
        """None confidence should default to 0.5."""
        output = PostValidationOutput.model_validate(
            {"is_valid": True, "error_type": "none", "confidence": None}
        )
        assert output.confidence == 0.5

    def test_minimal_valid_output(self):
        """Minimal valid output should work with all defaults."""
        output = PostValidationOutput.model_validate({})
        assert output.is_valid is True
        assert output.error_type == "none"
        assert output.error_details == ""
        assert output.card_issues == []
        assert output.suggested_corrections == []
        assert output.confidence == 0.5
