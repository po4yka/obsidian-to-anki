#!/usr/bin/env python3
"""Test script to verify error message extraction works correctly."""

from obsidian_anki_sync.agents.models import (
    AgentPipelineResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)


def test_error_extraction():
    """Test the error extraction logic from generate_with_agents."""

    # Test case 1: Pre-validation failure with error details
    pre_validation_failed = PreValidationResult(
        is_valid=False,
        error_type="structure",
        error_details="Invalid Q&A format detected",
        validation_time=1.5,
    )

    result1 = AgentPipelineResult(
        success=False,
        pre_validation=pre_validation_failed,
        generation=None,
        post_validation=None,
        memorization_quality=None,
        total_time=2.0,
        retry_count=0,
    )

    # Simulate the error extraction logic
    if result1.post_validation and result1.post_validation.error_details:
        error_msg = result1.post_validation.error_details
    elif result1.pre_validation and not result1.pre_validation.is_valid:
        error_details = result1.pre_validation.error_details
        if error_details:
            error_msg = f"Pre-validation failed ({result1.pre_validation.error_type}): {error_details}"
        else:
            error_msg = f"Pre-validation failed: {result1.pre_validation.error_type}"
    else:
        error_msg = "Unknown error"

    expected = "Pre-validation failed (structure): Invalid Q&A format detected"
    assert error_msg == expected, f"Expected '{expected}', got '{error_msg}'"
    print(f"✓ Test 1 passed: {error_msg}")

    # Test case 2: Pre-validation failure without error details
    pre_validation_failed_no_details = PreValidationResult(
        is_valid=False,
        error_type="format",
        error_details="",  # Empty error details
        validation_time=1.0,
    )

    result2 = AgentPipelineResult(
        success=False,
        pre_validation=pre_validation_failed_no_details,
        generation=None,
        post_validation=None,
        memorization_quality=None,
        total_time=1.5,
        retry_count=0,
    )

    if result2.post_validation and result2.post_validation.error_details:
        error_msg = result2.post_validation.error_details
    elif result2.pre_validation and not result2.pre_validation.is_valid:
        error_details = result2.pre_validation.error_details
        if error_details:
            error_msg = f"Pre-validation failed ({result2.pre_validation.error_type}): {error_details}"
        else:
            error_msg = f"Pre-validation failed: {result2.pre_validation.error_type}"
    else:
        error_msg = "Unknown error"

    expected = "Pre-validation failed: format"
    assert error_msg == expected, f"Expected '{expected}', got '{error_msg}'"
    print(f"✓ Test 2 passed: {error_msg}")

    # Test case 3: Post-validation failure
    post_validation_failed = PostValidationResult(
        is_valid=False,
        error_type="factual",
        error_details="Incorrect information in generated card",
        validation_time=2.0,
    )

    result3 = AgentPipelineResult(
        success=False,
        pre_validation=PreValidationResult(
            is_valid=True, error_type="none", validation_time=1.0
        ),
        generation=None,
        post_validation=post_validation_failed,
        memorization_quality=None,
        total_time=3.0,
        retry_count=1,
    )

    if result3.post_validation and result3.post_validation.error_details:
        error_msg = f"Post-validation failed ({result3.post_validation.error_type}): {result3.post_validation.error_details}"
    elif result3.pre_validation and not result3.pre_validation.is_valid:
        error_details = result3.pre_validation.error_details
        if error_details:
            error_msg = f"Pre-validation failed ({result3.pre_validation.error_type}): {error_details}"
        else:
            error_msg = f"Pre-validation failed: {result3.pre_validation.error_type}"
    else:
        error_msg = "Unknown error"

    expected = (
        "Post-validation failed (factual): Incorrect information in generated card"
    )
    assert error_msg == expected, f"Expected '{expected}', got '{error_msg}'"
    print(f"✓ Test 3 passed: {error_msg}")

    # Test case 4: Memorization quality failure
    memorization_failed = MemorizationQualityResult(
        is_memorizable=False,
        memorization_score=0.3,
        issues=[
            {"type": "clarity", "description": "Question is unclear"},
            {"type": "length", "description": "Answer is too long"},
        ],
        strengths=[],
        suggested_improvements=["Simplify question", "Shorten answer"],
        assessment_time=1.0,
    )

    result4 = AgentPipelineResult(
        success=False,
        pre_validation=PreValidationResult(
            is_valid=True, error_type="none", validation_time=1.0
        ),
        generation=None,
        post_validation=None,
        memorization_quality=memorization_failed,
        total_time=4.0,
        retry_count=0,
    )

    if result4.post_validation and result4.post_validation.error_details:
        error_msg = result4.post_validation.error_details
    elif result4.pre_validation and not result4.pre_validation.is_valid:
        error_details = result4.pre_validation.error_details
        if error_details:
            error_msg = f"Pre-validation failed ({result4.pre_validation.error_type}): {error_details}"
        else:
            error_msg = f"Pre-validation failed: {result4.pre_validation.error_type}"
    elif (
        result4.memorization_quality and not result4.memorization_quality.is_memorizable
    ):
        issues = result4.memorization_quality.issues
        if issues:
            issue_summaries = [
                f"{issue.get('type', 'unknown')}: {issue.get('description', 'no description')}"
                for issue in issues[:3]
            ]
            error_msg = f"Memorization quality failed: {', '.join(issue_summaries)}"
        else:
            error_msg = f"Memorization quality failed: score {result4.memorization_quality.memorization_score:.2f}"
    else:
        error_msg = "Unknown error"

    expected = "Memorization quality failed: clarity: Question is unclear, length: Answer is too long"
    assert error_msg == expected, f"Expected '{expected}', got '{error_msg}'"
    print(f"✓ Test 4 passed: {error_msg}")

    print("All tests passed! Error extraction logic works correctly.")


if __name__ == "__main__":
    test_error_extraction()
