#!/usr/bin/env python3
"""Test script to verify robust error handling works correctly."""

from obsidian_anki_sync.agents.models import (
    AgentPipelineResult,
    MemorizationQualityResult,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.sync.card_generator import CardGenerator


def test_robust_error_handling():
    """Test the robust error extraction with various failure scenarios."""

    # Create a mock CardGenerator instance for testing
    class MockCardGenerator(CardGenerator):
        def __init__(self):
            # Don't call super().__init__ to avoid dependencies
            pass

    generator = MockCardGenerator()

    print("Testing robust error handling...")

    # Test case 1: Pre-validation failure with detailed error
    pre_validation_failed = PreValidationResult(
        is_valid=False,
        error_type="structure",
        error_details="No Q&A pairs are present in the provided note content.",
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

    error_msg1 = generator._extract_pipeline_error_message(result1)
    expected1 = "Content validation failed (structure): No Q&A pairs are present in the provided note content."
    assert error_msg1 == expected1, f"Expected '{expected1}', got '{error_msg1}'"
    print(f"✓ Test 1 passed: {error_msg1}")

    # Test case 2: Pre-validation failure without error details (should use user-friendly message)
    pre_validation_failed_no_details = PreValidationResult(
        is_valid=False,
        error_type="structure",
        error_details="",  # Empty details
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

    error_msg2 = generator._extract_pipeline_error_message(result2)
    expected2 = "Content validation failed: Note structure issue detected. Check Q&A format and YAML frontmatter."
    assert error_msg2 == expected2, f"Expected '{expected2}', got '{error_msg2}'"
    print(f"✓ Test 2 passed: {error_msg2}")

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

    error_msg3 = generator._extract_pipeline_error_message(result3)
    expected3 = (
        "Card validation failed (factual): Incorrect information in generated card"
    )
    assert error_msg3 == expected3, f"Expected '{expected3}', got '{error_msg3}'"
    print(f"✓ Test 3 passed: {error_msg3}")

    # Test case 4: Memorization quality failure with issues
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

    error_msg4 = generator._extract_pipeline_error_message(result4)
    expected4 = "Quality check failed (score: 0.30): clarity: Question is unclear, length: Answer is too long"
    assert error_msg4 == expected4, f"Expected '{expected4}', got '{error_msg4}'"
    print(f"✓ Test 4 passed: {error_msg4}")

    # Test case 5: Pipeline failure with no generation (fallback diagnostics)
    # Use pre_validation that passes but still no generation
    result5 = AgentPipelineResult(
        success=False,
        pre_validation=PreValidationResult(
            is_valid=True, error_type="none", validation_time=1.0
        ),
        generation=None,
        post_validation=None,
        memorization_quality=None,
        total_time=5.5,
        retry_count=2,
    )

    error_msg5 = generator._extract_pipeline_error_message(result5)
    assert "Pipeline failed to generate cards" in error_msg5
    assert "pre_validation=passed" in error_msg5
    assert "time=5.5s" in error_msg5
    assert "retries=2" in error_msg5
    print(f"✓ Test 5 passed: {error_msg5}")

    # Test case 6: Invalid result object (edge case)
    class InvalidResult:
        pass

    result6 = InvalidResult()
    error_msg6 = generator._extract_pipeline_error_message(result6)
    expected6 = "Pipeline failed: Invalid result structure"
    assert error_msg6 == expected6, f"Expected '{expected6}', got '{error_msg6}'"
    print(f"✓ Test 6 passed: {error_msg6}")

    # Test case 7: Exception during error extraction (ultimate fallback)
    class BadResult:
        def __init__(self):
            self.success = False
            self.total_time = 1.0

        def __getattr__(self, name):
            if name == "generation":
                raise RuntimeError("Simulated error during attribute access")
            return None

    result7 = BadResult()
    error_msg7 = generator._extract_pipeline_error_message(result7)
    assert "error analysis unavailable" in error_msg7
    assert "RuntimeError" in error_msg7
    print(f"✓ Test 7 passed: {error_msg7}")

    print("\nAll tests passed! Robust error handling works correctly.")
    print("✓ Error messages are user-friendly and informative")
    print("✓ Edge cases are handled gracefully")
    print("✓ Fallback mechanisms prevent crashes")
    print("✓ Diagnostic information is preserved for debugging")


if __name__ == "__main__":
    test_robust_error_handling()
