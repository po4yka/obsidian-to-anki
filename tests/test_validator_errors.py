import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.validation.orchestrator import NoteValidator


def test_validate_file_logs_unexpected_exception(caplog):
    """Test that validate_file logs unexpected exceptions with traceback."""
    # Setup
    validator = NoteValidator(Path("/tmp/vault"))
    mock_path = MagicMock(spec=Path)
    mock_path.__str__.return_value = "/tmp/vault/note.md"

    # Mock parse_note to raise an exception
    with (
        patch.object(validator, "parse_note", side_effect=ValueError("Test error")),
        caplog.at_level(logging.ERROR),
    ):
        result = validator.validate_file(mock_path)

    # Verify result
    assert result["success"] is False
    assert "Unexpected error: Test error" in result["error"]

    # Verify logging
    assert "validation_failed_unexpectedly" in caplog.text
    assert "Test error" in caplog.text
    # Check that traceback is included (logger.exception adds it)
    # Note: caplog.text might not contain the full traceback string depending on formatter,
    # but the presence of the message logged via logger.exception implies exc_info=True.
    # We can check if the log record has exc_info set.
    assert any(
        record.exc_info
        for record in caplog.records
        if "validation_failed_unexpectedly" in record.message
    )


def test_apply_fixes_logs_exception(caplog):
    """Test that apply_fixes logs exceptions during fix application."""
    # Setup
    validator = NoteValidator(Path("/tmp/vault"))
    mock_path = MagicMock(spec=Path)
    mock_path.__str__.return_value = "/tmp/vault/note.md"
    mock_path.read_text.return_value = "---\n---\ncontent"

    # Create a mock fix that raises an exception
    mock_fix = MagicMock()
    mock_fix.description = "Test Fix"
    mock_fix.fix_function.side_effect = RuntimeError("Fix failed")

    # Run apply_fixes
    with caplog.at_level(logging.WARNING):
        count, applied = validator.apply_fixes(mock_path, [mock_fix])

    # Verify result
    assert count == 0
    assert applied == []

    # Verify logging
    assert "fix_application_failed" in caplog.text
    assert "Fix failed" in caplog.text
    # Check that traceback is included in the text (structlog formats it into the message)
    # We check for a common part of the traceback
    assert "Traceback" in caplog.text or "RuntimeError: Fix failed" in caplog.text
