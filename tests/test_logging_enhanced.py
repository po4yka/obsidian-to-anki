"""Tests for enhanced logging configuration."""

import json
import logging

import pytest
import structlog

from obsidian_anki_sync.utils.logging import (
    ConsoleNoiseFilterProcessor,
    HighVolumeEventPolicy,
    configure_logging,
    get_logger,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Alias for pytest's tmp_path fixture."""
    return tmp_path


class TestEnhancedLogging:
    """Test enhanced logging configuration."""

    def test_project_log_file_creation(self, temp_dir):
        """Test that project-level log files are created."""
        project_log_dir = temp_dir / "logs"
        configure_logging(
            log_level="DEBUG",
            project_log_dir=project_log_dir,
            error_log_retention_days=90,
        )

        logger = get_logger("test")
        logger.info("test_message", test_field="test_value")
        logger.error("test_error", error_type="TestError")

        # Flush handlers to ensure logs are written
        import logging

        logging.shutdown()

        # Check that log files exist
        log_files = list(project_log_dir.glob("*.log"))
        assert len(log_files) >= 2

        # Check for error log file
        error_logs = list(project_log_dir.glob("errors*.log"))
        assert len(error_logs) >= 1

    def test_error_log_separation(self, temp_dir):
        """Test that errors are written to separate error log file."""
        project_log_dir = temp_dir / "logs"
        configure_logging(
            log_level="INFO",
            project_log_dir=project_log_dir,
        )

        logger = get_logger("test")
        logger.info("info_message")
        logger.warning("warning_message")
        logger.error("error_message", error_type="TestError")

        # Flush handlers to ensure logs are written
        import logging

        logging.shutdown()

        # Find error log file
        error_logs = list(project_log_dir.glob("errors*.log"))
        assert len(error_logs) >= 1

        error_log_content = error_logs[0].read_text(encoding="utf-8")
        # Check for JSON structure
        lines = [line.strip() for line in error_log_content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify at least one line is valid JSON
        found_error = False
        for line in lines:
            try:
                log_entry = json.loads(line)
                if log_entry.get("event") == "error_message":
                    found_error = True
                    assert log_entry.get("error_type") == "TestError"
                    assert log_entry.get("level") == "error"
                    break
            except json.JSONDecodeError:
                continue

        assert found_error, "Error message not found in error log"

        # Error log should not contain INFO messages
        assert "info_message" not in error_log_content

    def test_log_retention_configuration(self, temp_dir):
        """Test that error log retention is configurable."""
        project_log_dir = temp_dir / "logs"
        configure_logging(
            log_level="INFO",
            project_log_dir=project_log_dir,
            error_log_retention_days=30,
        )

        # Configuration should not raise errors
        logger = get_logger("test")
        logger.error("test_error")

        # Flush handlers to ensure logs are written
        import logging

        logging.shutdown()

        # Verify error log was created
        error_logs = list(project_log_dir.glob("errors*.log"))
        assert len(error_logs) >= 1

    def test_json_structure_in_file_logs(self, temp_dir):
        """Test that file logs contain valid JSON structure."""
        project_log_dir = temp_dir / "logs"
        configure_logging(
            log_level="DEBUG",
            project_log_dir=project_log_dir,
        )

        logger = get_logger("test")
        logger.info("test_json_message", test_key="test_value", number=42)

        # Flush handlers to ensure logs are written
        import logging

        logging.shutdown()

        # Find main log file
        log_files = [
            f
            for f in project_log_dir.glob("obsidian-anki-sync*.log")
            if "errors" not in f.name
        ]
        assert len(log_files) >= 1

        log_content = log_files[0].read_text(encoding="utf-8")
        lines = [line.strip() for line in log_content.split("\n") if line.strip()]
        assert len(lines) > 0

        # Verify JSON structure
        found_message = False
        for line in lines:
            try:
                log_entry = json.loads(line)
                if log_entry.get("event") == "test_json_message":
                    found_message = True
                    assert log_entry.get("test_key") == "test_value"
                    assert log_entry.get("number") == 42
                    assert "logger" in log_entry
                    assert "level" in log_entry
                    assert "timestamp" in log_entry
                    break
            except json.JSONDecodeError:
                continue

        assert found_message, (
            "Test message not found in log file with correct JSON structure"
        )

    def test_structured_fields_preserved(self, temp_dir):
        """Test that structured fields are preserved in logs."""
        project_log_dir = temp_dir / "logs"
        configure_logging(
            log_level="INFO",
            project_log_dir=project_log_dir,
        )

        logger = get_logger("test")
        logger.info(
            "structured_test",
            file="test.md",
            title="Test Title",
            note_id="test-123",
            custom_field="custom_value",
        )

        # Flush handlers to ensure logs are written
        import logging

        logging.shutdown()

        log_files = [
            f
            for f in project_log_dir.glob("obsidian-anki-sync*.log")
            if "errors" not in f.name
        ]
        assert len(log_files) >= 1

        log_content = log_files[0].read_text(encoding="utf-8")
        lines = [line.strip() for line in log_content.split("\n") if line.strip()]

        found = False
        for line in lines:
            try:
                log_entry = json.loads(line)
                if log_entry.get("event") == "structured_test":
                    found = True
                    assert log_entry.get("file") == "test.md"
                    assert log_entry.get("title") == "Test Title"
                    assert log_entry.get("note_id") == "test-123"
                    assert log_entry.get("custom_field") == "custom_value"
                    break
            except json.JSONDecodeError:
                continue

        assert found, "Structured fields not preserved in log"


class TestConsoleNoiseFilterProcessor:
    """Tests for console noise suppression processor."""

    def test_module_level_override_blocks_low_priority_provider_logs(self) -> None:
        """Ensure INFO logs from provider modules are suppressed."""
        processor = ConsoleNoiseFilterProcessor(
            level_overrides={"obsidian_anki_sync.providers": "WARNING"},
            high_volume_policies={},
        )

        # Create event dict for INFO level from provider module
        event_dict = {
            "logger": "obsidian_anki_sync.providers.factory",
            "level": logging.INFO,
            "event": "creating_provider",
        }

        # Should drop INFO level events from provider modules
        try:
            result = processor(None, "info", event_dict.copy())
            # If we get here, it wasn't dropped - check if level is high enough
            assert result["level"] >= logging.WARNING
        except structlog.DropEvent:
            # Expected - event was dropped
            pass

        # WARNING-level events should pass through
        event_dict_warning = {
            "logger": "obsidian_anki_sync.providers.factory",
            "level": logging.WARNING,
            "event": "provider_warning",
        }
        result = processor(None, "warning", event_dict_warning.copy())
        assert result["level"] == logging.WARNING

    def test_high_volume_policy_rate_limits_events(self) -> None:
        """Verify repeated events are rate-limited within the window."""
        current_time = {"value": 0.0}

        def fake_time() -> float:
            return current_time["value"]

        policy = {"creating_provider": HighVolumeEventPolicy(2, 60.0)}
        processor = ConsoleNoiseFilterProcessor(
            level_overrides={},
            high_volume_policies=policy,
            time_func=fake_time,
        )

        event_dict = {
            "logger": "test",
            "level": logging.INFO,
            "event": "creating_provider",
        }

        # First two should pass
        result1 = processor(None, "info", event_dict.copy())
        assert result1 is not None

        result2 = processor(None, "info", event_dict.copy())
        assert result2 is not None

        # Third call within the same window should be dropped
        try:
            result3 = processor(None, "info", event_dict.copy())
            # Should not reach here
            assert False, "Event should have been dropped"
        except structlog.DropEvent:
            # Expected
            pass

        # Advance time beyond the window and ensure events are allowed again
        current_time["value"] = 61.0
        result4 = processor(None, "info", event_dict.copy())
        assert result4 is not None
