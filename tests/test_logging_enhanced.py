"""Tests for enhanced logging configuration."""

from pathlib import Path

import pytest

from obsidian_anki_sync.utils.logging import configure_logging, get_logger


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

        # Check that log files exist
        log_files = list(project_log_dir.glob("*.log"))
        assert len(log_files) >= 2

        # Check for error log file
        error_logs = list(project_log_dir.glob("errors_*.log"))
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

        # Find error log file
        error_logs = list(project_log_dir.glob("errors_*.log"))
        assert len(error_logs) >= 1

        error_log_content = error_logs[0].read_text(encoding="utf-8")
        assert "error_message" in error_log_content
        assert "ERROR" in error_log_content

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

        # Verify error log was created
        error_logs = list(project_log_dir.glob("errors_*.log"))
        assert len(error_logs) >= 1

