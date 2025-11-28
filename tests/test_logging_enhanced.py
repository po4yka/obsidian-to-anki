"""Tests for enhanced logging configuration."""

import pytest

from obsidian_anki_sync.utils.logging import (
    ConsoleNoiseFilter,
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


class LevelStub:
    """Minimal stand-in for loguru's Level object."""

    _level_map = {
        "TRACE": 5,
        "DEBUG": 10,
        "INFO": 20,
        "SUCCESS": 25,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }

    def __init__(self, name: str) -> None:
        level_name = name.upper()
        if level_name not in self._level_map:
            raise ValueError(f"Unsupported level: {name}")
        self.no = self._level_map[level_name]
        self.name = level_name


def _make_record(
    *,
    level: str = "INFO",
    module: str = "obsidian_anki_sync.providers.factory",
    message: str = "creating_provider",
) -> dict:
    """Create a record dict resembling loguru's filter input."""
    return {
        "level": LevelStub(level),
        "name": module,
        "message": message,
        "extra": {},
    }


class TestConsoleNoiseFilter:
    """Tests for console noise suppression."""

    def test_module_level_override_blocks_low_priority_provider_logs(self) -> None:
        """Ensure INFO logs from provider modules are suppressed."""
        filter_fn = ConsoleNoiseFilter(
            base_filter=lambda record: True,
            level_overrides={"obsidian_anki_sync.providers": "WARNING"},
            high_volume_policies={},
        )

        assert filter_fn(_make_record(level="INFO")) is False
        # WARNING-level events should still pass through
        assert filter_fn(_make_record(
            level="WARNING", message="provider_warning")) is True

    def test_high_volume_policy_rate_limits_events(self) -> None:
        """Verify repeated events are rate-limited within the window."""
        current_time = {"value": 0.0}

        def fake_time() -> float:
            return current_time["value"]

        policy = {"creating_provider": HighVolumeEventPolicy(2, 60.0)}
        filter_fn = ConsoleNoiseFilter(
            base_filter=lambda record: True,
            level_overrides={},
            high_volume_policies=policy,
            time_func=fake_time,
        )

        assert filter_fn(_make_record()) is True
        assert filter_fn(_make_record()) is True
        # Third call within the same window should be filtered out
        assert filter_fn(_make_record()) is False

        # Advance time beyond the window and ensure events are allowed again
        current_time["value"] = 61.0
        assert filter_fn(_make_record()) is True
