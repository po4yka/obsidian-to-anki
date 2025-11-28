from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.preflight import PreflightChecker


@pytest.fixture
def mock_config():
    config = Mock(spec=Config)
    config.vault_path = Path("/tmp/vault")
    config.source_dir = Path("source")
    config.source_subdirs = None
    config.db_path = Path("/tmp/db/state.db")
    config.project_log_dir = Path("/tmp/logs")
    config.llm_provider = "ollama"
    config.use_agent_system = False
    config.anki_connect_url = "http://localhost:8765"
    config.anki_note_type = "APF::Simple"
    config.anki_deck_name = "Test Deck"
    return config


@pytest.fixture
def checker(mock_config):
    return PreflightChecker(mock_config)


def test_check_git_repo_exists(checker):
    with patch("pathlib.Path.exists") as mock_exists:
        # vault/.git exists
        mock_exists.return_value = True

        checker._check_git_repo()

        assert len(checker.results) == 1
        assert checker.results[0].name == "Git Repository"
        assert checker.results[0].passed is True


def test_check_git_repo_missing(checker):
    with patch("pathlib.Path.exists") as mock_exists:
        # vault/.git does not exist
        mock_exists.return_value = False

        checker._check_git_repo()

        assert len(checker.results) == 1
        assert checker.results[0].name == "Git Repository"
        assert checker.results[0].passed is False
        assert checker.results[0].severity == "warning"


def test_check_vault_structure_valid(checker):
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True

        checker._check_vault_structure()

        assert len(checker.results) == 1
        assert checker.results[0].name == "Vault Structure"
        assert checker.results[0].passed is True


def test_check_disk_space_warning(checker):
    with (
        patch("shutil.disk_usage") as mock_disk_usage,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.resolve", side_effect=lambda: Path("/tmp/resolved")),
    ):
        # 200MB free (between 100MB and 500MB)
        # total, used, free
        mock_disk_usage.return_value = (1000, 500, 200 * 1024 * 1024)

        checker._check_disk_space()

        # Should have results for Database and Logs (deduplicated if paths resolve to same)
        # In our mock, resolve returns same path, so only 1 result expected
        assert len(checker.results) >= 1
        assert any(r.severity == "warning" for r in checker.results)
        assert any("Low disk space" in r.message for r in checker.results)


def test_check_memory_warning(checker):
    with patch("psutil.virtual_memory") as mock_memory:
        # 2GB available (< 4GB)
        mock_memory.return_value = Mock(available=2 * 1024 * 1024 * 1024)

        checker._check_memory()

        assert len(checker.results) == 1
        assert checker.results[0].name == "System Memory"
        assert checker.results[0].passed is False
        assert checker.results[0].severity == "warning"


def test_check_network_latency_high(checker):
    with (
        patch("obsidian_anki_sync.anki.client.AnkiClient") as MockClient,
        patch("time.time") as mock_time,
    ):
        # Mock context manager
        mock_client_instance = MockClient.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance

        # Mock time to simulate 300ms latency
        # First call (start) -> 0
        # Second call (end) -> 0.3
        mock_time.side_effect = [1000.0, 1000.3]

        checker._check_network_latency(check_anki=True, check_llm=False)

        assert len(checker.results) == 1
        assert checker.results[0].name == "Anki Latency"
        assert checker.results[0].passed is False
        assert "High latency" in checker.results[0].message
