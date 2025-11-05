"""CLI inspection command tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from obsidian_anki_sync.cli import app
from obsidian_anki_sync.config import Config


@pytest.fixture
def test_config(tmp_path):
    return Config(
        vault_path=tmp_path,
        source_dir=Path("InterviewQuestions"),
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Deck",
        anki_note_type="APF::Simple",
        openrouter_api_key="dummy",
        openrouter_model="mock",
        llm_temperature=0.1,
        llm_top_p=0.2,
        run_mode="apply",
        delete_mode="delete",
        db_path=tmp_path / "test.db",
        log_level="INFO",
    )


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def patch_logging(monkeypatch):
    monkeypatch.setattr("obsidian_anki_sync.cli.configure_logging", lambda level: None)
    monkeypatch.setattr("obsidian_anki_sync.cli.get_logger", lambda name: MagicMock())


def _patch_setup(monkeypatch, test_config, decks=None, models=None, fields=None):
    monkeypatch.setattr(
        "obsidian_anki_sync.cli.load_config", lambda path=None: test_config
    )
    monkeypatch.setattr("obsidian_anki_sync.cli.set_config", lambda cfg: None)

    class DummyAnki:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def get_deck_names(self):
            if decks is not None:
                return decks
            return []

        def get_model_names(self):
            if models is not None:
                return models
            return []

        def get_model_field_names(self, model_name):
            if fields is None:
                return []
            return fields.get(model_name, [])

    monkeypatch.setattr("obsidian_anki_sync.anki.client.AnkiClient", DummyAnki)


def test_decks_command_lists_names(runner, test_config, monkeypatch):
    _patch_setup(monkeypatch, test_config, decks=["Deck B", "Deck A"])
    result = runner.invoke(app, ["decks"])
    assert result.exit_code == 0
    assert "Deck A" in result.output
    assert "Deck B" in result.output


def test_models_command_lists_names(runner, test_config, monkeypatch):
    _patch_setup(monkeypatch, test_config, models=["ModelB", "ModelA"])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "ModelB" in result.output


def test_model_fields_command_shows_fields(runner, test_config, monkeypatch):
    _patch_setup(monkeypatch, test_config, fields={"Basic": ["Front", "Back"]})
    result = runner.invoke(app, ["model-fields", "--model", "Basic"])
    assert result.exit_code == 0
    assert "Front" in result.output
    assert "Back" in result.output


def test_format_command_runs_subprocess(runner, test_config, monkeypatch):
    _patch_setup(monkeypatch, test_config)

    calls = []

    def fake_run(cmd, check):
        calls.append((tuple(cmd), check))
        return MagicMock()

    monkeypatch.setattr("obsidian_anki_sync.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["format", "--check"])
    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0][1] is True
