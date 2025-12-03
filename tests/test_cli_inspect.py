"""CLI inspection command tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from obsidian_anki_sync.cli import app
from obsidian_anki_sync.config import Config


@pytest.fixture
def sample_note_content():
    """Sample note content for CLI lint tests."""
    return """---
id: test-001
title: Test Question
topic: Testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-02
---

# Question (EN)

> What is unit testing?

# Вопрос (RU)

> Что такое юнит-тестирование?

---

## Answer (EN)

Unit testing is testing individual components.

## Ответ (RU)

Юнит-тестирование - это тестирование отдельных компонентов.
"""


@pytest.fixture
def test_config(tmp_path):
    source_dir = Path("InterviewQuestions")
    (tmp_path / source_dir).mkdir(parents=True, exist_ok=True)
    return Config(
        vault_path=tmp_path,
        source_dir=source_dir,
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
def _patch_logging(monkeypatch):
    monkeypatch.setattr(
        "obsidian_anki_sync.cli_commands.shared.configure_logging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "obsidian_anki_sync.cli_commands.shared.get_logger", lambda name: MagicMock()
    )


def _patch_setup(
    monkeypatch, test_config, decks=None, models=None, fields=None
) -> None:
    monkeypatch.setattr(
        "obsidian_anki_sync.cli_commands.shared.load_config",
        lambda path=None: test_config,
    )
    monkeypatch.setattr(
        "obsidian_anki_sync.cli_commands.shared.set_config", lambda cfg: None
    )

    class DummyAnki:
        def __init__(self, url) -> None:
            self.url = url

        def __enter__(self) -> None:
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            return False

        def get_deck_names(self) -> None:
            if decks is not None:
                return decks
            return []

        def get_model_names(self) -> None:
            if models is not None:
                return models
            return []

        def get_model_field_names(self, model_name) -> None:
            if fields is None:
                return []
            return fields.get(model_name, [])

    monkeypatch.setattr("obsidian_anki_sync.anki.client.AnkiClient", DummyAnki)


def test_decks_command_lists_names(runner, test_config, monkeypatch) -> None:
    _patch_setup(monkeypatch, test_config, decks=["Deck B", "Deck A"])
    result = runner.invoke(app, ["decks"])
    assert result.exit_code == 0
    assert "Deck A" in result.output
    assert "Deck B" in result.output


def test_models_command_lists_names(runner, test_config, monkeypatch) -> None:
    _patch_setup(monkeypatch, test_config, models=["ModelB", "ModelA"])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "ModelA" in result.output
    assert "ModelB" in result.output


def test_model_fields_command_shows_fields(runner, test_config, monkeypatch) -> None:
    _patch_setup(monkeypatch, test_config, fields={"Basic": ["Front", "Back"]})
    result = runner.invoke(app, ["model-fields", "--model", "Basic"])
    assert result.exit_code == 0
    assert "Front" in result.output
    assert "Back" in result.output


def test_format_command_runs_subprocess(runner, test_config, monkeypatch) -> None:
    _patch_setup(monkeypatch, test_config)

    calls = []

    def fake_run(cmd, check) -> None:
        calls.append((tuple(cmd), check))
        return MagicMock()

    monkeypatch.setattr("obsidian_anki_sync.cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["format", "--check"])
    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0][1] is True


def test_lint_note_passes(
    runner, test_config, sample_note_content, tmp_path, monkeypatch
) -> None:
    _patch_setup(monkeypatch, test_config)
    note_path = tmp_path / "note.md"
    note_path.write_text(sample_note_content)

    result = runner.invoke(app, ["lint-note", str(note_path)])

    assert result.exit_code == 0
    assert "No lint issues detected" in result.output


def test_lint_note_detects_missing_answer(
    runner, test_config, sample_note_content, tmp_path, monkeypatch
) -> None:
    _patch_setup(monkeypatch, test_config)
    note_path = tmp_path / "broken.md"
    broken_content = sample_note_content.replace(
        "## Answer (EN)", "## Answer Missing (EN)", 1
    )
    note_path.write_text(broken_content)

    result = runner.invoke(app, ["lint-note", str(note_path)])

    assert result.exit_code == 1
    assert "Missing '## Answer (EN)'" in result.output
