"""Tests for APF generator helper behaviour."""

from datetime import datetime
from pathlib import Path

import pytest

from obsidian_anki_sync.apf.generator import APFGenerator
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    """Prevent real OpenAI client instantiation."""

    class DummyClient:
        pass

    monkeypatch.setattr(
        "obsidian_anki_sync.apf.generator.OpenAI", lambda *args, **kwargs: DummyClient()
    )


@pytest.fixture
def dummy_config(tmp_path):
    """Create a minimal config for APFGenerator."""
    # Create source directory in vault
    source_dir = tmp_path / "interview_questions" / "InterviewQuestions"
    source_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        vault_path=tmp_path,
        source_dir=Path("interview_questions/InterviewQuestions"),
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Test Deck",
        anki_note_type="APF::Simple",
        openrouter_api_key="dummy",
        openrouter_model="mock-model",
        llm_temperature=0.1,
        llm_top_p=0.2,
        run_mode="dry-run",
        delete_mode="delete",
        db_path=tmp_path / "test.db",
        log_level="INFO",
    )


@pytest.fixture
def sample_metadata_for_code():
    return NoteMetadata(
        id="test",
        title="Sample",
        topic="kotlin",
        language_tags=["en"],
        created=datetime.utcnow(),
        updated=datetime.utcnow(),
        tags=["kotlin", "testing"],
    )


@pytest.fixture
def plain_metadata():
    return NoteMetadata(
        id="plain",
        title="Plain",
        topic="general",
        language_tags=["en"],
        created=datetime.utcnow(),
        updated=datetime.utcnow(),
        tags=[],
    )


@pytest.fixture
def sample_manifest():
    return Manifest(
        slug="sample-slug-en",
        slug_base="sample-slug",
        lang="en",
        source_path="relative/path.md",
        source_anchor="p01",
        note_id="test",
        note_title="Sample",
        card_index=1,
        guid="guid-sample",
    )


@pytest.fixture
def sample_qa_pair_for_prompt():
    return QAPair(
        card_index=1,
        question_en="What is code?",
        question_ru="Что такое код?",
        answer_en="```\nkotlin code\n```",
        answer_ru="код",
    )


def test_code_language_hint_detects_known_language(
    dummy_config, sample_metadata_for_code
) -> None:
    gen = APFGenerator(dummy_config)
    assert gen._code_language_hint(sample_metadata_for_code) == "kotlin"


def test_code_language_hint_falls_back_to_plaintext(
    dummy_config, plain_metadata
) -> None:
    gen = APFGenerator(dummy_config)
    assert gen._code_language_hint(plain_metadata) == "plaintext"


def test_prompt_includes_code_language_instruction(
    dummy_config, sample_metadata_for_code, sample_qa_pair_for_prompt, sample_manifest
) -> None:
    gen = APFGenerator(dummy_config)
    prompt = gen._build_user_prompt(
        question=sample_qa_pair_for_prompt.question_en,
        answer=sample_qa_pair_for_prompt.answer_en,
        qa_pair=sample_qa_pair_for_prompt,
        metadata=sample_metadata_for_code,
        manifest=sample_manifest,
        lang="en",
    )
    assert '<pre><code class="language-kotlin"' in prompt


def test_ensure_manifest_updates_guid_and_tags(dummy_config, sample_manifest) -> None:
    gen = APFGenerator(dummy_config)
    html = '<!-- manifest: {"slug":"old","guid":"bad","tags":[]} -->'
    updated = gen._ensure_manifest(
        html, sample_manifest, ["en", "testing"], "APF::Simple"
    )
    assert '"guid":"guid-sample"' in updated
    assert '"slug":"sample-slug-en"' in updated
    assert '"tags":["en","testing"]' in updated


def test_normalize_code_blocks_converts_markdown(
    dummy_config, sample_metadata_for_code
) -> None:
    gen = APFGenerator(dummy_config)
    html = 'Intro\n```kotlin\nprintln("hi")\n```\nOutro'
    normalized = gen._normalize_code_blocks(
        html, gen._code_language_hint(sample_metadata_for_code)
    )
    assert "```" not in normalized
    assert '<pre><code class="language-kotlin">' in normalized
    assert 'println("hi")' in normalized


def test_normalize_code_blocks_uses_default_when_missing_lang(
    dummy_config, plain_metadata
) -> None:
    gen = APFGenerator(dummy_config)
    html = """```
val x = 1
```"""
    normalized = gen._normalize_code_blocks(
        html, gen._code_language_hint(plain_metadata)
    )
    assert "```" not in normalized
    assert '<pre><code class="language-plaintext">' in normalized
    assert "val x = 1" in normalized


def test_extract_tags_sanitizes_slashes(dummy_config) -> None:
    metadata = NoteMetadata(
        id="t-1",
        title="Tagged",
        topic="Android",
        language_tags=["en"],
        created=datetime.utcnow(),
        updated=datetime.utcnow(),
        subtopics=["ui-compose"],
        tags=["difficulty/easy", "android/ui-compose", "lang/ru"],
    )

    gen = APFGenerator(dummy_config)
    tags = gen._extract_tags(metadata, "en")

    assert "en" in tags
    assert "android" in tags
    assert "ui-compose" in tags
    assert "difficulty_easy" in tags
    assert "android_ui-compose" in tags
    assert "lang_ru" in tags
    assert all("/" not in tag for tag in tags)
