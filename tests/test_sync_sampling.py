"""Tests for sampling behaviour in sync engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.models import Card, Manifest
from obsidian_anki_sync.sync.engine import SyncEngine


@pytest.fixture
def sample_note_list():
    """Create a list of fake note paths."""
    return [(Path(f"/tmp/note{i}.md"), f"relative/note{i}.md") for i in range(20)]


def _make_fake_card(
    relative_path: str,
    lang: str,
    qa_pair_index: int,
    note_id: str,
    note_title: str,
    counter: int,
) -> Card:
    slug = f"{relative_path.replace('/', '-')}-{lang}-{counter}"
    manifest = Manifest(
        slug=slug,
        slug_base=slug,
        lang=lang,
        source_path=relative_path,
        source_anchor=f"p{qa_pair_index:02d}",
        note_id=note_id,
        note_title=note_title,
        card_index=qa_pair_index,
        guid=f"guid-{counter}-{lang}",
        hash6=None,
    )
    return Card(
        slug=slug,
        lang=lang,
        apf_html="<html></html>",
        manifest=manifest,
        content_hash=f"hash-{counter}",
        note_type="APF::Simple",
        tags=[],
        guid=f"guid-{counter}-{lang}",
    )


def test_scan_sampling_limits_number_of_notes(
    sample_note_list, sample_metadata, sample_qa_pair, test_config
) -> None:
    """Ensure only the requested number of notes are parsed when sampling."""
    from obsidian_anki_sync.anki.client import AnkiClient
    from obsidian_anki_sync.sync.state_db import StateDB

    db = MagicMock(spec=StateDB)
    anki = MagicMock(spec=AnkiClient)
    engine = SyncEngine(test_config, db, anki)

    with (
        patch(
            "obsidian_anki_sync.sync.engine.discover_notes",
            return_value=sample_note_list,
        ),
        patch(
            "obsidian_anki_sync.sync.engine.random.sample",
            side_effect=lambda seq, k: seq[:k],
        ),
        patch(
            "obsidian_anki_sync.sync.engine.parse_note",
            return_value=(sample_metadata, [sample_qa_pair]),
        ) as mock_parse,
    ):
        counter = {"value": 0}

        def fake_generate(
            qa_pair,
            metadata,
            relative_path,
            lang,
            existing_slugs,
            note_content="",
            all_qa_pairs=None,
        ) -> None:
            card = _make_fake_card(
                relative_path,
                lang,
                qa_pair.card_index,
                metadata.id,
                metadata.title,
                counter["value"],
            )
            counter["value"] += 1
            return card

        with patch.object(SyncEngine, "_generate_card", side_effect=fake_generate):
            cards = engine._scan_obsidian_notes(sample_size=5)

    # parse_note should be called exactly 5 times (once per sampled note)
    assert mock_parse.call_count == 5
    # ensure resulting cards correspond to languages processed (>=5 entries)
    assert len(cards) >= 5
