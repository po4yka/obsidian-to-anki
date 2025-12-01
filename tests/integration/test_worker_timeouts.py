"""Integration tests for worker stage-level SLA enforcement."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from obsidian_anki_sync.agents.models import (
    AgentPipelineResult,
    GenerationResult,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.worker import process_note_job


class DummyOrchestrator:
    """Minimal orchestrator stub that returns a prepared pipeline result."""

    def __init__(self, pipeline_result: AgentPipelineResult) -> None:
        self._result = pipeline_result

    async def process_note(self, *args: Any, **kwargs: Any) -> AgentPipelineResult:
        return self._result

    def convert_to_cards(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _build_pipeline_result(
    *,
    generation_seconds: float,
    validation_seconds: float,
    success: bool = True,
) -> AgentPipelineResult:
    """Construct a minimal AgentPipelineResult with custom stage timings."""
    pre = PreValidationResult(is_valid=True, error_type="none", validation_time=0.0)
    generation = GenerationResult(
        cards=[],
        total_cards=0,
        generation_time=generation_seconds,
        model_used="test-model",
    )
    post = PostValidationResult(
        is_valid=success,
        error_type="none",
        error_details="",
        corrected_cards=None,
        validation_time=validation_seconds,
    )
    stage_times = {
        "generation": generation_seconds,
        "post_validation": validation_seconds,
    }
    return AgentPipelineResult(
        success=success,
        pre_validation=pre,
        generation=generation,
        post_validation=post,
        memorization_quality=None,
        highlight_result=None,
        total_time=generation_seconds + validation_seconds,
        retry_count=0,
        stage_times=stage_times,
        last_error=None,
    )


@pytest.mark.asyncio()
async def test_worker_flags_generation_sla(tmp_path: Path) -> None:
    """Generation overruns should be surfaced as actionable errors."""
    note_path = tmp_path / "note.md"
    note_path.write_text("# demo")

    result = _build_pipeline_result(generation_seconds=500.0, validation_seconds=5.0)
    metadata_dict = {
        "id": "note-gen",
        "title": "Demo",
        "topic": "demo",
        "language_tags": ["en"],
        "created": datetime.now(tz=UTC).isoformat(),
        "updated": datetime.now(tz=UTC).isoformat(),
    }
    qa_dict = {
        "card_index": 1,
        "question_en": "Q?",
        "question_ru": "Q?",
        "answer_en": "A.",
        "answer_ru": "A.",
    }
    ctx = {
        "config": SimpleNamespace(
            worker_generation_timeout_seconds=100.0,
            worker_validation_timeout_seconds=200.0,
        ),
        "orchestrator": DummyOrchestrator(result),
    }

    response = await process_note_job(
        ctx,
        str(note_path),
        "note.md",
        metadata_dict=metadata_dict,
        qa_pairs_dicts=[qa_dict],
    )

    assert response["success"] is False
    assert "generation" in response["error"]


@pytest.mark.asyncio()
async def test_worker_flags_validation_sla(tmp_path: Path) -> None:
    """Post-validation overruns should be reported with stage metadata."""
    note_path = tmp_path / "note.md"
    note_path.write_text("# demo")

    result = _build_pipeline_result(generation_seconds=5.0, validation_seconds=999.0)
    metadata_dict = {
        "id": "note-val",
        "title": "Demo",
        "topic": "demo",
        "language_tags": ["en"],
        "created": datetime.now(tz=UTC).isoformat(),
        "updated": datetime.now(tz=UTC).isoformat(),
    }
    qa_dict = {
        "card_index": 1,
        "question_en": "Q?",
        "question_ru": "Q?",
        "answer_en": "A.",
        "answer_ru": "A.",
    }
    ctx = {
        "config": SimpleNamespace(
            worker_generation_timeout_seconds=200.0,
            worker_validation_timeout_seconds=100.0,
        ),
        "orchestrator": DummyOrchestrator(result),
    }

    response = await process_note_job(
        ctx,
        str(note_path),
        "note.md",
        metadata_dict=metadata_dict,
        qa_pairs_dicts=[qa_dict],
    )

    assert response["success"] is False
    assert "post_validation" in response["error"]
