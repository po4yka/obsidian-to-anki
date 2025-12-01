"""Regression harness for LangGraph post-validation retries."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from obsidian_anki_sync.agents.langgraph.nodes import post_validation_node
from obsidian_anki_sync.agents.models import GeneratedCard, PostValidationResult
from obsidian_anki_sync.models import NoteMetadata


@pytest.fixture()
def post_validation_state() -> Callable[[], dict]:
    """Factory that produces a minimal PipelineState for post-validation tests."""

    def _factory() -> dict:
        metadata = NoteMetadata(
            id="note-bfs-vs-dfs",
            title="Graph Traversal Strategies",
            topic="algorithms",
            tags=["algorithms", "graphs"],
            file_path="/vault/20-Algorithms/q-graph-algorithms-bfs-dfs.md",
            language_tags=["en"],
            created=datetime(2024, 1, 1, tzinfo=UTC),
            updated=datetime(2024, 1, 2, tzinfo=UTC),
        )
        card = GeneratedCard(
            card_index=1,
            slug="algorithms-graph-traversal-1-en",
            lang="en",
            apf_html="<div>How does BFS differ from DFS?</div>",
            confidence=0.42,
            content_hash="hash-a",
        )
        state = {
            "metadata_dict": metadata.model_dump(),
            "generation": {"cards": [card.model_dump()], "total_cards": 1},
            "stage_times": {},
            "messages": [],
            "errors": [],
            "retry_count": 0,
            "max_retries": 2,
            "auto_fix_enabled": True,
            "strict_mode": True,
            "linter_valid": True,
            "linter_results": [],
            "current_stage": "post_validation",
            "enable_context_enrichment": True,
            "step_count": 0,
            "max_steps": 5,
            "last_error": None,
            "last_error_severity": None,
        }
        return state

    return _factory


class _DummyValidator:
    """Deterministic validator used to simulate retry paths."""

    def __init__(self, responses: list[PostValidationResult]) -> None:
        self._responses = responses
        self.calls = 0

    async def validate(self, *args, **kwargs) -> PostValidationResult:
        """Return the next canned response."""
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


@pytest.mark.asyncio()
async def test_post_validation_node_retries_and_records_stage_time(
    post_validation_state: Callable[[], dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the LangGraph node retries and records timing data."""

    responses = [
        PostValidationResult(
            is_valid=False,
            error_type="semantic",
            error_details="Missing precision on BFS definition",
            corrected_cards=None,
            validation_time=0.01,
        ),
        PostValidationResult(
            is_valid=True,
            error_type="none",
            error_details="",
            corrected_cards=None,
            validation_time=0.02,
        ),
    ]
    dummy_validator = _DummyValidator(responses)
    monkeypatch.setattr(
        "obsidian_anki_sync.agents.langgraph.nodes.PostValidatorAgentAI",
        lambda *_, **__: dummy_validator,
    )
    monkeypatch.setattr(
        "obsidian_anki_sync.agents.langgraph.nodes.get_model",
        lambda *_: object(),
    )

    state = post_validation_state()
    first_pass_state = await post_validation_node(state)

    assert first_pass_state["retry_count"] == 1
    assert first_pass_state["current_stage"] == "post_validation"
    assert "post_validation" in first_pass_state["stage_times"]

    second_pass_state = await post_validation_node(first_pass_state)

    assert second_pass_state["current_stage"] == "context_enrichment"
    assert second_pass_state["post_validation"]["is_valid"] is True
