"""Tests for agent temperature resolution logic."""

from __future__ import annotations

import sys
import types

# Provide lightweight stubs when LangGraph is not installed locally.
if "langgraph" not in sys.modules:  # pragma: no cover - test bootstrap
    langgraph_stub = types.ModuleType("langgraph")
    sys.modules["langgraph"] = langgraph_stub

    checkpoint_pkg = types.ModuleType("langgraph.checkpoint")
    sys.modules["langgraph.checkpoint"] = checkpoint_pkg

    checkpoint_memory = types.ModuleType("langgraph.checkpoint.memory")

    class _FakeMemorySaver:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    # type: ignore[attr-defined]
    checkpoint_memory.MemorySaver = _FakeMemorySaver
    sys.modules["langgraph.checkpoint.memory"] = checkpoint_memory

    graph_module = types.ModuleType("langgraph.graph")

    class _FakeStateGraph:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def add_node(self, *args: object, **kwargs: object) -> "_FakeStateGraph":
            return self

        def add_edge(self, *args: object, **kwargs: object) -> "_FakeStateGraph":
            return self

    graph_module.StateGraph = _FakeStateGraph  # type: ignore[attr-defined]
    graph_module.END = object()
    sys.modules["langgraph.graph"] = graph_module

if "bs4" not in sys.modules:  # pragma: no cover - test bootstrap
    bs4_module = types.ModuleType("bs4")

    class _FakeBeautifulSoup:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.contents: list[object] = []

        def find_all(self, *args: object, **kwargs: object) -> list[object]:
            return self.contents

    bs4_module.BeautifulSoup = _FakeBeautifulSoup  # type: ignore[attr-defined]
    sys.modules["bs4"] = bs4_module

from obsidian_anki_sync.agents.orchestrator import resolve_agent_temperature


class ConfigStub:
    """Minimal config stub providing temperature overrides."""

    def __init__(
        self,
        *,
        temps: dict[str, float] | None = None,
        has_task_method: bool = True,
        **attrs: float | None,
    ):
        self.__dict__.update(attrs)
        self._temps = temps or {}
        if has_task_method:
            # type: ignore[attr-defined]
            self.get_model_config_for_task = self._get_model_config_for_task

    def _get_model_config_for_task(self, task: str) -> dict[str, float]:
        """Return mock task config."""
        if task not in self._temps:
            return {}
        return {"temperature": self._temps[task]}


def test_resolve_agent_temperature_prefers_override() -> None:
    """Override temperature should take precedence."""
    config = ConfigStub(
        pre_validator_temperature=0.42,
        temps={"pre_validation": 0.1},
    )

    result = resolve_agent_temperature(
        config,
        "pre_validator_temperature",
        "pre_validation",
        0.0,
    )

    assert result == 0.42


def test_resolve_agent_temperature_uses_task_config_when_override_none() -> None:
    """Preset temperature should be used when override is None."""
    config = ConfigStub(
        pre_validator_temperature=None,
        temps={"pre_validation": 0.25},
    )

    result = resolve_agent_temperature(
        config,
        "pre_validator_temperature",
        "pre_validation",
        0.0,
    )

    assert result == 0.25


def test_resolve_agent_temperature_defaults_without_sources() -> None:
    """Fallback to default when neither override nor preset is available."""
    config = ConfigStub(
        pre_validator_temperature=None,
        temps={},
    )

    result = resolve_agent_temperature(
        config,
        "pre_validator_temperature",
        "pre_validation",
        0.0,
    )

    assert result == 0.0


def test_resolve_agent_temperature_handles_missing_model_config_method() -> None:
    """Fallback to default when config lacks get_model_config_for_task."""
    config = ConfigStub(
        pre_validator_temperature=None,
        has_task_method=False,
    )

    result = resolve_agent_temperature(
        config,
        "pre_validator_temperature",
        "pre_validation",
        0.0,
    )

    assert result == 0.0


def test_resolve_agent_temperature_handles_invalid_override_value() -> None:
    """Invalid override types should not crash and fall back to default."""
    config = ConfigStub(
        pre_validator_temperature="high",  # type: ignore[arg-type]
        temps={"pre_validation": 0.75},
    )

    result = resolve_agent_temperature(
        config,
        "pre_validator_temperature",
        "pre_validation",
        0.0,
    )

    assert result == 0.0
