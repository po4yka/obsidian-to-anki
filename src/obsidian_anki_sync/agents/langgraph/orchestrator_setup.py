"""Helper utilities for LangGraph orchestrator setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obsidian_anki_sync.agents.agent_monitoring import (
    PerformanceTracker,
    get_pipeline_performance_tracker,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestratorResources:
    """Container for orchestrator auxiliary resources."""

    performance_tracker: PerformanceTracker | None
    memory_store: Any | None
    advanced_memory_store: Any | None
    observability: Any | None
    rag_integration: Any | None
    enable_rag: bool


def resolve_agent_framework(config, agent_framework: str | None) -> str:
    """Resolve agent framework with config fallback."""
    return (
        agent_framework
        if agent_framework is not None
        else getattr(config, "agent_framework", "pydantic_ai")
    )


def init_performance_tracker(config) -> PerformanceTracker | None:
    """Initialize performance tracker."""
    return get_pipeline_performance_tracker(config)


def init_memory_stores(config):
    """Initialize optional memory stores."""
    try:
        from obsidian_anki_sync.agents.agent_memory import AgentMemoryStore
    except ImportError:
        AgentMemoryStore = None

    try:
        from obsidian_anki_sync.agents.advanced_memory import AdvancedMemoryStore
    except ImportError:
        AdvancedMemoryStore = None

    memory_store = None
    advanced_memory_store = None

    if getattr(config, "enable_agent_memory", True) and AgentMemoryStore:
        try:
            memory_storage_path = getattr(
                config, "memory_storage_path", Path(".agent_memory")
            )
            enable_semantic_search = getattr(config, "enable_semantic_search", True)
            memory_store = AgentMemoryStore(
                storage_path=memory_storage_path,
                config=config,
                enable_semantic_search=enable_semantic_search,
            )
            logger.info(
                "langgraph_memory_store_initialized",
                path=str(memory_storage_path),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("langgraph_memory_store_init_failed", error=str(exc))

    if getattr(config, "use_advanced_memory", False) and AdvancedMemoryStore:
        try:
            mongodb_url = getattr(config, "mongodb_url", "mongodb://localhost:27017")
            memory_db_name = getattr(config, "memory_db_name", "obsidian_anki_memory")
            advanced_memory_store = AdvancedMemoryStore(
                config=config,
                mongodb_url=mongodb_url,
                db_name=memory_db_name,
                embedding_store=memory_store,
            )
            logger.info("advanced_memory_store_deferred_connection")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("advanced_memory_store_init_failed", error=str(exc))

    return memory_store, advanced_memory_store


def init_observability(config):
    """Initialize optional enhanced observability system."""
    try:
        from obsidian_anki_sync.agents.enhanced_observability import (
            EnhancedObservabilitySystem,
        )
    except ImportError:
        EnhancedObservabilitySystem = None

    if not (
        getattr(config, "enable_enhanced_observability", False)
        and EnhancedObservabilitySystem
    ):
        return None

    try:
        system = EnhancedObservabilitySystem(config)
        logger.info("enhanced_observability_system_initialized")
        return system
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("enhanced_observability_init_failed", error=str(exc))
        return None


def init_rag(config):
    """Initialize optional RAG integration."""
    try:
        from obsidian_anki_sync.rag.integration import (
            RAGIntegration,
            get_rag_integration,
        )
    except ImportError:
        RAGIntegration = None
        get_rag_integration = None

    enable_rag = getattr(config, "rag_enabled", False)
    if not (enable_rag and get_rag_integration is not None):
        return None, enable_rag

    try:
        rag_integration = get_rag_integration(config)
        if rag_integration.is_enabled:
            logger.info(
                "rag_integration_initialized",
                context_enrichment=getattr(config, "rag_context_enrichment", True),
                duplicate_detection=getattr(config, "rag_duplicate_detection", True),
                few_shot_examples=getattr(config, "rag_few_shot_examples", True),
            )
        else:
            logger.info("rag_integration_disabled")
        return rag_integration, enable_rag
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("rag_integration_init_failed", error=str(exc))
        return None, enable_rag


def build_resources(
    config,
    *,
    agent_framework: str | None = None,
) -> OrchestratorResources:
    """Construct all orchestrator auxiliary resources."""
    performance_tracker = init_performance_tracker(config)
    memory_store, advanced_memory_store = init_memory_stores(config)
    observability = init_observability(config)
    rag_integration, enable_rag = init_rag(config)

    return OrchestratorResources(
        performance_tracker=performance_tracker,
        memory_store=memory_store,
        advanced_memory_store=advanced_memory_store,
        observability=observability,
        rag_integration=rag_integration,
        enable_rag=enable_rag,
    )


__all__ = [
    "OrchestratorResources",
    "build_resources",
    "init_memory_stores",
    "init_observability",
    "init_performance_tracker",
    "init_rag",
    "resolve_agent_framework",
]
