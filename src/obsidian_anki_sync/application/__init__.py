"""Application layer for the Obsidian to Anki sync service.

This package contains use cases, application services, and dependency injection
that orchestrate domain logic and coordinate with external interfaces.
"""

from .container import DependencyContainer, get_container, inject
from .factories.component_factory import ComponentFactory
from .services.agent_pipeline_orchestrator import AgentPipelineOrchestrator
from .services.note_discovery_service import NoteDiscoveryService
from .services.retry_handler import RetryConfig, RetryHandler, RetryResult
from .services.sync_orchestrator import (
    SyncOrchestrator,
    SyncOrchestratorConfig,
    SyncResult,
)
from .use_cases.apply_changes import ApplyChangesUseCase
from .use_cases.determine_sync_actions import DetermineSyncActionsUseCase
from .use_cases.generate_cards import GenerateCardsUseCase
from .use_cases.process_notes import ProcessNotesUseCase
from .use_cases.sync_notes import SyncNotesUseCase

__all__ = [
    # DI Container
    "DependencyContainer",
    "get_container",
    "inject",
    # Factories
    "ComponentFactory",
    # Services
    "AgentPipelineOrchestrator",
    "NoteDiscoveryService",
    "RetryHandler",
    "RetryConfig",
    "RetryResult",
    "SyncOrchestrator",
    "SyncOrchestratorConfig",
    "SyncResult",
    # Use Cases
    "ApplyChangesUseCase",
    "DetermineSyncActionsUseCase",
    "GenerateCardsUseCase",
    "ProcessNotesUseCase",
    "SyncNotesUseCase",
]
