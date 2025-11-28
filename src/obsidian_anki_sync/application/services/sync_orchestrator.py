"""Application service for orchestrating the sync workflow."""

from dataclasses import dataclass
from typing import Any

from ...domain.interfaces.anki_client import IAnkiClient
from ...domain.interfaces.card_generator import ICardGenerator
from ...domain.interfaces.note_parser import INoteParser
from ...domain.interfaces.state_repository import IStateRepository
from ...infrastructure.cache.cache_manager import CacheManager
from ...utils.logging import get_logger
from ..use_cases.apply_changes import ApplyChangesUseCase
from ..use_cases.determine_sync_actions import DetermineSyncActionsUseCase
from ..use_cases.generate_cards import GenerateCardsUseCase
from ..use_cases.sync_notes import SyncNotesRequest, SyncNotesUseCase

logger = get_logger(__name__)


@dataclass
class SyncOrchestratorConfig:
    """Configuration for the sync orchestrator."""

    dry_run: bool = False
    sample_size: int | None = None
    incremental: bool = False
    build_index: bool = True


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    processed_notes: int
    generated_cards: int
    applied_changes: int
    errors: list[str]
    stats: dict[str, Any]


class SyncOrchestrator:
    """Orchestrates the complete sync workflow using use cases.

    This class follows the Single Responsibility Principle by focusing
    solely on orchestration - coordinating use cases and managing
    the high-level sync workflow.
    """

    def __init__(
        self,
        note_parser: INoteParser,
        card_generator: ICardGenerator,
        state_repository: IStateRepository,
        anki_client: IAnkiClient,
        cache_manager: CacheManager,
    ):
        """Initialize orchestrator with dependencies.

        Args:
            note_parser: Parser for Obsidian notes
            card_generator: Generator for Anki cards
            state_repository: Repository for state persistence
            anki_client: Client for Anki communication
            cache_manager: Manager for persistent caches
        """
        self.note_parser = note_parser
        self.card_generator = card_generator
        self.state_repository = state_repository
        self.anki_client = anki_client
        self.cache_manager = cache_manager

        # Initialize use cases
        self.sync_notes_use_case = SyncNotesUseCase(
            note_parser, card_generator, state_repository, anki_client
        )
        self.generate_cards_use_case = GenerateCardsUseCase(card_generator)
        self.determine_actions_use_case = DetermineSyncActionsUseCase()
        self.apply_changes_use_case = ApplyChangesUseCase(
            anki_client, state_repository)

        logger.info("sync_orchestrator_initialized")

    def execute_sync(self, config: SyncOrchestratorConfig) -> SyncResult:
        """Execute the complete sync workflow.

        Args:
            config: Sync configuration

        Returns:
            Sync result with statistics and status
        """
        logger.info(
            "sync_orchestration_started",
            dry_run=config.dry_run,
            sample_size=config.sample_size,
            incremental=config.incremental,
        )

        errors = []
        stats = {
            "processed_notes": 0,
            "generated_cards": 0,
            "applied_changes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        try:
            # Step 1: Sync notes using the dedicated use case
            sync_request = SyncNotesRequest(
                dry_run=config.dry_run,
                sample_size=config.sample_size,
                incremental=config.incremental,
                build_index=config.build_index,
            )

            sync_response = self.sync_notes_use_case.execute(sync_request)

            if not sync_response.success:
                errors.extend(sync_response.errors)
                return SyncResult(
                    success=False,
                    processed_notes=sync_response.stats.get("processed", 0),
                    generated_cards=sync_response.stats.get("generated", 0),
                    applied_changes=0,
                    errors=errors,
                    stats=stats,
                )

            # Update stats from sync response
            stats["processed_notes"] = sync_response.stats.get("processed", 0)
            stats["generated_cards"] = sync_response.stats.get("generated", 0)
            stats["applied_changes"] = sync_response.stats.get("applied", 0)

            # Get cache statistics
            cache_stats = self.cache_manager.get_cache_stats()
            stats["cache_hits"] = cache_stats.get("hits", 0)
            stats["cache_misses"] = cache_stats.get("misses", 0)

            success = len(errors) == 0

            logger.info(
                "sync_orchestration_completed",
                success=success,
                processed_notes=stats["processed_notes"],
                generated_cards=stats["generated_cards"],
                applied_changes=stats["applied_changes"],
                error_count=len(errors),
            )

            return SyncResult(
                success=success,
                processed_notes=stats["processed_notes"],
                generated_cards=stats["generated_cards"],
                applied_changes=stats["applied_changes"],
                errors=errors,
                stats=stats,
            )

        except Exception as e:
            error_msg = f"Sync orchestration failed: {e}"
            logger.error("sync_orchestration_failed", error=str(e))
            errors.append(error_msg)

            return SyncResult(
                success=False,
                processed_notes=stats["processed_notes"],
                generated_cards=stats["generated_cards"],
                applied_changes=stats["applied_changes"],
                errors=errors,
                stats=stats,
            )

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get current cache statistics.

        Returns:
            Cache statistics dictionary
        """
        return self.cache_manager.get_cache_stats()

    def get_component_status(self) -> dict[str, Any]:
        """Get status of all components.

        Returns:
            Component status dictionary
        """
        status: dict[str, Any] = {}

        try:
            status["anki_client"] = {
                "connected": self.anki_client.check_connection()}
        except Exception as e:
            status["anki_client"] = {"error": str(e)}

        try:
            status["state_repository"] = {
                "accessible": True,  # Basic check
                "stats": self.state_repository.get_sync_stats(),
            }
        except Exception as e:
            status["state_repository"] = {"error": str(e)}

        try:
            status["cache_manager"] = self.cache_manager.get_cache_size_info()
        except Exception as e:
            status["cache_manager"] = {"error": str(e)}

        return status

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.cache_manager.close_caches()
            logger.debug("sync_orchestrator_cleanup_completed")
        except Exception as e:
            logger.warning("error_during_orchestrator_cleanup", error=str(e))
