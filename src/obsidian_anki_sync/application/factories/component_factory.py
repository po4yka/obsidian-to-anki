"""Factory for creating sync engine components."""

from ...domain.interfaces.anki_client import IAnkiClient
from ...domain.interfaces.card_generator import ICardGenerator
from ...domain.interfaces.llm_provider import ILLMProvider
from ...domain.interfaces.note_parser import INoteParser
from ...domain.interfaces.state_repository import IStateRepository
from ...domain.interfaces.vault_config import IVaultConfig
from ...infrastructure.cache.cache_manager import CacheManager
from ...utils.logging import get_logger
from ..container import get_container
from ..services.note_discovery_service import NoteDiscoveryService

logger = get_logger(__name__)


class ComponentFactory:
    """Factory for creating and configuring sync engine components.

    This factory now uses the dependency injection container to resolve
    dependencies, following the Dependency Inversion Principle.
    """

    def __init__(self):
        """Initialize factory with DI container."""
        self.container = get_container()
        self._cache_manager = None

    def create_cache_manager(self) -> CacheManager:
        """Create and configure cache manager.

        Returns:
            Configured CacheManager instance
        """
        if self._cache_manager is None:
            # Get config from container
            config = self.container.resolve(IVaultConfig)
            db_path = getattr(config, "db_path", None)
            if not db_path:
                # Fallback - create with a default path
                from pathlib import Path

                db_path = Path(".sync_state.db")

            self._cache_manager = CacheManager(db_path)

        return self._cache_manager  # type: ignore[no-any-return]

    def create_anki_client(self) -> IAnkiClient:
        """Create Anki client instance.

        Returns:
            Configured IAnkiClient implementation
        """
        return self.container.resolve(IAnkiClient)  # type: ignore[no-any-return]

    def create_state_repository(self) -> IStateRepository:
        """Create state repository instance.

        Returns:
            Configured IStateRepository implementation
        """
        return self.container.resolve(IStateRepository)  # type: ignore[no-any-return]

    def create_llm_provider(self) -> ILLMProvider:
        """Create LLM provider instance.

        Returns:
            Configured ILLMProvider implementation
        """
        return self.container.resolve(ILLMProvider)  # type: ignore[no-any-return]

    def create_card_generator(self) -> ICardGenerator:
        """Create card generator instance.

        Returns:
            Configured ICardGenerator implementation
        """
        return self.container.resolve(ICardGenerator)  # type: ignore[no-any-return]

    def create_note_parser(self) -> INoteParser:
        """Create note parser instance.

        Returns:
            Configured INoteParser implementation
        """
        return self.container.resolve(INoteParser)  # type: ignore[no-any-return]

    def create_note_discovery_service(self) -> NoteDiscoveryService:
        """Create note discovery service.

        Returns:
            Configured NoteDiscoveryService instance
        """
        vault_config = self.container.resolve(IVaultConfig)
        state_repo = self.create_state_repository()
        note_parser = self.create_note_parser()

        return NoteDiscoveryService(
            vault_path=vault_config.vault_path,
            source_dir=str(vault_config.source_dir),
            state_repository=state_repo,
            note_parser=note_parser,
        )

    def create_all_components(self) -> dict:
        """Create all components needed by the sync engine.

        Returns:
            Dictionary mapping component names to instances
        """
        logger.info("creating_sync_engine_components")

        components = {
            "cache_manager": self.create_cache_manager(),
            "anki_client": self.create_anki_client(),
            "state_repository": self.create_state_repository(),
            "llm_provider": self.create_llm_provider(),
            "card_generator": self.create_card_generator(),
            "note_parser": self.create_note_parser(),
            "note_discovery_service": self.create_note_discovery_service(),
        }

        # Log component creation
        for name, component in components.items():
            logger.debug("component_created", name=name, type=type(component).__name__)

        logger.info("all_components_created", component_count=len(components))

        return components

    def validate_components(self, components: dict) -> list[str]:
        """Validate that all required components are present and functional.

        Args:
            components: Dictionary of created components

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        required_components = [
            "cache_manager",
            "anki_client",
            "state_repository",
            "card_generator",
            "note_parser",
            "note_discovery_service",
        ]

        # Check required components are present
        for component_name in required_components:
            if component_name not in components:
                errors.append(f"Missing required component: {component_name}")
                continue

            component = components[component_name]

            # Validate component functionality
            try:
                if component_name == "anki_client":
                    if not component.check_connection():
                        errors.append("Anki client connection check failed")
                elif component_name == "llm_provider":
                    if not component.check_connection():
                        errors.append("LLM provider connection check failed")
                elif component_name == "state_repository":
                    # Basic validation - try to get stats
                    component.get_sync_stats()
                elif component_name == "cache_manager":
                    # Check cache directory exists
                    cache_info = component.get_cache_size_info()
                    if not cache_info["cache_dir_exists"]:
                        errors.append("Cache directory does not exist")

            except Exception as e:
                errors.append(f"Component validation failed for {component_name}: {e}")

        return errors

    def cleanup_components(self, components: dict) -> None:
        """Clean up components that need explicit cleanup.

        Args:
            components: Dictionary of components to clean up
        """
        logger.info("cleaning_up_components")

        # Close cache manager
        if "cache_manager" in components:
            try:
                components["cache_manager"].close_caches()
                logger.debug("cache_manager_closed")
            except Exception as e:
                logger.warning("error_closing_cache_manager", error=str(e))

        # Close Anki client
        if "anki_client" in components:
            try:
                components["anki_client"].close()
                logger.debug("anki_client_closed")
            except Exception as e:
                logger.warning("error_closing_anki_client", error=str(e))

        logger.info("component_cleanup_completed")
