"""Dependency injection container for the application."""

from typing import Any, TypeVar

from ..domain.interfaces.anki_client import IAnkiClient
from ..domain.interfaces.anki_config import IAnkiConfig
from ..domain.interfaces.card_generator import ICardGenerator
from ..domain.interfaces.llm_config import ILLMConfig
from ..domain.interfaces.llm_provider import ILLMProvider
from ..domain.interfaces.note_parser import INoteParser
from ..domain.interfaces.state_repository import IStateRepository
from ..domain.interfaces.vault_config import IVaultConfig
from ..utils.logging import get_logger

T = TypeVar('T')

logger = get_logger(__name__)


class DependencyContainer:
    """Simple dependency injection container.

    This container manages the registration and resolution of dependencies
    throughout the application, enabling loose coupling and testability.
    """

    def __init__(self):
        """Initialize the dependency container."""
        self._services: dict[type, Any] = {}
        self._factories: dict[type, callable] = {}
        self._singletons: dict[type, Any] = {}

        logger.debug("dependency_container_initialized")

    def register(self, interface: type[T], implementation: Any) -> None:
        """Register a service implementation.

        Args:
            interface: The interface/abstract class
            implementation: The concrete implementation
        """
        self._services[interface] = implementation
        logger.debug(
            f"service_registered: {interface.__name__} -> {type(implementation).__name__}")

    def register_factory(self, interface: type[T], factory: callable) -> None:
        """Register a factory function for creating services.

        Args:
            interface: The interface/abstract class
            factory: Factory function that returns the implementation
        """
        self._factories[interface] = factory
        logger.debug(f"factory_registered: {interface.__name__}")

    def register_singleton(self, interface: type[T], implementation: Any) -> None:
        """Register a singleton service.

        Args:
            interface: The interface/abstract class
            implementation: The singleton implementation
        """
        self._singletons[interface] = implementation
        logger.debug(f"singleton_registered: {interface.__name__}")

    def resolve(self, interface: type[T]) -> T:
        """Resolve a service implementation.

        Args:
            interface: The interface to resolve

        Returns:
            The implementation instance

        Raises:
            ValueError: If the interface is not registered
        """
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]

        # Check registered services
        if interface in self._services:
            return self._services[interface]

        # Check factories
        if interface in self._factories:
            implementation = self._factories[interface]()
            # Cache as singleton after first creation
            self._singletons[interface] = implementation
            return implementation

        raise ValueError(
            f"No implementation registered for {interface.__name__}")

    def has_registration(self, interface: type[T]) -> bool:
        """Check if an interface has a registered implementation.

        Args:
            interface: The interface to check

        Returns:
            True if registered, False otherwise
        """
        return (
            interface in self._services or
            interface in self._factories or
            interface in self._singletons
        )

    def clear(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        logger.debug("container_cleared")

    def get_registered_interfaces(self) -> list[type]:
        """Get all registered interface types.

        Returns:
            List of registered interface types
        """
        return list(set(self._services.keys()) | set(self._factories.keys()) | set(self._singletons.keys()))


# Global container instance
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global dependency container instance.

    Returns:
        The global container instance
    """
    global _container
    if _container is None:
        _container = DependencyContainer()
        setup_container(_container)
    return _container


def setup_container(container: DependencyContainer) -> None:
    """Setup the container with default service registrations.

    Args:
        container: The container to setup
    """
    # Import here to avoid circular imports
    from ..config import Config
    from ..providers.factory import ProviderFactory
    from ..anki.client import AnkiClient
    from ..sync.state_db import StateDB
    from ..apf.generator import APFGenerator
    from ..obsidian.parser import create_note_parser

    # Load configuration
    config = Config()

    # Register configuration interfaces
    container.register(IVaultConfig, config)
    container.register(ILLMConfig, config)
    container.register(IAnkiConfig, config)

    # Register service implementations
    container.register_factory(IAnkiClient, lambda: AnkiClient(
        url=config.anki_connect_url,
        timeout=config.llm_timeout,
        enable_health_checks=True,
    ))

    container.register_factory(
        IStateRepository, lambda: StateDB(config.db_path))

    container.register_factory(
        ILLMProvider, lambda: ProviderFactory.create_from_config(config))

    container.register_factory(ICardGenerator, lambda: APFGenerator(config))

    container.register_factory(INoteParser, lambda: create_note_parser())

    logger.info("container_setup_completed")


def inject[T](interface: type[T]) -> T:
    """Dependency injection decorator/helper.

    Args:
        interface: The interface to inject

    Returns:
        The resolved implementation
    """
    return get_container().resolve(interface)
