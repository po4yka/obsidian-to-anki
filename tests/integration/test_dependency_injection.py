"""Integration tests for dependency injection container."""

import pytest

from obsidian_anki_sync.application.container import DependencyContainer
from obsidian_anki_sync.domain.interfaces.anki_client import IAnkiClient
from obsidian_anki_sync.domain.interfaces.card_generator import ICardGenerator
from obsidian_anki_sync.domain.interfaces.llm_provider import ILLMProvider
from obsidian_anki_sync.domain.interfaces.note_parser import INoteParser
from obsidian_anki_sync.domain.interfaces.state_repository import IStateRepository
from tests.fixtures import (
    MockAnkiClient,
    MockCardGenerator,
    MockLLMProvider,
    MockNoteParser,
    MockStateRepository,
)


class TestDependencyInjection:
    """Test the dependency injection container functionality."""

    def setup_method(self):
        """Set up test container."""
        self.container = DependencyContainer()

    def test_register_and_resolve_service(self):
        """Test registering and resolving a service."""
        mock_client = MockAnkiClient()

        # Register service
        self.container.register(IAnkiClient, mock_client)

        # Resolve service
        resolved = self.container.resolve(IAnkiClient)

        # Should return the same instance
        assert resolved is mock_client
        assert isinstance(resolved, MockAnkiClient)

    def test_register_factory(self):
        """Test registering and resolving a factory."""

        def create_client():
            return MockAnkiClient()

        # Register factory
        self.container.register_factory(IAnkiClient, create_client)

        # Resolve service
        resolved1 = self.container.resolve(IAnkiClient)
        resolved2 = self.container.resolve(IAnkiClient)

        # Should be different instances (factory creates new each time)
        assert isinstance(resolved1, MockAnkiClient)
        assert isinstance(resolved2, MockAnkiClient)
        assert resolved1 is not resolved2

    def test_register_singleton(self):
        """Test registering and resolving a singleton."""
        mock_client = MockAnkiClient()

        # Register singleton
        self.container.register_singleton(IAnkiClient, mock_client)

        # Resolve service multiple times
        resolved1 = self.container.resolve(IAnkiClient)
        resolved2 = self.container.resolve(IAnkiClient)

        # Should return the same instance
        assert resolved1 is mock_client
        assert resolved2 is mock_client
        assert resolved1 is resolved2

    def test_unregistered_interface_raises_error(self):
        """Test that resolving unregistered interface raises error."""
        with pytest.raises(ValueError, match="No implementation registered"):
            self.container.resolve(IAnkiClient)

    def test_has_registration(self):
        """Test checking if interface has registration."""
        # Initially not registered
        assert not self.container.has_registration(IAnkiClient)

        # Register service
        self.container.register(IAnkiClient, MockAnkiClient())
        assert self.container.has_registration(IAnkiClient)

        # Register factory
        self.container.register_factory(ICardGenerator, lambda: MockCardGenerator())
        assert self.container.has_registration(ICardGenerator)

        # Register singleton
        self.container.register_singleton(ILLMProvider, MockLLMProvider())
        assert self.container.has_registration(ILLMProvider)

    def test_get_registered_interfaces(self):
        """Test getting list of registered interfaces."""
        # Register some services
        self.container.register(IAnkiClient, MockAnkiClient())
        self.container.register_factory(ICardGenerator, lambda: MockCardGenerator())
        self.container.register_singleton(ILLMProvider, MockLLMProvider())

        registered = self.container.get_registered_interfaces()

        assert IAnkiClient in registered
        assert ICardGenerator in registered
        assert ILLMProvider in registered
        assert len(registered) == 3

    def test_clear_container(self):
        """Test clearing all registrations."""
        # Register services
        self.container.register(IAnkiClient, MockAnkiClient())
        self.container.register_factory(ICardGenerator, lambda: MockCardGenerator())

        assert self.container.has_registration(IAnkiClient)
        assert self.container.has_registration(ICardGenerator)

        # Clear container
        self.container.clear()

        assert not self.container.has_registration(IAnkiClient)
        assert not self.container.has_registration(ICardGenerator)
        assert len(self.container.get_registered_interfaces()) == 0

    def test_complex_dependency_resolution(self):
        """Test resolving complex dependencies."""
        # Register all mock services
        self.container.register(IAnkiClient, MockAnkiClient())
        self.container.register(IStateRepository, MockStateRepository())
        self.container.register(ILLMProvider, MockLLMProvider())
        self.container.register(ICardGenerator, MockCardGenerator())
        self.container.register(INoteParser, MockNoteParser())

        # Resolve all services
        anki_client = self.container.resolve(IAnkiClient)
        state_repo = self.container.resolve(IStateRepository)
        llm_provider = self.container.resolve(ILLMProvider)
        card_generator = self.container.resolve(ICardGenerator)
        note_parser = self.container.resolve(INoteParser)

        # Verify all are correct types
        assert isinstance(anki_client, MockAnkiClient)
        assert isinstance(state_repo, MockStateRepository)
        assert isinstance(llm_provider, MockLLMProvider)
        assert isinstance(card_generator, MockCardGenerator)
        assert isinstance(note_parser, MockNoteParser)

        # Verify functionality
        assert anki_client.check_connection()
        assert llm_provider.check_connection()
        assert len(card_generator.get_supported_languages()) > 0

    def test_service_replacement(self):
        """Test replacing a service implementation."""
        client1 = MockAnkiClient()
        client2 = MockAnkiClient()

        # Register first client
        self.container.register(IAnkiClient, client1)
        resolved1 = self.container.resolve(IAnkiClient)
        assert resolved1 is client1

        # Replace with second client
        self.container.register(IAnkiClient, client2)
        resolved2 = self.container.resolve(IAnkiClient)
        assert resolved2 is client2
        assert resolved2 is not client1
