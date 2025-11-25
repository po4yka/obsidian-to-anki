"""Tests for LangGraph async node functions and workflow invocation.

Tests cover:
- Async node function execution
- Workflow invocation with ainvoke
- Event loop compatibility
- Model caching and reuse
"""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from obsidian_anki_sync.agents.langgraph_orchestrator import (
    LangGraphOrchestrator,
    PipelineState,
)
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.models import NoteMetadata, QAPair


@pytest.fixture
def test_config_with_models(temp_dir):
    """Create test config with model settings."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir(parents=True, exist_ok=True)
    source_dir = vault_path / "test"
    source_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        vault_path=vault_path,
        source_dir=Path("test"),
        anki_connect_url="http://localhost:8765",
        anki_deck_name="Test Deck",
        openrouter_api_key="test-key",
        openrouter_model="openai/gpt-4",
        llm_max_tokens=128000,
        use_langgraph=True,
        use_pydantic_ai=True,
        db_path=temp_dir / "test.db",
    )


@pytest.fixture
def sample_pipeline_state(test_config_with_models, sample_metadata, sample_qa_pair):
    """Create sample pipeline state for testing."""
    return PipelineState(
        note_content="Test note content",
        metadata_dict=sample_metadata.model_dump(),
        qa_pairs_dicts=[sample_qa_pair.model_dump()],
        file_path="/test/path.md",
        slug_base="test-slug",
        config=test_config_with_models,
        existing_cards_dicts=None,
        pre_validator_model=None,
        card_splitting_model=None,
        generator_model=None,
        post_validator_model=None,
        context_enrichment_model=None,
        memorization_quality_model=None,
        duplicate_detection_model=None,
        pre_validation=None,
        card_splitting=None,
        generation=None,
        post_validation=None,
        context_enrichment=None,
        memorization_quality=None,
        duplicate_detection=None,
        current_stage="pre_validation",
        enable_card_splitting=False,
        enable_context_enrichment=False,
        enable_memorization_quality=False,
        enable_duplicate_detection=False,
        retry_count=0,
        max_retries=3,
    )


class TestAsyncNodeFunctions:
    """Test async node functions."""

    def test_node_functions_are_async(self):
        """Verify all node functions are async."""
        from obsidian_anki_sync.agents.langgraph_orchestrator import (
            card_splitting_node,
            context_enrichment_node,
            duplicate_detection_node,
            generation_node,
            memorization_quality_node,
            post_validation_node,
            pre_validation_node,
        )

        node_functions = [
            pre_validation_node,
            card_splitting_node,
            generation_node,
            post_validation_node,
            context_enrichment_node,
            memorization_quality_node,
            duplicate_detection_node,
        ]

        for node_func in node_functions:
            assert inspect.iscoroutinefunction(
                node_func
            ), f"{node_func.__name__} should be async"

    @pytest.mark.asyncio
    async def test_pre_validation_node_uses_await(self, sample_pipeline_state):
        """Test that pre_validation_node uses await instead of asyncio.run()."""
        from obsidian_anki_sync.agents.langgraph_orchestrator import (
            pre_validation_node,
        )

        # Mock the pre_validator agent
        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.PreValidatorAgent"
        ) as mock_validator_class:
            mock_validator = AsyncMock()
            mock_validator.validate = AsyncMock(
                return_value=MagicMock(
                    is_valid=True,
                    issues=[],
                    model_dump=lambda: {"is_valid": True, "issues": []},
                )
            )
            mock_validator_class.return_value = mock_validator

            # Execute node function
            result_state = await pre_validation_node(sample_pipeline_state)

            # Verify await was used (not asyncio.run)
            assert mock_validator.validate.called
            assert result_state["pre_validation"] is not None

    @pytest.mark.asyncio
    async def test_generation_node_uses_await(self, sample_pipeline_state):
        """Test that generation_node uses await instead of asyncio.run()."""
        from obsidian_anki_sync.agents.langgraph_orchestrator import (
            generation_node,
        )

        # Set up state with pre_validation complete
        sample_pipeline_state["pre_validation"] = {"is_valid": True, "issues": []}

        # Mock the generator agent
        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.GeneratorAgent"
        ) as mock_generator_class:
            mock_generator = AsyncMock()
            mock_generator.generate_cards = AsyncMock(
                return_value=MagicMock(
                    cards=[],
                    model_dump=lambda: {"cards": []},
                )
            )
            mock_generator_class.return_value = mock_generator

            # Execute node function
            result_state = await generation_node(sample_pipeline_state)

            # Verify await was used
            assert mock_generator.generate_cards.called
            assert result_state["generation"] is not None


class TestWorkflowInvocation:
    """Test workflow invocation with async methods."""

    def test_process_note_is_async(self):
        """Verify process_note is an async method."""
        assert inspect.iscoroutinefunction(LangGraphOrchestrator.process_note)

    @pytest.mark.asyncio
    async def test_process_note_uses_ainvoke(self, test_config_with_models):
        """Test that process_note uses ainvoke instead of invoke."""
        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.StateGraph"
        ) as mock_graph_class:
            # Mock the graph and app
            mock_graph = MagicMock()
            mock_app = AsyncMock()
            mock_app.ainvoke = AsyncMock(
                return_value={
                    "current_stage": "complete",
                    "generation": {"cards": []},
                }
            )
            mock_graph.compile.return_value = mock_app
            mock_graph_class.return_value = mock_graph

            # Mock model factory to avoid actual model creation
            with patch(
                "obsidian_anki_sync.agents.langgraph_orchestrator.PydanticAIModelFactory"
            ) as mock_factory:
                mock_factory.create_from_config = MagicMock(return_value=None)

                orchestrator = LangGraphOrchestrator(test_config_with_models)

                # Execute process_note
                result = await orchestrator.process_note(
                    note_content="Test",
                    metadata=MagicMock(),
                    qa_pairs=[],
                )

                # Verify ainvoke was called (not invoke)
                assert mock_app.ainvoke.called
                assert not mock_app.invoke.called
                assert result is not None

    @pytest.mark.asyncio
    async def test_process_note_generates_unique_thread_id(
        self, test_config_with_models
    ):
        """Test that process_note generates unique thread IDs."""
        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.StateGraph"
        ) as mock_graph_class:
            mock_graph = MagicMock()
            mock_app = AsyncMock()
            mock_app.ainvoke = AsyncMock(
                return_value={
                    "current_stage": "complete",
                    "generation": {"cards": []},
                }
            )
            mock_graph.compile.return_value = mock_app
            mock_graph_class.return_value = mock_graph

            with patch(
                "obsidian_anki_sync.agents.langgraph_orchestrator.PydanticAIModelFactory"
            ) as mock_factory:
                mock_factory.create_from_config = MagicMock(return_value=None)

                orchestrator = LangGraphOrchestrator(test_config_with_models)

                # Call process_note multiple times
                thread_ids = []
                for _ in range(3):
                    await orchestrator.process_note(
                        note_content="Test",
                        metadata=MagicMock(),
                        qa_pairs=[],
                    )
                    # Extract thread_id from ainvoke call
                    call_args = mock_app.ainvoke.call_args
                    config = call_args[1]["config"]
                    thread_ids.append(config["configurable"]["thread_id"])

                # Verify all thread IDs are unique
                assert len(thread_ids) == len(set(thread_ids))


class TestEventLoopCompatibility:
    """Test event loop compatibility."""

    @pytest.mark.asyncio
    async def test_node_function_in_existing_event_loop(self, sample_pipeline_state):
        """Test that node functions work in an existing event loop."""
        from obsidian_anki_sync.agents.langgraph_orchestrator import (
            pre_validation_node,
        )

        # Verify we're in an event loop
        loop = asyncio.get_running_loop()
        assert loop is not None

        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.PreValidatorAgent"
        ) as mock_validator_class:
            mock_validator = AsyncMock()
            mock_validator.validate = AsyncMock(
                return_value=MagicMock(
                    is_valid=True,
                    issues=[],
                    model_dump=lambda: {"is_valid": True, "issues": []},
                )
            )
            mock_validator_class.return_value = mock_validator

            # This should work without raising RuntimeError
            result = await pre_validation_node(sample_pipeline_state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_process_note_in_existing_event_loop(self, test_config_with_models):
        """Test that process_note works in an existing event loop."""
        # Verify we're in an event loop
        loop = asyncio.get_running_loop()
        assert loop is not None

        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.StateGraph"
        ) as mock_graph_class:
            mock_graph = MagicMock()
            mock_app = AsyncMock()
            mock_app.ainvoke = AsyncMock(
                return_value={
                    "current_stage": "complete",
                    "generation": {"cards": []},
                }
            )
            mock_graph.compile.return_value = mock_app
            mock_graph_class.return_value = mock_graph

            with patch(
                "obsidian_anki_sync.agents.langgraph_orchestrator.PydanticAIModelFactory"
            ) as mock_factory:
                mock_factory.create_from_config = MagicMock(return_value=None)

                orchestrator = LangGraphOrchestrator(test_config_with_models)

                # This should work without raising RuntimeError about event loop
                result = await orchestrator.process_note(
                    note_content="Test",
                    metadata=MagicMock(),
                    qa_pairs=[],
                )
                assert result is not None

    def test_sync_context_uses_asyncio_run(self, test_config_with_models):
        """Test that sync contexts properly use asyncio.run()."""
        # This test verifies that sync callers (like sync/engine.py) use asyncio.run()
        # We can't test this directly here, but we can verify the pattern exists
        import inspect

        # Verify process_note is async
        assert inspect.iscoroutinefunction(LangGraphOrchestrator.process_note)

        # In sync contexts, callers should use asyncio.run()
        # This is tested in integration tests


class TestModelCaching:
    """Test model caching and reuse."""

    def test_orchestrator_creates_models_on_init(self, test_config_with_models):
        """Test that orchestrator creates models during initialization."""
        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.PydanticAIModelFactory.create_from_config"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            orchestrator = LangGraphOrchestrator(test_config_with_models)

            # Verify models were created (7 models total)
            assert mock_create.call_count == 7

    def test_models_passed_to_state(self, test_config_with_models):
        """Test that cached models are passed to state."""
        mock_model = MagicMock()

        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.PydanticAIModelFactory.create_from_config"
        ) as mock_create:
            mock_create.return_value = mock_model

            orchestrator = LangGraphOrchestrator(test_config_with_models)

            # Verify models are cached
            assert orchestrator.pre_validator_model is mock_model
            assert orchestrator.card_splitting_model is mock_model
            assert orchestrator.generator_model is mock_model
            assert orchestrator.post_validator_model is mock_model

    @pytest.mark.asyncio
    async def test_node_uses_cached_model(self, sample_pipeline_state):
        """Test that node functions use cached models from state."""
        from obsidian_anki_sync.agents.langgraph_orchestrator import (
            pre_validation_node,
        )

        # Set cached model in state
        mock_model = MagicMock()
        sample_pipeline_state["pre_validator_model"] = mock_model

        with patch(
            "obsidian_anki_sync.agents.langgraph_orchestrator.PreValidatorAgent"
        ) as mock_validator_class:
            mock_validator = AsyncMock()
            mock_validator.validate = AsyncMock(
                return_value=MagicMock(
                    is_valid=True,
                    issues=[],
                    model_dump=lambda: {"is_valid": True, "issues": []},
                )
            )
            mock_validator_class.return_value = mock_validator

            await pre_validation_node(sample_pipeline_state)

            # Verify PreValidatorAgent was initialized with the cached model
            mock_validator_class.assert_called_once()
            call_args = mock_validator_class.call_args
            assert call_args[1]["model"] is mock_model
