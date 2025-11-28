from unittest.mock import MagicMock, patch

import pytest

from obsidian_anki_sync.agents.langgraph.model_factory import ModelFactory
from obsidian_anki_sync.agents.langgraph.workflow_builder import WorkflowBuilder
from obsidian_anki_sync.config import Config


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.get_model_for_agent.return_value = "gpt-4o"
    config.get_model_config_for_task.return_value = {"max_tokens": 100}
    config.langgraph_max_retries = 3
    config.langgraph_auto_fix = True
    config.langgraph_strict_mode = True
    config.enable_card_splitting = True
    config.enable_context_enrichment = True
    config.enable_memorization_quality = True
    config.enable_duplicate_detection = False
    return config


class TestModelFactory:
    def test_get_model_creates_new_model(self, mock_config):
        factory = ModelFactory(mock_config)
        with patch(
            "obsidian_anki_sync.agents.langgraph.model_factory.PydanticAIModelFactory"
        ) as mock_factory:
            mock_model = MagicMock()
            mock_factory.create_from_config.return_value = mock_model

            model = factory.get_model("generator")

            assert model == mock_model
            mock_factory.create_from_config.assert_called_once()
            mock_config.get_model_for_agent.assert_called_with("generator")

    def test_get_model_returns_cached_model(self, mock_config):
        factory = ModelFactory(mock_config)
        with patch(
            "obsidian_anki_sync.agents.langgraph.model_factory.PydanticAIModelFactory"
        ) as mock_factory:
            mock_model = MagicMock()
            mock_factory.create_from_config.return_value = mock_model

            model1 = factory.get_model("generator")
            model2 = factory.get_model("generator")

            assert model1 == model2
            mock_factory.create_from_config.assert_called_once()

    def test_clear_cache(self, mock_config):
        factory = ModelFactory(mock_config)
        with patch(
            "obsidian_anki_sync.agents.langgraph.model_factory.PydanticAIModelFactory"
        ) as mock_factory:
            mock_model = MagicMock()
            mock_factory.create_from_config.return_value = mock_model

            factory.get_model("generator")
            factory.clear_cache()
            factory.get_model("generator")

            assert mock_factory.create_from_config.call_count == 2


class TestWorkflowBuilder:
    def test_build_workflow(self, mock_config):
        builder = WorkflowBuilder(mock_config)
        workflow = builder.build_workflow()

        assert workflow is not None
        # Basic check that workflow has nodes
        # Accessing internal graph structure might be implementation dependent
        # But we can check if compile works
        app = workflow.compile()
        assert app is not None
