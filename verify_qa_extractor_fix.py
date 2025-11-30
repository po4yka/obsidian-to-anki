import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.sync.engine import SyncEngine


def verify_fix():
    print("Verifying QA Extractor Fix...")

    # Mock dependencies
    config = MagicMock(spec=Config)
    config.vault_path = Path("/tmp/vault")
    config.problematic_notes_dir = Path("/tmp/problems")
    config.enable_problematic_notes_archival = False
    config.use_langgraph = True
    config.use_pydantic_ai = False
    config.llm_provider = "ollama"
    config.ollama_base_url = "http://localhost:11434"
    config.get_model_for_agent.return_value = "qwen3:8b"
    config.get_model_config_for_task.return_value = {}
    config.db_path = Path("/tmp/db.sqlite")
    config.model_names = {}
    config.enable_memory_cleanup = False

    state_db = MagicMock()
    anki_client = MagicMock()

    # Mock ProviderFactory to avoid actual network calls or complex init
    # We want to verify that ProviderFactory.create_from_config is called
    # and the result is passed to QAExtractorAgent

    with SyncEngine(config, state_db, anki_client) as engine:
        if not hasattr(engine, "qa_extractor"):
            print("FAIL: qa_extractor not initialized")
            return

        extractor = engine.qa_extractor
        if extractor is None:
            print("FAIL: qa_extractor is None")
            return

        provider = extractor.llm_provider
        print(f"Provider type: {type(provider)}")

        # Check if provider has generate_json method
        if not hasattr(provider, "generate_json"):
            print("FAIL: Provider missing generate_json method")
            return

        # Check if generate_json is a coroutine function (it should NOT be, usually)
        # But wait, BaseLLMProvider.generate_json is a regular method that calls generate()
        # The issue was that the dummy provider's generate() was async, but BaseLLMProvider.generate_json
        # expects generate() to be sync (or handled correctly).

        # Actually, let's check if the provider is NOT the dummy one from LangGraphOrchestrator
        if "LangGraphCompatibilityProvider" in str(type(provider)):
            print("FAIL: Still using LangGraphCompatibilityProvider")
            return

        print("SUCCESS: QA Extractor initialized with a real provider")


if __name__ == "__main__":
    try:
        verify_fix()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
