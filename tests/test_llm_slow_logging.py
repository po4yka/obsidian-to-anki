"""Tests for slow LLM request logging wrappers."""

import importlib.util
import logging
from pathlib import Path

from obsidian_anki_sync.config import Config


def _build_test_config(**overrides):
    defaults = {
        "llm_provider": "openai",
        "openrouter_api_key": "test-key",  # pragma: allowlist secret
        "openrouter_base_url": "http://example.com",
        "openrouter_site_url": "http://example.com",
        "openrouter_site_name": "example",
        "llm_timeout": 5.0,
        "llm_max_tokens": 1024,
        "llm_streaming_enabled": False,
        "llm_temperature": 0.1,
        "llm_reasoning_enabled": False,
        "llm_slow_request_threshold": 0.0,
        "generator_model": "stub-model",
        "default_llm_model": "stub-model",
        "embedding_model": "text-embedding-3-small",
    }
    defaults.update(overrides)
    return Config.model_construct(**defaults)


def test_embedding_provider_logs_slow_request(caplog):
    config = _build_test_config()

    embedding_path = (
        Path(__file__).parents[1]
        / "src"
        / "obsidian_anki_sync"
        / "rag"
        / "embedding_provider.py"
    )
    spec = importlib.util.spec_from_file_location(
        "obsidian_anki_sync.rag.embedding_provider", embedding_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider = module.EmbeddingProvider(config)

    class _StubEmbeddings:
        def embed_query(self, text: str):
            return [0.1, 0.2]

    provider._embeddings = _StubEmbeddings()

    with caplog.at_level(logging.WARNING):
        provider.embed_text("hello", use_cache=False)

    assert any(
        "llm_request_slow" in record.getMessage()
        and provider.model_name in record.getMessage()
        for record in caplog.records
    )
