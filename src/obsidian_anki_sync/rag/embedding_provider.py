"""Embedding provider for RAG system.

Supports multiple embedding backends:
- OpenRouter (via OpenAI-compatible API)
- OpenAI directly
- Local models via Ollama

Uses LangChain embeddings for compatibility with vector stores.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    """Provides embeddings for RAG system using configured LLM provider.

    Supports:
    - OpenRouter: Uses OpenAI-compatible embeddings API
    - OpenAI: Uses native OpenAI embeddings
    - Ollama: Uses local Ollama embeddings

    Implements caching to avoid re-embedding unchanged content.
    """

    # Default embedding models per provider
    DEFAULT_MODELS = {
        "openrouter": "openai/text-embedding-3-small",
        "openai": "text-embedding-3-small",
        "ollama": "nomic-embed-text",
    }

    # Embedding dimensions per model (for validation)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "nomic-embed-text": 768,
    }

    def __init__(
        self,
        config: Config,
        model_name: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize embedding provider.

        Args:
            config: Application configuration
            model_name: Override embedding model name (uses config default if None)
            cache_dir: Directory for embedding cache (uses memory cache if None)
        """
        self.config = config
        self.provider = config.llm_provider.lower()
        self.model_name = model_name or self._get_default_model()
        self.cache_dir = cache_dir
        self._embeddings: Embeddings | None = None
        self._cache: dict[str, list[float]] = {}

        logger.info(
            "embedding_provider_initialized",
            provider=self.provider,
            model=self.model_name,
            cache_enabled=cache_dir is not None,
        )

    def _get_default_model(self) -> str:
        """Get default embedding model for configured provider."""
        # Use config embedding_model if set, otherwise provider default
        if self.config.embedding_model:
            return self.config.embedding_model
        return self.DEFAULT_MODELS.get(self.provider, "text-embedding-3-small")

    def _create_embeddings(self) -> Embeddings:
        """Create LangChain embeddings instance for configured provider.

        Returns:
            LangChain Embeddings instance

        Raises:
            ValueError: If provider is not supported for embeddings
        """
        if self.provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            return OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base=self.config.openrouter_base_url,
                timeout=self.config.llm_timeout,
            )

        elif self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.config.openai_api_key,
                openai_api_base=self.config.openai_base_url,
                timeout=self.config.llm_timeout,
            )

        elif self.provider == "ollama":
            # Use OllamaEmbeddings from langchain-community
            try:
                from langchain_community.embeddings import OllamaEmbeddings

                return OllamaEmbeddings(
                    model=self.model_name,
                    base_url=self.config.ollama_base_url,
                )
            except ImportError:
                msg = (
                    "Ollama embeddings require langchain-community. "
                    "Install with: pip install langchain-community"
                )
                raise ValueError(msg)

        else:
            msg = (
                f"Provider '{self.provider}' does not support embeddings. "
                f"Supported providers: openrouter, openai, ollama"
            )
            raise ValueError(msg)

    @property
    def embeddings(self) -> Embeddings:
        """Get or create LangChain embeddings instance."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _compute_cache_key(self, text: str) -> str:
        """Compute cache key for text."""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embedding if available

        Returns:
            Embedding vector as list of floats
        """
        if use_cache:
            cache_key = self._compute_cache_key(text)
            if cache_key in self._cache:
                logger.debug("embedding_cache_hit", key=cache_key[:16])
                return self._cache[cache_key]

        try:
            embedding = self.embeddings.embed_query(text)

            if use_cache:
                self._cache[cache_key] = embedding
                logger.debug(
                    "embedding_cached",
                    key=cache_key[:16],
                    dimension=len(embedding),
                )

            return list(embedding)  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(
                "embedding_failed",
                error=str(e),
                text_length=len(text),
            )
            raise

    def embed_texts(
        self,
        texts: list[str],
        use_cache: bool = True,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._compute_cache_key(text)
                if cache_key in self._cache:
                    results.append(self._cache[cache_key])
                else:
                    results.append([])  # Placeholder
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
            results = [[] for _ in texts]

        if not uncached_texts:
            logger.info("all_embeddings_cached", count=len(texts))
            return results

        # Embed uncached texts in batches
        logger.info(
            "embedding_texts",
            total=len(texts),
            uncached=len(uncached_texts),
            batch_size=batch_size,
        )

        try:
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]

                batch_embeddings = self.embeddings.embed_documents(batch)

                # Store results and cache
                for j, embedding in enumerate(batch_embeddings):
                    idx = uncached_indices[batch_start + j]
                    results[idx] = embedding

                    if use_cache:
                        cache_key = self._compute_cache_key(batch[j])
                        self._cache[cache_key] = embedding

                logger.debug(
                    "batch_embedded",
                    batch_start=batch_start,
                    batch_end=batch_end,
                )

            return results

        except Exception as e:
            logger.error(
                "batch_embedding_failed",
                error=str(e),
                uncached_count=len(uncached_texts),
            )
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this provider.

        Returns:
            Embedding dimension (e.g., 1536 for text-embedding-3-small)
        """
        # Extract model name without provider prefix
        model_short = self.model_name.split("/")[-1]

        if model_short in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[model_short]

        # Generate test embedding to determine dimension
        logger.info("detecting_embedding_dimension", model=self.model_name)
        test_embedding = self.embed_text("test", use_cache=False)
        dimension = len(test_embedding)

        logger.info(
            "embedding_dimension_detected",
            model=self.model_name,
            dimension=dimension,
        )

        return dimension

    def clear_cache(self) -> int:
        """Clear the embedding cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info("embedding_cache_cleared", entries=count)
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "entries": len(self._cache),
            "model": self.model_name,
            "provider": self.provider,
        }


@lru_cache(maxsize=4)
def get_embedding_provider(
    provider: str,
    model_name: str | None = None,
) -> EmbeddingProvider:
    """Get or create a cached embedding provider instance.

    Args:
        provider: Provider name (openrouter, openai, ollama)
        model_name: Optional model name override

    Returns:
        EmbeddingProvider instance
    """
    from obsidian_anki_sync.config import get_config

    config = get_config()
    return EmbeddingProvider(config, model_name)
