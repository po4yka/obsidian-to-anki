"""Agentic memory system using LangGraph Memory with ChromaDB backend.

Provides persistent memory storage for agent learning, pattern recognition,
and adaptive routing based on historical performance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import chromadb
import numpy as np
from chromadb.config import Settings

from obsidian_anki_sync.utils.logging import get_logger

from .specialized import ProblemDomain

if TYPE_CHECKING:
    from obsidian_anki_sync.config import Config
    from obsidian_anki_sync.rag.embedding_provider import EmbeddingProvider

logger = get_logger(__name__)


class DummyEmbeddingFunction:
    """Minimal embedding function to avoid external downloads during tests."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[0.0] for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed query texts - same as __call__ for dummy function."""
        return self(input)

    def name(self) -> str:  # pragma: no cover - simple identifier
        return "default"


class RAGEmbeddingFunction:
    """RAG embedding function compatible with ChromaDB."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedding_provider.embed_texts(input, use_cache=True)

    def name(self) -> str:
        return f"rag-{self.embedding_provider.model_name}"


class AgentMemoryStore:
    """Persistent memory store for agent learning using ChromaDB.

    Uses lazy initialization to avoid opening file handles until actually needed.
    """

    def __init__(
        self,
        storage_path: Path,
        config: Config | None = None,
        enable_semantic_search: bool = True,
    ):
        """Initialize agent memory store.

        Args:
            storage_path: Path to store ChromaDB data
            config: Application configuration for embedding provider
            enable_semantic_search: Enable semantic search capabilities
        """
        self.storage_path = Path(storage_path)
        self.config = config
        self._enable_semantic_search = enable_semantic_search

        # Lazy-initialized resources (to avoid file descriptor exhaustion)
        self._client: chromadb.api.client.ClientAPI | None = None
        self._collections: dict[str, Any] | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._embedding_function: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize ChromaDB and collections when first needed."""
        if self._initialized:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize embeddings using RAG embedding provider
        if self._enable_semantic_search and self.config is not None:
            try:
                from obsidian_anki_sync.rag.embedding_provider import EmbeddingProvider

                self._embedding_provider = EmbeddingProvider(self.config)
                self._embedding_function = RAGEmbeddingFunction(
                    self._embedding_provider
                )
                logger.info(
                    "agent_memory_embeddings_initialized",
                    provider=self.config.llm_provider,
                    model=self._embedding_provider.model_name,
                )
            except Exception as e:
                logger.debug(
                    "agent_memory_embeddings_unavailable",
                    error=str(e),
                    fallback="Using simple text matching",
                )
                self._embedding_provider = None
                self._enable_semantic_search = False
        elif self._enable_semantic_search:
            logger.debug(
                "agent_memory_semantic_search_disabled",
                reason="No config provided",
            )
            self._enable_semantic_search = False

        if not self._embedding_function:
            # Prevent ChromaDB from downloading default embedding models during tests
            self._embedding_function = DummyEmbeddingFunction()

        # Initialize collections
        self._initialize_collections()
        self._initialized = True

    @property
    def client(self) -> chromadb.api.client.ClientAPI:
        """Get ChromaDB client (lazy initialization)."""
        self._ensure_initialized()
        return self._client  # type: ignore[return-value]

    @property
    def collections(self) -> dict[str, Any]:
        """Get collections (lazy initialization)."""
        self._ensure_initialized()
        return self._collections  # type: ignore[return-value]

    @property
    def enable_semantic_search(self) -> bool:
        """Check if semantic search is enabled."""
        return self._enable_semantic_search

    @property
    def embedding_function(self) -> Any:
        """Get embedding function (lazy initialization)."""
        self._ensure_initialized()
        return self._embedding_function

    def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections for different memory types."""
        self._collections = {}

        memory_types = [
            "failure_patterns",
            "success_patterns",
            "agent_performance",
            "routing_decisions",
        ]

        for memory_type in memory_types:
            try:
                if self._client:
                    collection = self._client.get_or_create_collection(
                        name=memory_type,
                        metadata={"description": f"Memory store for {memory_type}"},
                        embedding_function=self._embedding_function,
                    )
                    self._collections[memory_type] = collection
            except Exception as e:
                logger.error(
                    "failed_to_create_collection",
                    collection=memory_type,
                    error=str(e),
                )
                raise

    def store_failure_pattern(
        self,
        error_context: dict[str, Any],
        attempted_agents: list[ProblemDomain],
    ) -> str:
        """Store failure pattern with embedding.

        Args:
            error_context: Context about the error
            attempted_agents: List of agents that were tried

        Returns:
            Memory ID
        """
        collection = self.collections["failure_patterns"]

        # Create content string for embedding
        error_msg = error_context.get("error_message", "")
        error_type = error_context.get("error_type", "unknown")
        content = f"{error_type}: {error_msg}"

        # Create metadata
        metadata = {
            "error_type": error_type,
            "attempted_agents": json.dumps([a.value for a in attempted_agents]),
            "timestamp": str(time.time()),
            "file_path": error_context.get("file_path", ""),
            "processing_stage": error_context.get("processing_stage", ""),
        }

        # Generate embedding if enabled
        embedding = None
        if self.enable_semantic_search and self._embedding_provider:
            try:
                embedding = self._embedding_provider.embed_text(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        # Store in ChromaDB
        memory_id = f"failure_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[np.array(embedding)] if embedding else None,
        )

        logger.debug(
            "failure_pattern_stored",
            memory_id=memory_id,
            error_type=error_type,
            attempted_agents=[a.value for a in attempted_agents],
        )

        return memory_id

    def store_success_pattern(
        self,
        error_context: dict[str, Any],
        successful_agent: ProblemDomain,
    ) -> str:
        """Store success pattern with embedding.

        Args:
            error_context: Context about the error
            successful_agent: Agent that successfully repaired

        Returns:
            Memory ID
        """
        collection = self.collections["success_patterns"]

        # Create content string for embedding
        error_msg = error_context.get("error_message", "")
        error_type = error_context.get("error_type", "unknown")
        content = f"{error_type}: {error_msg}"

        # Create metadata
        metadata = {
            "error_type": error_type,
            "successful_agent": successful_agent.value,
            "timestamp": str(time.time()),
            "file_path": error_context.get("file_path", ""),
            "processing_stage": error_context.get("processing_stage", ""),
        }

        # Generate embedding if enabled
        embedding = None
        if self.enable_semantic_search and self._embedding_provider:
            try:
                embedding = self._embedding_provider.embed_text(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        # Store in ChromaDB
        memory_id = f"success_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[np.array(embedding)] if embedding else None,
        )

        logger.debug(
            "success_pattern_stored",
            memory_id=memory_id,
            error_type=error_type,
            successful_agent=successful_agent.value,
        )

        return memory_id

    def find_similar_failures(
        self,
        error_context: dict[str, Any],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find similar past failures using semantic search.

        Args:
            error_context: Current error context
            limit: Maximum number of results

        Returns:
            List of similar failure patterns
        """
        collection = self.collections["failure_patterns"]

        # Create query text
        error_msg = error_context.get("error_message", "")
        error_type = error_context.get("error_type", "unknown")
        query_text = f"{error_type}: {error_msg}"

        try:
            if self.enable_semantic_search and self._embedding_provider:
                # Semantic search using embeddings
                query_embedding = self._embedding_provider.embed_text(query_text)
                results = collection.query(
                    query_embeddings=[np.array(query_embedding)],
                    n_results=limit,
                )
            else:
                # Fallback to text search
                results = collection.query(
                    query_texts=[query_text],
                    n_results=limit,
                )

            # Format results
            similar_failures = []
            ids: list[list[str]] = cast("list[list[str]]", results.get("ids", [[]]))
            metadatas: list[list[dict[str, Any]]] = cast(
                "list[list[dict[str, Any]]]", results.get("metadatas", [[]])
            )
            documents: list[list[str]] = cast(
                "list[list[str]]", results.get("documents", [[]])
            )
            distances: list[list[float]] = cast(
                "list[list[float]]", results.get("distances", [[]])
            )

            # Validate all arrays have consistent structure
            if ids and len(ids[0]) > 0:
                num_results = len(ids[0])
                for i in range(num_results):
                    # Safe access with bounds checking
                    memory_id = ids[0][i] if i < len(ids[0]) else None
                    metadata = (
                        metadatas[0][i] if metadatas and len(metadatas[0]) > i else {}
                    )
                    document = (
                        documents[0][i] if documents and len(documents[0]) > i else ""
                    )
                    distance = (
                        distances[0][i] if distances and len(distances[0]) > i else None
                    )

                    if memory_id is None:
                        continue

                    similar_failures.append(
                        {
                            "memory_id": memory_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity": 1.0 - distance if distance else None,
                        }
                    )

            logger.debug(
                "similar_failures_found",
                count=len(similar_failures),
                query=query_text[:100],
            )

            return similar_failures

        except Exception as e:
            logger.error("semantic_search_failed", error=str(e))
            return []

    def get_agent_recommendation(
        self,
        error_context: dict[str, Any],
    ) -> ProblemDomain | None:
        """Get agent recommendation based on similar successful patterns.

        Args:
            error_context: Current error context

        Returns:
            Recommended agent or None
        """
        collection = self.collections["success_patterns"]

        # Create query text
        error_msg = error_context.get("error_message", "")
        error_type = error_context.get("error_type", "unknown")
        query_text = f"{error_type}: {error_msg}"

        try:
            if self.enable_semantic_search and self._embedding_provider:
                # Semantic search using embeddings
                query_embedding = self._embedding_provider.embed_text(query_text)
                results = collection.query(
                    query_embeddings=[np.array(query_embedding)],
                    n_results=1,  # Get best match
                )
            else:
                # Fallback to text search
                results = collection.query(
                    query_texts=[query_text],
                    n_results=1,
                )

            # Extract recommendation with safe access
            ids: list[list[str]] = cast("list[list[str]]", results.get("ids", [[]]))
            metadatas: list[list[dict[str, Any]]] = cast(
                "list[list[dict[str, Any]]]", results.get("metadatas", [[]])
            )
            if ids and len(ids[0]) > 0 and metadatas and len(metadatas[0]) > 0:
                metadata = metadatas[0][0]
                successful_agent_str = (
                    metadata.get("successful_agent") if metadata else None
                )

                if successful_agent_str:
                    try:
                        agent = ProblemDomain(successful_agent_str)
                        logger.info(
                            "agent_recommendation_from_memory",
                            agent=successful_agent_str,
                            error_type=error_type,
                        )
                        return agent
                    except ValueError:
                        logger.warning(
                            "invalid_agent_in_memory",
                            agent=successful_agent_str,
                        )

            return None

        except Exception as e:
            logger.error("recommendation_search_failed", error=str(e))
            return None

    def store_performance_metric(
        self,
        agent_name: str,
        metric_name: str,
        value: float,
        metadata: dict[str, Any | None] | None = None,
    ) -> str:
        """Store performance metric.

        Args:
            agent_name: Name of the agent
            metric_name: Name of the metric
            value: Metric value
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        collection = self.collections["agent_performance"]

        content = f"{agent_name}: {metric_name} = {value}"
        memory_metadata = {
            "agent_name": agent_name,
            "metric_name": metric_name,
            "value": str(value),
            "timestamp": str(time.time()),
            **(metadata or {}),
        }

        # Generate embedding if enabled
        embedding = None
        if self.enable_semantic_search and self._embedding_provider:
            try:
                embedding = self._embedding_provider.embed_text(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        memory_id = f"perf_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[memory_metadata],
            embeddings=[np.array(embedding)] if embedding else None,
        )

        return memory_id

    def store_routing_decision(
        self,
        error_context: dict[str, Any],
        selected_agent: ProblemDomain,
        success: bool,
        confidence: float,
    ) -> str:
        """Store routing decision and outcome.

        Args:
            error_context: Error context
            selected_agent: Agent that was selected
            success: Whether the routing was successful
            confidence: Confidence score

        Returns:
            Memory ID
        """
        collection = self.collections["routing_decisions"]

        error_msg = error_context.get("error_message", "")
        content = f"Routing decision: {selected_agent.value} for {error_msg}"

        metadata = {
            "selected_agent": selected_agent.value,
            "success": str(success),
            "confidence": str(confidence),
            "error_type": error_context.get("error_type", "unknown"),
            "timestamp": str(time.time()),
        }

        # Generate embedding if enabled
        embedding = None
        if self.enable_semantic_search and self._embedding_provider:
            try:
                embedding = self._embedding_provider.embed_text(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        memory_id = f"routing_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[np.array(embedding)] if embedding else None,
        )

        return memory_id

    def cleanup_old_memories(self, retention_days: int) -> int:
        """Clean up memories older than retention period.

        Args:
            retention_days: Number of days to retain memories

        Returns:
            Number of memories deleted
        """
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        deleted_count = 0

        for memory_type, collection in self.collections.items():
            try:
                # Get all memories
                all_memories = collection.get()

                # Filter old memories
                old_ids = []
                if all_memories["ids"] and all_memories.get("metadatas"):
                    metadatas = all_memories["metadatas"]
                    if metadatas:
                        for i, metadata in enumerate(metadatas):
                            if metadata:
                                timestamp_raw = metadata.get("timestamp", "0")
                                try:
                                    # Handle various possible types safely
                                    if isinstance(timestamp_raw, int | float):
                                        timestamp = float(timestamp_raw)
                                    elif isinstance(timestamp_raw, str):
                                        timestamp = (
                                            float(timestamp_raw)
                                            if timestamp_raw
                                            else 0.0
                                        )
                                    else:
                                        # Skip non-numeric types
                                        continue
                                    if timestamp < cutoff_time:
                                        old_ids.append(all_memories["ids"][i])
                                except (ValueError, TypeError):
                                    # Skip invalid timestamp values
                                    continue

                # Delete old memories
                if old_ids:
                    collection.delete(ids=old_ids)
                    deleted_count += len(old_ids)
                    logger.info(
                        "old_memories_cleaned",
                        memory_type=memory_type,
                        deleted=len(old_ids),
                    )

            except Exception as e:
                logger.warning(
                    "cleanup_failed",
                    memory_type=memory_type,
                    error=str(e),
                )

        return deleted_count
