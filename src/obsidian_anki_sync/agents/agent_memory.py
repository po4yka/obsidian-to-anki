"""Agentic memory system using LangGraph Memory with ChromaDB backend.

Provides persistent memory storage for agent learning, pattern recognition,
and adaptive routing based on historical performance.
"""

import json
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from ..utils.logging import get_logger
from .specialized import ProblemDomain

logger = get_logger(__name__)


class OpenAIEmbeddings:
    """Simple OpenAI embeddings wrapper."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings.

        Args:
            model: Embedding model name

        Raises:
            ValueError: If OpenAI API key is not configured
        """
        try:
            self.client = OpenAI()
            self.model = model
        except Exception as e:
            raise ValueError(
                f"Failed to initialize OpenAI client: {e}. "
                "Set OPENAI_API_KEY environment variable or disable semantic search."
            ) from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class AgentMemoryStore:
    """Persistent memory store for agent learning using ChromaDB."""

    def __init__(
        self,
        storage_path: Path,
        embedding_model: str | None = None,
        enable_semantic_search: bool = True,
    ):
        """Initialize agent memory store.

        Args:
            storage_path: Path to store ChromaDB data
            embedding_model: Embedding model name (default: text-embedding-3-small)
            enable_semantic_search: Enable semantic search capabilities
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize embeddings
        self.enable_semantic_search = enable_semantic_search
        if enable_semantic_search:
            model = embedding_model or "text-embedding-3-small"
            try:
                self.embeddings = OpenAIEmbeddings(model=model)
            except Exception as e:
                logger.warning(
                    "openai_embeddings_unavailable",
                    error=str(e),
                    fallback="Using simple text matching",
                )
                self.embeddings = None
                self.enable_semantic_search = False
        else:
            self.embeddings = None

        # Initialize collections
        self._initialize_collections()

    def _initialize_collections(self) -> None:
        """Initialize ChromaDB collections for different memory types."""
        self.collections = {}

        memory_types = [
            "failure_patterns",
            "success_patterns",
            "agent_performance",
            "routing_decisions",
        ]

        for memory_type in memory_types:
            try:
                collection = self.client.get_or_create_collection(
                    name=memory_type,
                    metadata={"description": f"Memory store for {memory_type}"},
                )
                self.collections[memory_type] = collection
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
        if self.enable_semantic_search and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        # Store in ChromaDB
        memory_id = f"failure_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
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
        if self.enable_semantic_search and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        # Store in ChromaDB
        memory_id = f"success_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
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
            if self.enable_semantic_search and self.embeddings:
                # Semantic search using embeddings
                query_embedding = self.embeddings.embed_query(query_text)
                results = collection.query(
                    query_embeddings=[query_embedding],
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
            ids = results.get("ids", [[]])
            metadatas = results.get("metadatas", [[]])
            documents = results.get("documents", [[]])
            distances = results.get("distances", [[]])

            # Validate all arrays have consistent structure
            if ids and len(ids[0]) > 0:
                num_results = len(ids[0])
                for i in range(num_results):
                    # Safe access with bounds checking
                    memory_id = ids[0][i] if i < len(ids[0]) else None
                    metadata = metadatas[0][i] if metadatas and len(
                        metadatas[0]) > i else {}
                    document = documents[0][i] if documents and len(
                        documents[0]) > i else ""
                    distance = distances[0][i] if distances and len(
                        distances[0]) > i else None

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
            if self.enable_semantic_search and self.embeddings:
                # Semantic search using embeddings
                query_embedding = self.embeddings.embed_query(query_text)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,  # Get best match
                )
            else:
                # Fallback to text search
                results = collection.query(
                    query_texts=[query_text],
                    n_results=1,
                )

            # Extract recommendation with safe access
            ids = results.get("ids", [[]])
            metadatas = results.get("metadatas", [[]])
            if ids and len(ids[0]) > 0 and metadatas and len(metadatas[0]) > 0:
                metadata = metadatas[0][0]
                successful_agent_str = metadata.get(
                    "successful_agent") if metadata else None

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
        metadata: dict[str, Any | None] = None,
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
        if self.enable_semantic_search and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        memory_id = f"perf_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[memory_metadata],
            embeddings=[embedding] if embedding else None,
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
        if self.enable_semantic_search and self.embeddings:
            try:
                embedding = self.embeddings.embed_query(content)
            except Exception as e:
                logger.warning("embedding_generation_failed", error=str(e))

        memory_id = f"routing_{int(time.time() * 1000)}"
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
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
                if all_memories["ids"]:
                    for i, metadata in enumerate(all_memories["metadatas"]):
                        timestamp = float(metadata.get("timestamp", "0"))
                        if timestamp < cutoff_time:
                            old_ids.append(all_memories["ids"][i])

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
