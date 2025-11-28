"""Vector store for RAG system using ChromaDB.

Provides persistent vector storage with:
- Incremental indexing (only index changed files)
- Metadata filtering (by topic, difficulty, language)
- Similarity search with configurable thresholds
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from ..config import Config
from ..utils.logging import get_logger
from .document_chunker import DocumentChunk, DocumentChunker
from .embedding_provider import EmbeddingProvider

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result from vector search."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]
    source_file: str

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1)."""
        # ChromaDB returns L2 distance by default
        # Convert to similarity: 1 / (1 + distance)
        return 1 / (1 + self.score)


class VaultVectorStore:
    """ChromaDB-based vector store for Obsidian vault content.

    Features:
    - Persistent storage on disk
    - Incremental indexing
    - Metadata filtering
    - Batch operations for efficiency
    """

    COLLECTION_NAME = "obsidian_vault"

    def __init__(
        self,
        config: Config,
        persist_directory: Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize vector store.

        Args:
            config: Application configuration
            persist_directory: Directory for ChromaDB persistence
            embedding_provider: Custom embedding provider (creates default if None)
        """
        self.config = config

        # Set up persistence directory
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            vault_path = Path(config.vault_path)
            self.persist_directory = vault_path / ".chroma_db"

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding provider
        self.embedding_provider = embedding_provider or EmbeddingProvider(config)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            "vector_store_initialized",
            persist_directory=str(self.persist_directory),
            collection=self.COLLECTION_NAME,
            existing_count=self.collection.count(),
        )

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> int:
        """Index document chunks into the vector store.

        Args:
            chunks: List of document chunks to index
            batch_size: Number of chunks per batch

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        start_time = time.time()
        indexed = 0

        # Filter out already indexed chunks (by content hash)
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("no_new_chunks_to_index", total=len(chunks))
            return 0

        logger.info(
            "indexing_chunks",
            new=len(new_chunks),
            existing=len(chunks) - len(new_chunks),
        )

        # Process in batches
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            # Generate embeddings
            texts = [c.content for c in batch]
            embeddings = self.embedding_provider.embed_texts(texts)

            # Prepare data for ChromaDB
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [c.to_dict() for c in batch]

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            indexed += len(batch)

            logger.debug(
                "batch_indexed",
                batch_start=i,
                batch_size=len(batch),
                total_indexed=indexed,
            )

        elapsed = time.time() - start_time
        logger.info(
            "indexing_complete",
            indexed=indexed,
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(indexed / elapsed, 1) if elapsed > 0 else 0,
        )

        return indexed

    def index_vault(
        self,
        vault_path: Path | None = None,
        source_dirs: list[Path] | None = None,
        chunker: DocumentChunker | None = None,
    ) -> int:
        """Index entire vault or specific directories.

        Args:
            vault_path: Path to vault (uses config if None)
            source_dirs: Specific directories to index
            chunker: Custom chunker (creates default if None)

        Returns:
            Number of chunks indexed
        """
        if vault_path is None:
            vault_path = Path(self.config.vault_path)

        if source_dirs is None and self.config.source_subdirs:
            source_dirs = self.config.source_subdirs

        if chunker is None:
            chunker = DocumentChunker()

        logger.info(
            "indexing_vault",
            vault=str(vault_path),
            source_dirs=[str(d) for d in source_dirs] if source_dirs else None,
        )

        # Chunk the vault
        chunks = chunker.chunk_vault(vault_path, source_dirs)

        # Index chunks
        return self.index_chunks(chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        min_similarity: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar content.

        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of search results sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_text(query)

        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = self._build_where_clause(filter_metadata)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []

        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 / (1 + distance)

                # Filter by minimum similarity
                if similarity < min_similarity:
                    continue

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=(
                            results["documents"][0][i] if results["documents"] else ""
                        ),
                        score=distance,
                        metadata=(
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                        source_file=(
                            results["metadatas"][0][i].get("source_file", "")
                            if results["metadatas"]
                            else ""
                        ),
                    )
                )

        logger.debug(
            "search_completed",
            query_length=len(query),
            results=len(search_results),
            filter=filter_metadata,
        )

        return search_results

    def search_similar_cards(
        self,
        content: str,
        threshold: float = 0.85,
        k: int = 10,
    ) -> list[SearchResult]:
        """Find cards similar to given content (for duplicate detection).

        Args:
            content: Card content to check
            threshold: Similarity threshold (0-1)
            k: Maximum candidates to check

        Returns:
            List of similar cards above threshold
        """
        results = self.search(query=content, k=k, min_similarity=threshold)
        return results

    def _build_where_clause(
        self,
        filter_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build ChromaDB where clause from metadata filters.

        Args:
            filter_metadata: Metadata filters

        Returns:
            ChromaDB where clause
        """
        conditions = []

        for key, value in filter_metadata.items():
            if isinstance(value, list):
                # Use $in for list values
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": str(value)}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        else:
            return {}

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file.

        Args:
            source_file: Source file path

        Returns:
            Number of chunks deleted
        """
        # Get chunks from this source
        results = self.collection.get(
            where={"source_file": {"$eq": source_file}},
        )

        if not results["ids"]:
            return 0

        # Delete chunks
        self.collection.delete(ids=results["ids"])

        logger.info(
            "chunks_deleted",
            source_file=source_file,
            count=len(results["ids"]),
        )

        return len(results["ids"])

    def update_file(
        self,
        file_path: Path,
        chunker: DocumentChunker | None = None,
    ) -> tuple[int, int]:
        """Update index for a single file (delete and re-index).

        Args:
            file_path: Path to file
            chunker: Custom chunker

        Returns:
            Tuple of (deleted_count, indexed_count)
        """
        if chunker is None:
            chunker = DocumentChunker()

        # Delete existing chunks
        deleted = self.delete_by_source(str(file_path))

        # Re-index file
        chunks = chunker.chunk_file(file_path)
        indexed = self.index_chunks(chunks)

        return deleted, indexed

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        collection_count = self.collection.count()

        # Get unique source files
        all_data = self.collection.get(include=["metadatas"])
        source_files = set()
        topics = set()
        chunk_types = set()

        for metadata in all_data.get("metadatas", []):
            if metadata:
                if "source_file" in metadata:
                    source_files.add(metadata["source_file"])
                if "topic" in metadata:
                    topics.add(metadata["topic"])
                if "chunk_type" in metadata:
                    chunk_types.add(metadata["chunk_type"])

        return {
            "total_chunks": collection_count,
            "unique_files": len(source_files),
            "topics": list(topics),
            "chunk_types": list(chunk_types),
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_provider.model_name,
            "embedding_cache_stats": self.embedding_provider.cache_stats(),
        }

    def reset(self) -> bool:
        """Reset the vector store (delete all data).

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("vector_store_reset")
            return True
        except Exception as e:
            logger.error("vector_store_reset_failed", error=str(e))
            return False
