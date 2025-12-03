"""RAG Service - High-level API for RAG operations.

Provides unified interface for:
- Context enrichment during card generation
- Duplicate detection via semantic similarity
- Few-shot example retrieval
- Related concept discovery
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import diskcache

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.logging import get_logger

from .document_chunker import ChunkType
from .vector_store import SearchResult, VaultVectorStore

logger = get_logger(__name__)


@dataclass
class RelatedConcept:
    """A related concept from the knowledge base."""

    title: str
    content: str
    topic: str
    similarity: float
    source_file: str
    chunk_type: str


@dataclass
class DuplicateCheckResult:
    """Result of duplicate detection check."""

    is_duplicate: bool
    confidence: float
    similar_items: list[SearchResult]
    recommendation: str


@dataclass
class FewShotExample:
    """A few-shot example for card generation."""

    question: str
    answer: str
    topic: str
    difficulty: str
    source_file: str


class RAGService:
    """High-level RAG service for flashcard generation enhancement.

    Provides methods for:
    - Getting related concepts to enrich card context
    - Finding similar cards for duplicate detection
    - Retrieving few-shot examples for improved generation
    """

    def __init__(
        self,
        config: Config,
        vector_store: VaultVectorStore | None = None,
    ):
        """Initialize RAG service.

        Args:
            config: Application configuration
            vector_store: Custom vector store (creates default if None)
        """
        self.config = config
        self._vector_store = vector_store
        self._initialized = False

        # Initialize cache
        cache_dir = config.db_path.parent / ".cache" / "rag_service"
        self._cache = diskcache.Cache(
            directory=str(cache_dir),
            size_limit=500 * 1024 * 1024,  # 500MB limit
            eviction_policy="least-recently-used",
        )

        logger.info("rag_service_created", cache_dir=str(cache_dir))

    @property
    def vector_store(self) -> VaultVectorStore:
        """Get or create vector store (lazy initialization)."""
        if self._vector_store is None:
            self._vector_store = VaultVectorStore(self.config)
            self._initialized = True
        return self._vector_store

    def is_indexed(self) -> bool:
        """Check if vault has been indexed.

        Returns:
            True if index has content
        """
        try:
            stats = self.vector_store.get_stats()
            return bool(stats.get("total_chunks", 0) > 0)  # type: ignore[no-any-return]
        except Exception:
            return False

    def index_vault(
        self,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Index the vault for RAG.

        Args:
            force_reindex: If True, reset and re-index everything

        Returns:
            Indexing statistics
        """
        if force_reindex:
            self.vector_store.reset()
            logger.info("vault_index_reset")

        indexed = self.vector_store.index_vault()

        return {
            "chunks_indexed": indexed,
            "stats": self.vector_store.get_stats(),
        }

    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a deterministic cache key."""
        # Convert kwargs to string representation for hashing
        # We sort keys to ensure deterministic order
        content_parts = []
        for k, v in sorted(kwargs.items()):
            content_parts.append(f"{k}:{v}")

        content = "|".join(content_parts)
        hash_val = hashlib.md5(content.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    async def get_related_concepts(
        self,
        content: str,
        k: int = 5,
        topic_filter: str | None = None,
        language: str | None = None,
    ) -> list[RelatedConcept]:
        """Get related concepts for context enrichment.

        Args:
            content: Note content or question
            k: Number of related concepts to return
            topic_filter: Optional topic filter
            language: Optional language filter (en/ru)

        Returns:
            List of related concepts
        """
        # Check cache
        cache_key = self._get_cache_key(
            "related", content=content, k=k, topic=topic_filter, language=language
        )

        if cache_key in self._cache:
            logger.debug("rag_cache_hit", operation="get_related_concepts")
            return self._cache[cache_key]

        # Build filters
        filters: dict[str, Any] = {}
        if topic_filter:
            filters["topic"] = topic_filter

        # Search for related content
        results = self.vector_store.search(
            query=content,
            k=k * 2,  # Get more to filter
            filter_metadata=filters if filters else None,
            min_similarity=0.3,  # Lower threshold for related concepts
        )

        # Filter by chunk type (prefer summaries and key points)
        preferred_types = {
            ChunkType.SUMMARY_EN.value,
            ChunkType.SUMMARY_RU.value,
            ChunkType.KEY_POINTS_EN.value,
            ChunkType.KEY_POINTS_RU.value,
        }

        # If language specified, filter by language
        if language:
            if language.lower() == "en":
                preferred_types = {
                    ChunkType.SUMMARY_EN.value,
                    ChunkType.KEY_POINTS_EN.value,
                }
            elif language.lower() == "ru":
                preferred_types = {
                    ChunkType.SUMMARY_RU.value,
                    ChunkType.KEY_POINTS_RU.value,
                }

        # Sort by preference and similarity
        def score_result(r: SearchResult) -> tuple[int, float]:
            chunk_type = r.metadata.get("chunk_type", "")
            type_score = 0 if chunk_type in preferred_types else 1
            return (type_score, -r.similarity)

        sorted_results = sorted(results, key=score_result)

        # Convert to RelatedConcept objects
        concepts: list[RelatedConcept] = []
        seen_files: set[str] = set()

        for result in sorted_results:
            # Deduplicate by source file
            source = result.source_file
            if source in seen_files:
                continue
            seen_files.add(source)

            concepts.append(
                RelatedConcept(
                    title=result.metadata.get("title", ""),
                    content=result.content,
                    topic=result.metadata.get("topic", ""),
                    similarity=result.similarity,
                    source_file=source,
                    chunk_type=result.metadata.get("chunk_type", ""),
                )
            )

            if len(concepts) >= k:
                break

        logger.debug(
            "related_concepts_retrieved",
            query_length=len(content),
            results=len(concepts),
            topic_filter=topic_filter,
        )

        # Cache result
        self._cache[cache_key] = concepts
        return concepts

    async def check_duplicate(
        self,
        question: str,
        answer: str,
        threshold: float = 0.85,
    ) -> DuplicateCheckResult:
        """Check if a card is a potential duplicate.

        Args:
            question: Card question
            answer: Card answer
            threshold: Similarity threshold for duplicate detection

        Returns:
            Duplicate check result with recommendations
        """
        # Search for similar questions
        combined_content = f"Q: {question}\nA: {answer}"
        results = self.vector_store.search_similar_cards(
            content=combined_content,
            threshold=threshold,
            k=5,
        )

        # Filter to only Q&A chunks
        qa_types = {
            ChunkType.QUESTION_EN.value,
            ChunkType.QUESTION_RU.value,
            ChunkType.ANSWER_EN.value,
            ChunkType.ANSWER_RU.value,
        }

        similar_qa = [r for r in results if r.metadata.get("chunk_type") in qa_types]

        # Determine if duplicate
        is_duplicate = len(similar_qa) > 0
        confidence = max((r.similarity for r in similar_qa), default=0.0)

        # Generate recommendation
        if confidence >= 0.95:
            recommendation = "Highly likely duplicate - skip this card"
        elif confidence >= 0.85:
            recommendation = "Probable duplicate - review before creating"
        elif confidence >= 0.70:
            recommendation = (
                "Similar content exists - consider merging or differentiating"
            )
        else:
            recommendation = "No significant duplicates found"

        logger.debug(
            "duplicate_check_completed",
            is_duplicate=is_duplicate,
            confidence=confidence,
            similar_count=len(similar_qa),
        )

        return DuplicateCheckResult(
            is_duplicate=is_duplicate,
            confidence=confidence,
            similar_items=similar_qa,
            recommendation=recommendation,
        )

    async def get_few_shot_examples(
        self,
        topic: str,
        difficulty: str | None = None,
        k: int = 3,
    ) -> list[FewShotExample]:
        """Get few-shot examples for card generation.

        Retrieves similar successful Q&A pairs to use as examples
        for the generation model.

        Args:
            topic: Topic to find examples for
            difficulty: Optional difficulty filter
            k: Number of examples to return

        Returns:
            List of few-shot examples
        """
        # Check cache
        cache_key = self._get_cache_key(
            "few_shot", topic=topic, difficulty=difficulty, k=k
        )

        if cache_key in self._cache:
            logger.debug("rag_cache_hit", operation="get_few_shot_examples")
            return self._cache[cache_key]

        # Build query from topic
        query = f"Example cards for topic: {topic}"
        if difficulty:
            query += f" (difficulty: {difficulty})"

        # Build filters
        filters: dict[str, Any] = {"topic": topic}
        if difficulty:
            filters["difficulty"] = difficulty

        # Search for Q&A content
        results = self.vector_store.search(
            query=query,
            k=k * 3,  # Get more to find Q/A pairs
            filter_metadata=filters,
            min_similarity=0.3,
        )

        # Group by source file to find Q/A pairs
        by_file: dict[str, dict[str, SearchResult]] = {}

        for result in results:
            source = result.source_file
            chunk_type = result.metadata.get("chunk_type", "")

            if source not in by_file:
                by_file[source] = {}

            # Store question and answer chunks
            if chunk_type in (ChunkType.QUESTION_EN.value, ChunkType.QUESTION_RU.value):
                by_file[source]["question"] = result
            elif chunk_type in (ChunkType.ANSWER_EN.value, ChunkType.ANSWER_RU.value):
                by_file[source]["answer"] = result

        # Create examples from complete Q/A pairs
        examples: list[FewShotExample] = []

        for source, chunks in by_file.items():
            if "question" in chunks and "answer" in chunks:
                q_result = chunks["question"]
                a_result = chunks["answer"]

                examples.append(
                    FewShotExample(
                        question=q_result.content,
                        answer=a_result.content,
                        topic=q_result.metadata.get("topic", topic),
                        difficulty=q_result.metadata.get("difficulty", "medium"),
                        source_file=source,
                    )
                )

                if len(examples) >= k:
                    break

        logger.debug(
            "few_shot_examples_retrieved",
            topic=topic,
            difficulty=difficulty,
            examples=len(examples),
        )

        # Cache result
        self._cache[cache_key] = examples
        return examples

    async def enrich_context(
        self,
        note_content: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        """Enrich note context with related information.

        Combines multiple RAG operations to provide comprehensive
        context for card generation.

        Args:
            note_content: Original note content
            topic: Optional topic hint

        Returns:
            Enrichment data including related concepts and examples
        """
        # Get related concepts
        related = await self.get_related_concepts(
            content=note_content,
            k=3,
            topic_filter=topic,
        )

        # Get few-shot examples if topic known
        examples: list[FewShotExample] = []
        if topic:
            examples = await self.get_few_shot_examples(topic=topic, k=2)

        # Build enrichment data
        enrichment = {
            "related_concepts": [
                {
                    "title": c.title,
                    "content": c.content[:500],  # Truncate for context
                    "topic": c.topic,
                    "similarity": c.similarity,
                }
                for c in related
            ],
            "few_shot_examples": [
                {
                    "question": e.question[:300],
                    "answer": e.answer[:500],
                    "topic": e.topic,
                }
                for e in examples
            ],
            "topics_found": list({c.topic for c in related if c.topic}),
        }

        logger.info(
            "context_enriched",
            related_count=len(related),
            examples_count=len(examples),
            topics=enrichment["topics_found"],
        )

        return enrichment

    def get_stats(self) -> dict[str, Any]:
        """Get RAG service statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "initialized": self._initialized,
            "indexed": self.is_indexed(),
            "vector_store_stats": (
                self.vector_store.get_stats() if self._vector_store else None
            ),
        }


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service(config: Config | None = None) -> RAGService:
    """Get or create singleton RAG service instance.

    Args:
        config: Application configuration (uses global if None)

    Returns:
        RAGService instance
    """
    global _rag_service

    if _rag_service is None:
        if config is None:
            from obsidian_anki_sync.config import get_config

            config = get_config()
        _rag_service = RAGService(config)

    return _rag_service


def reset_rag_service() -> None:
    """Reset the singleton RAG service (for testing)."""
    global _rag_service
    _rag_service = None
