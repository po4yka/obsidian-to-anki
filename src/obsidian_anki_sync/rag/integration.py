"""RAG integration helpers for the agent pipeline.

Provides convenient functions to integrate RAG capabilities
into card generation, validation, and enrichment agents.
"""

from typing import Any

from ..config import Config
from ..utils.logging import get_logger
from .rag_service import RAGService, get_rag_service

logger = get_logger(__name__)


class RAGIntegration:
    """Helper class for integrating RAG into the agent pipeline.

    Provides simplified methods for:
    - Context enrichment during generation
    - Duplicate detection during pre-validation
    - Few-shot example retrieval
    """

    def __init__(self, config: Config):
        """Initialize RAG integration.

        Args:
            config: Application configuration
        """
        self.config = config
        self._service: RAGService | None = None
        self._enabled = getattr(config, "rag_enabled", False)

        logger.info(
            "rag_integration_initialized",
            enabled=self._enabled,
        )

    @property
    def service(self) -> RAGService | None:
        """Get RAG service (lazy initialization)."""
        if not self._enabled:
            return None

        if self._service is None:
            self._service = get_rag_service(self.config)

            # Check if index exists
            if not self._service.is_indexed():
                logger.warning(
                    "rag_index_missing",
                    hint="Run 'obsidian-anki-sync rag index' to build the index",
                )

        return self._service

    @property
    def is_enabled(self) -> bool:
        """Check if RAG is enabled and indexed."""
        return self._enabled and self.service is not None and self.service.is_indexed()

    async def enrich_generation_context(
        self,
        note_content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Enrich context for card generation.

        Args:
            note_content: Original note content
            metadata: Note metadata (topic, difficulty, etc.)

        Returns:
            Enrichment data with related concepts and examples
        """
        if not self.is_enabled:
            return {}

        if not getattr(self.config, "rag_context_enrichment", True):
            return {}

        try:
            topic = metadata.get("topic", "")
            enrichment = await self.service.enrich_context(
                note_content=note_content,
                topic=topic,
            )

            logger.debug(
                "generation_context_enriched",
                related=len(enrichment.get("related_concepts", [])),
                examples=len(enrichment.get("few_shot_examples", [])),
            )

            return enrichment

        except Exception as e:
            logger.warning(
                "rag_enrichment_failed",
                error=str(e),
            )
            return {}

    async def check_for_duplicates(
        self,
        question: str,
        answer: str = "",
    ) -> dict[str, Any]:
        """Check if content is a potential duplicate.

        Args:
            question: Question content
            answer: Answer content (optional)

        Returns:
            Duplicate check result
        """
        if not self.is_enabled:
            return {"is_duplicate": False, "confidence": 0.0}

        if not getattr(self.config, "rag_duplicate_detection", True):
            return {"is_duplicate": False, "confidence": 0.0}

        try:
            threshold = getattr(self.config, "rag_similarity_threshold", 0.85)
            result = await self.service.check_duplicate(
                question=question,
                answer=answer,
                threshold=threshold,
            )

            return {
                "is_duplicate": result.is_duplicate,
                "confidence": result.confidence,
                "recommendation": result.recommendation,
                "similar_count": len(result.similar_items),
            }

        except Exception as e:
            logger.warning(
                "rag_duplicate_check_failed",
                error=str(e),
            )
            return {"is_duplicate": False, "confidence": 0.0}

    async def get_examples_for_generation(
        self,
        topic: str,
        difficulty: str | None = None,
    ) -> list[dict[str, str]]:
        """Get few-shot examples for card generation.

        Args:
            topic: Topic to find examples for
            difficulty: Optional difficulty filter

        Returns:
            List of example Q&A pairs
        """
        if not self.is_enabled:
            return []

        if not getattr(self.config, "rag_few_shot_examples", True):
            return []

        try:
            k = getattr(self.config, "rag_search_k", 5)
            examples = await self.service.get_few_shot_examples(
                topic=topic,
                difficulty=difficulty,
                k=k,
            )

            return [
                {
                    "question": ex.question,
                    "answer": ex.answer,
                    "topic": ex.topic,
                }
                for ex in examples
            ]

        except Exception as e:
            logger.warning(
                "rag_examples_failed",
                error=str(e),
            )
            return []

    def build_enhanced_prompt(
        self,
        base_prompt: str,
        enrichment: dict[str, Any],
        max_context_length: int = 2000,
    ) -> str:
        """Build enhanced prompt with RAG context.

        Args:
            base_prompt: Original prompt
            enrichment: Enrichment data from enrich_generation_context()
            max_context_length: Maximum characters for context section

        Returns:
            Enhanced prompt with RAG context
        """
        if not enrichment:
            return base_prompt

        context_parts: list[str] = []

        # Add related concepts
        related = enrichment.get("related_concepts", [])
        if related:
            concepts_text = []
            for concept in related[:3]:  # Limit to top 3
                title = concept.get("title", "")
                content = concept.get("content", "")[:300]
                if title:
                    concepts_text.append(f"- {title}: {content}")
                else:
                    concepts_text.append(f"- {content}")

            if concepts_text:
                context_parts.append(
                    "Related concepts from knowledge base:\n" + "\n".join(concepts_text)
                )

        # Add few-shot examples
        examples = enrichment.get("few_shot_examples", [])
        if examples:
            examples_text = []
            for ex in examples[:2]:  # Limit to top 2
                q = ex.get("question", "")[:150]
                a = ex.get("answer", "")[:200]
                examples_text.append(f"Q: {q}\nA: {a}")

            if examples_text:
                context_parts.append(
                    "Example Q&A from similar cards:\n" + "\n---\n".join(examples_text)
                )

        if not context_parts:
            return base_prompt

        # Combine context
        context_section = "\n\n".join(context_parts)

        # Truncate if needed
        if len(context_section) > max_context_length:
            context_section = context_section[:max_context_length] + "..."

        # Insert context into prompt
        enhanced_prompt = (
            f"{base_prompt}\n\n"
            f"## Additional Context (from RAG)\n\n"
            f"{context_section}"
        )

        return enhanced_prompt


# Singleton integration instance
_rag_integration: RAGIntegration | None = None


def get_rag_integration(config: Config | None = None) -> RAGIntegration:
    """Get or create singleton RAG integration instance.

    Args:
        config: Application configuration (uses global if None)

    Returns:
        RAGIntegration instance
    """
    global _rag_integration

    if _rag_integration is None:
        if config is None:
            from ..config import get_config

            config = get_config()
        _rag_integration = RAGIntegration(config)

    return _rag_integration


def reset_rag_integration() -> None:
    """Reset the singleton RAG integration (for testing)."""
    global _rag_integration
    _rag_integration = None
