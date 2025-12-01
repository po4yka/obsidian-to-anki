"""Application service for orchestrating agent pipelines."""

import time
from pathlib import Path

from obsidian_anki_sync.agents.slug_utils import generate_agent_slug_base
from obsidian_anki_sync.application.services.retry_handler import RetryHandler
from obsidian_anki_sync.domain.entities.note import NoteMetadata, QAPair
from obsidian_anki_sync.domain.interfaces.llm_provider import ILLMProvider
from obsidian_anki_sync.infrastructure.cache.cache_strategy import AgentCacheStrategy
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class AgentPipelineOrchestrator:
    """Orchestrates agent pipelines for card generation.

    This service focuses solely on coordinating the agent pipeline,
    delegating retry logic and caching to specialized services.
    Follows the Single Responsibility Principle.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        retry_handler: RetryHandler,
        cache_strategy: AgentCacheStrategy,
        config: dict | None = None,
    ):
        """Initialize pipeline orchestrator.

        Args:
            llm_provider: LLM provider for agent operations
            retry_handler: Handler for retry logic
            cache_strategy: Strategy for caching
            config: Optional configuration dictionary
        """
        self.llm_provider = llm_provider
        self.retry_handler = retry_handler
        self.cache_strategy = cache_strategy
        self.config = config or {}
        self.config.setdefault("post_validator_timeout_seconds", 300.0)
        self.config.setdefault("post_validator_retry_backoff_seconds", 3.0)
        self.config.setdefault("post_validator_retry_jitter_seconds", 1.5)

        logger.info("agent_pipeline_orchestrator_initialized")

    def orchestrate_pipeline(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> dict:
        """Orchestrate the complete agent pipeline.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q&A pairs
            file_path: Optional file path for validation

        Returns:
            Pipeline result dictionary
        """
        start_time = time.time()
        correlation_id = self._generate_correlation_id(metadata, start_time)

        logger.info(
            "pipeline_orchestration_start",
            correlation_id=correlation_id,
            note_id=metadata.id,
            qa_pairs_count=len(qa_pairs),
        )

        result = {
            "correlation_id": correlation_id,
            "success": False,
            "stages": {},
            "total_time": 0.0,
            "retry_count": 0,
        }

        try:
            pre_val_result = self._execute_pre_validation(
                note_content, metadata, qa_pairs, file_path
            )
            result["stages"]["pre_validation"] = pre_val_result

            if not pre_val_result["success"]:
                result["error"] = pre_val_result.get("error", "Pre-validation failed")
                return self._finalize_result(result, start_time)

            # Stage 2: Card Generation
            gen_result = self._execute_card_generation(
                note_content, metadata, qa_pairs, correlation_id
            )
            result["stages"]["generation"] = gen_result

            if not gen_result["success"]:
                result["error"] = gen_result.get("error", "Generation failed")
                return self._finalize_result(result, start_time)

            # Stage 3: Post-validation with retries
            post_val_result = self._execute_post_validation_with_retries(
                gen_result["cards"], metadata, correlation_id
            )
            result["stages"]["post_validation"] = post_val_result
            result["retry_count"] = post_val_result.get("retry_count", 0)

            # Finalize result
            result["success"] = post_val_result["success"]
            if result["success"]:
                result["cards"] = post_val_result["cards"]
            else:
                result["error"] = post_val_result.get("error", "Post-validation failed")

        except Exception as e:
            result["error"] = str(e)
            logger.error(
                "pipeline_orchestration_error",
                correlation_id=correlation_id,
                error=str(e),
            )

        return self._finalize_result(result, start_time)

    def _execute_pre_validation(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None,
    ) -> dict:
        """Execute pre-validation stage.

        Args:
            note_content: Note content
            metadata: Note metadata
            qa_pairs: Q&A pairs
            file_path: Optional file path

        Returns:
            Pre-validation result
        """
        pre_validation_enabled = self.config.get("pre_validation_enabled", True)

        if not pre_validation_enabled:
            return {
                "success": True,
                "message": "Pre-validation skipped",
                "time": 0.0,
            }

        start_time = time.time()

        # Basic validation logic
        is_valid = (
            len(qa_pairs) > 0 and metadata.topic and len(metadata.language_tags) > 0
        )

        result = {
            "success": is_valid,
            "time": time.time() - start_time,
        }

        if not is_valid:
            result["error"] = "Basic validation failed"
            result["details"] = "Note missing required fields"

        return result

    def _execute_card_generation(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        correlation_id: str,
    ) -> dict:
        """Execute card generation stage.

        Args:
            note_content: Note content
            metadata: Note metadata
            qa_pairs: Q&A pairs
            correlation_id: Correlation ID for tracking

        Returns:
            Generation result
        """
        start_time = time.time()

        try:
            # Generate slug base
            slug_base = generate_agent_slug_base(metadata)

            # Check cache first
            cache_key = f"generation:{metadata.id}:{hash(note_content)}"
            cached_result = self.cache_strategy.get(cache_key)

            if cached_result:
                logger.debug("using_cached_generation", correlation_id=correlation_id)
                return {
                    "success": True,
                    "cards": cached_result,
                    "cached": True,
                    "time": time.time() - start_time,
                }

            cards = self._generate_cards_with_llm(
                note_content, metadata, qa_pairs, slug_base
            )

            # Cache the result
            if self.cache_strategy.should_cache_result("generation"):
                ttl = self.cache_strategy.get_cache_ttl("generation")
                self.cache_strategy.set(cache_key, cards, ttl)

            return {
                "success": True,
                "cards": cards,
                "cached": False,
                "time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "time": time.time() - start_time,
            }

    def _execute_post_validation_with_retries(
        self,
        cards: list,
        metadata: NoteMetadata,
        correlation_id: str,
    ) -> dict:
        """Execute post-validation with retry logic.

        Args:
            cards: Generated cards
            metadata: Note metadata
            correlation_id: Correlation ID

        Returns:
            Post-validation result
        """

        def validate_operation():
            return self._validate_cards(cards, metadata)

        # Use retry handler for post-validation
        retry_result = self.retry_handler.execute_with_retry(
            validate_operation,
            context={"correlation_id": correlation_id, "stage": "post_validation"},
        )

        if retry_result.success:
            return {
                "success": True,
                "cards": retry_result.result.get("cards", cards),
                "retry_count": retry_result.attempts - 1,  # Don't count initial attempt
                "time": retry_result.total_time,
                "needs_highlight": False,
            }
        else:
            return {
                "success": False,
                "error": str(retry_result.error),
                "retry_count": retry_result.attempts - 1,
                "time": retry_result.total_time,
                "needs_highlight": retry_result.context.get("stage")
                == "post_validation",
            }

    def _generate_cards_with_llm(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> list:
        """Generate cards using LLM provider.

        Args:
            note_content: Note content
            metadata: Note metadata
            qa_pairs: Q&A pairs
            slug_base: Base slug for cards

        Returns:
            List of generated cards
        """
        cards = []

        for i, qa_pair in enumerate(qa_pairs):
            for lang in metadata.language_tags:
                card = {
                    "slug": f"{slug_base}-p{i}-{lang}",
                    "language": lang,
                    "content": f"Question: {qa_pair.question_en}\nAnswer: {qa_pair.answer_en}",
                    "note_id": metadata.id,
                }
                cards.append(card)

        return cards

    def _validate_cards(self, cards: list, metadata: NoteMetadata) -> dict:
        """Validate generated cards.

        Args:
            cards: Cards to validate
            metadata: Note metadata

        Returns:
            Validation result
        """
        is_valid = len(cards) > 0 and all(
            card.get("slug") and card.get("language") for card in cards
        )

        result = {
            "valid": is_valid,
            "cards": cards,
        }

        if not is_valid:
            result["error"] = "Card validation failed"
            # Simulate auto-fix by adding missing fields
            for card in cards:
                if not card.get("slug"):
                    card["slug"] = f"fixed-{hash(str(card))}"
                if not card.get("language"):
                    card["language"] = metadata.primary_language

        return result

    def _generate_correlation_id(
        self, metadata: NoteMetadata, start_time: float
    ) -> str:
        """Generate correlation ID for tracking."""
        import hashlib

        key = f"{metadata.id}:{int(start_time * 1000)}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def _finalize_result(self, result: dict, start_time: float) -> dict:
        """Finalize pipeline result with timing."""
        result["total_time"] = time.time() - start_time
        return result
