"""Card generation component for SyncEngine.

Handles generation of cards from Q/A pairs using either agent system or direct APF generation.
"""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import diskcache

from obsidian_anki_sync.agents.models import HighlightResult
from obsidian_anki_sync.apf.generator import APFGenerator
from obsidian_anki_sync.apf.html_validator import validate_card_html
from obsidian_anki_sync.apf.linter import validate_apf
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import CardGenerationError
from obsidian_anki_sync.models import Card, NoteMetadata, QAPair
from obsidian_anki_sync.sync.slug_generator import create_manifest, generate_slug
from obsidian_anki_sync.utils.async_runner import AsyncioRunner
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.guid import deterministic_guid
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator
    from obsidian_anki_sync.agents.models import AgentPipelineResult

logger = get_logger(__name__)


class CardGenerator:
    """Handles card generation from Q/A pairs."""

    def __init__(
        self,
        config: Config,
        apf_gen: APFGenerator,
        agent_orchestrator: "LangGraphOrchestrator | None" = None,
        use_agents: bool = False,
        agent_card_cache: diskcache.Cache | None = None,
        apf_card_cache: diskcache.Cache | None = None,
        cache_hits: int = 0,
        cache_misses: int = 0,
        cache_stats: dict[str, Any] | None = None,
        slug_counters: dict[str, int] | None = None,
        slug_counter_lock: Any = None,
        stats: dict[str, Any] | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
        async_runner: AsyncioRunner | None = None,
    ):
        """Initialize card generator.

        Args:
            config: Service configuration
            apf_gen: APFGenerator instance
            agent_orchestrator: Optional agent orchestrator
            use_agents: Whether to use agent system
            agent_card_cache: Cache for agent-generated cards
            apf_card_cache: Cache for APF-generated cards
            cache_hits: Cache hit counter
            cache_misses: Cache miss counter
            cache_stats: Cache statistics dict
            slug_counters: Thread-safe slug counters dict
            slug_counter_lock: Lock for slug counters
            stats: Statistics dictionary to update
            existing_cards_for_duplicate_detection: Existing cards from Anki for duplicate detection
        """
        self.config = config
        self.apf_gen = apf_gen
        self.agent_orchestrator = agent_orchestrator
        self.use_agents = use_agents
        self._agent_card_cache = agent_card_cache
        self._apf_card_cache = apf_card_cache
        self._cache_hits = cache_hits
        self._cache_misses = cache_misses
        self._cache_stats = cache_stats or {
            "hits": 0,
            "misses": 0,
            "generation_times": [],
        }
        self._slug_counters = slug_counters or {}
        self._slug_counter_lock = slug_counter_lock
        self.stats = stats or {}
        self.existing_cards_for_duplicate_detection = (
            existing_cards_for_duplicate_detection
        )
        self._async_runner = async_runner or AsyncioRunner.get_global()

    def set_existing_cards_for_duplicate_detection(self, existing_cards: list | None):
        """Set existing cards for duplicate detection.

        Args:
            existing_cards: List of existing cards from Anki
        """
        self.existing_cards_for_duplicate_detection = existing_cards

    def generate_with_agents(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        relative_path: str,
    ) -> list[Card]:
        """Generate all cards for a note using the agent system.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: List of Q/A pairs
            relative_path: Relative path to note

        Returns:
            List of generated cards
        """
        if not self.use_agents or not self.agent_orchestrator:
            msg = "Agent system not initialized"
            raise RuntimeError(msg)

        # Compute content hash for the entire note
        content_components = [
            metadata.id,
            metadata.title,
            metadata.topic,
            ",".join(sorted(metadata.subtopics)),
            ",".join(sorted(metadata.tags)),
            note_content,
        ]
        note_content_hash = hashlib.sha256(
            "\n".join(str(c) for c in content_components).encode("utf-8")
        ).hexdigest()[:16]

        # Check cache
        cache_key = f"{metadata.id}:{relative_path}:{note_content_hash}"
        cache_hit = False
        try:
            if self._agent_card_cache:
                cache_start_time = time.time()
                cached_cards = self._agent_card_cache.get(cache_key)
                cache_duration = time.time() - cache_start_time
                if cached_cards is not None:
                    cache_hit = True
                    self._cache_hits += 1
                    logger.debug(
                        "cache_hit",
                        operation="agent_card_generation",
                        cache_key=cache_key[:8],
                        note=relative_path,
                        cards_count=len(cached_cards),
                        cache_duration=round(cache_duration, 4),
                        content_hash=note_content_hash[:8],
                    )
                    return cached_cards  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(
                "cache_read_error",
                operation="agent_card_generation",
                cache_key=cache_key[:8],
                error=str(e),
                error_type=type(e).__name__,
            )

        if not cache_hit:
            self._cache_misses += 1
            logger.debug(
                "cache_miss",
                operation="agent_card_generation",
                cache_key=cache_key[:8],
                note=relative_path,
                reason="not_found",
            )
            logger.info(
                "generating_cards_with_agents",
                note=relative_path,
                qa_pairs=len(qa_pairs),
                cache_miss=True,
            )

        file_path = self.config.vault_path / relative_path

        if not file_path.exists():
            logger.warning(
                "file_path_not_found_for_agent_processing",
                relative_path=relative_path,
                computed_path=str(file_path),
            )

        # Handle both sync and async orchestrators
        if hasattr(self.agent_orchestrator, "process_note"):
            import inspect

            if inspect.iscoroutinefunction(self.agent_orchestrator.process_note):
                result = self._async_runner.run(
                    self.agent_orchestrator.process_note(
                        note_content=note_content,
                        metadata=metadata,
                        qa_pairs=qa_pairs,
                        file_path=Path(
                            file_path) if file_path.exists() else None,
                        existing_cards=self.existing_cards_for_duplicate_detection,
                    )
                )
            else:
                result = self.agent_orchestrator.process_note(
                    note_content=note_content,
                    metadata=metadata,
                    qa_pairs=qa_pairs,
                    file_path=Path(file_path) if file_path.exists() else None,
                    existing_cards=self.existing_cards_for_duplicate_detection,
                )
        else:
            msg = "Orchestrator does not have process_note method"
            raise RuntimeError(msg)

        # Track metrics
        if result.post_validation and not result.post_validation.is_valid:
            self.stats["validation_errors"] = self.stats.get(
                "validation_errors", 0) + 1
        if result.retry_count > 0:
            self.stats["auto_fix_attempts"] = (
                self.stats.get("auto_fix_attempts", 0) + result.retry_count
            )
            if result.success:
                self.stats["auto_fix_successes"] = (
                    self.stats.get("auto_fix_successes", 0) + 1
                )

        if not result.success or not result.generation:
            # Use the robust error extraction method
            error_msg = self._extract_pipeline_error_message(result)
            highlight_hint = self._format_highlight_hint(
                result.highlight_result)
            if highlight_hint:
                logger.info(
                    "highlight_agent_summary",
                    note=relative_path,
                    candidates=len(result.highlight_result.qa_candidates)
                    if result.highlight_result
                    else 0,
                )
                error_msg = f"{error_msg}\n{highlight_hint}"

            # Determine error type for fail-fast categorization
            error_type = self._determine_error_type(result)
            error_details = self._extract_error_details(result)

            msg = f"Agent pipeline failed: {error_msg}"
            raise CardGenerationError(
                msg,
                error_type=error_type,
                error_details=error_details,
                note_path=relative_path,
                suggestion=self._get_error_suggestion(error_type),
            )

        # Convert GeneratedCard to Card instances
        cards = self.agent_orchestrator.convert_to_cards(
            result.generation.cards,
            metadata,
            qa_pairs,
            Path(file_path) if file_path.exists() else None,
        )

        # Update card metadata
        for card in cards:
            card.manifest.source_path = relative_path
            card.manifest.note_id = metadata.id
            card.manifest.note_title = metadata.title

        # Cache the results
        try:
            if self._agent_card_cache:
                cache_write_start = time.time()
                self._agent_card_cache.set(cache_key, cards)
                cache_write_duration = time.time() - cache_write_start
                logger.debug(
                    "cache_write",
                    operation="agent_card_generation",
                    cache_key=cache_key[:8],
                    duration=round(cache_write_duration, 4),
                    cards_count=len(cards),
                    cache_size_bytes=sum(
                        len(str(card).encode()) for card in cards
                    ),
                )
        except Exception as e:
            logger.warning(
                "cache_write_error",
                operation="agent_card_generation",
                cache_key=cache_key[:8],
                error=str(e),
                error_type=type(e).__name__,
            )

        logger.info(
            "agent_generation_success",
            note=relative_path,
            cards_generated=len(cards),
            time=result.total_time,
        )

        return cards

    def generate_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        relative_path: str,
        lang: str,
        existing_slugs: set[str],
        note_content: str = "",
        all_qa_pairs: list[QAPair] | None = None,
    ) -> Card:
        """Generate a single card.

        Args:
            qa_pair: Q/A pair to generate card for
            metadata: Note metadata
            relative_path: Relative path to note
            lang: Language code
            existing_slugs: Set of existing slugs
            note_content: Full note content (required for agent system)
            all_qa_pairs: All Q/A pairs from note (required for agent system)

        Returns:
            Generated card
        """
        start_time = time.time()

        # Use agent system if enabled
        if self.use_agents:
            if not note_content or all_qa_pairs is None:
                msg = "note_content and all_qa_pairs required when using agent system"
                raise ValueError(msg)

            # Generate all cards for the note (cached)
            all_cards = self.generate_with_agents(
                note_content, metadata, all_qa_pairs, relative_path
            )

            # Find the specific card for this qa_pair and lang
            for card in all_cards:
                if card.manifest.card_index == qa_pair.card_index and card.lang == lang:
                    elapsed_ms = round((time.time() - start_time) * 1000, 2)
                    self._cache_stats["hits"] += 1
                    logger.debug(
                        "card_generation_cache_hit_agent",
                        slug=card.slug,
                        elapsed_ms=elapsed_ms,
                    )
                    return card

            msg = f"Agent system did not generate card for index={qa_pair.card_index}, lang={lang}"
            raise ValueError(msg)

        # Check cache for non-agent generation
        content_hash = compute_content_hash(qa_pair, metadata, lang)
        cache_key = f"{relative_path}:{qa_pair.card_index}:{lang}:{content_hash}"

        cache_hit = False
        try:
            if self._apf_card_cache:
                cache_start_time = time.time()
                cached_card = self._apf_card_cache.get(cache_key)
                cache_duration = time.time() - cache_start_time
                if cached_card is not None:
                    if cached_card.content_hash == content_hash:
                        cache_hit = True
                        elapsed_ms = round(
                            (time.time() - start_time) * 1000, 2)
                        self._cache_hits += 1
                        self._cache_stats["hits"] += 1
                        logger.debug(
                            "cache_hit",
                            operation="apf_card_generation",
                            cache_key=cache_key[:8],
                            slug=cached_card.slug,
                            elapsed_ms=elapsed_ms,
                            cache_duration=round(cache_duration, 4),
                        )
                        return cached_card  # type: ignore[no-any-return]
                    else:
                        logger.debug(
                            "cache_miss",
                            operation="apf_card_generation",
                            cache_key=cache_key[:8],
                            reason="content_hash_mismatch",
                        )
        except Exception as e:
            logger.warning(
                "cache_read_error",
                operation="apf_card_generation",
                cache_key=cache_key[:8],
                error=str(e),
                error_type=type(e).__name__,
            )

        if not cache_hit:
            self._cache_misses += 1
            self._cache_stats["misses"] += 1
            logger.debug(
                "cache_miss",
                operation="apf_card_generation",
                cache_key=cache_key[:8],
                reason="not_found",
            )

        # Generate slug - use thread-safe method in parallel mode
        if self._slug_counters and self._slug_counter_lock:
            import re
            import unicodedata

            path_parts = Path(relative_path).with_suffix("").parts
            slug_parts = []
            for part in path_parts:
                normalized = unicodedata.normalize("NFKD", part)
                ascii_segment = normalized.encode(
                    "ascii", "ignore").decode("ascii")
                ascii_segment = re.sub(
                    r"[^a-z0-9-]", "-", ascii_segment.lower())
                ascii_segment = re.sub(r"-+", "-", ascii_segment).strip("-")
                if ascii_segment:
                    slug_parts.append(ascii_segment)
            sanitized = "-".join(slug_parts) or "note"
            base_without_suffix = f"{sanitized}-p{qa_pair.card_index:02d}"
            slug_base = base_without_suffix[:70]

            # Use thread-safe counter
            with self._slug_counter_lock:
                initial_slug = f"{slug_base}-{lang}"
                if initial_slug not in self._slug_counters:
                    self._slug_counters[initial_slug] = 0
                    slug = initial_slug
                else:
                    self._slug_counters[initial_slug] += 1
                    collision_count = self._slug_counters[initial_slug]
                    slug = f"{initial_slug}-{collision_count}"
            hash6 = None
        else:
            # Sequential mode - use standard generate_slug
            slug, slug_base, hash6 = generate_slug(
                relative_path, qa_pair.card_index, lang, existing_slugs
            )

        # Compute deterministic GUID
        guid = deterministic_guid(
            [metadata.id, relative_path, str(qa_pair.card_index), lang]
        )

        # Create manifest
        manifest = create_manifest(
            slug,
            slug_base if "slug_base" in locals() else slug.rsplit("-", 1)[0],
            lang,
            relative_path,
            qa_pair.card_index,
            metadata,
            guid,
            hash6,
        )

        # Generate APF card via LLM
        card = cast(
            "Card", self.apf_gen.generate_card(
                qa_pair, metadata, manifest, lang)
        )

        # Ensure content hash is set
        if not card.content_hash:
            card.content_hash = content_hash

        # Validate APF format
        validation = validate_apf(card.apf_html, slug)
        if validation.errors:
            self.stats["validation_errors"] = self.stats.get(
                "validation_errors", 0
            ) + len(validation.errors)
            logger.error("apf_validation_errors", slug=slug,
                         errors=validation.errors)
            msg = f"APF validation failed for {slug}: {validation.errors[0]}"
            raise ValueError(msg)
        if validation.warnings:
            logger.debug(
                "apf_validation_warnings", slug=slug, warnings=validation.warnings
            )

        html_errors = validate_card_html(card.apf_html)
        if html_errors:
            logger.error("apf_html_invalid", slug=slug, errors=html_errors)
            msg = f"Invalid HTML formatting for {slug}: {html_errors[0]}"
            raise ValueError(msg)

        # Cache the generated card
        try:
            if self._apf_card_cache:
                cache_write_start = time.time()
                self._apf_card_cache.set(cache_key, card)
                cache_write_duration = time.time() - cache_write_start
                logger.debug(
                    "cache_write",
                    operation="apf_card_generation",
                    cache_key=cache_key[:8],
                    slug=slug,
                    duration=round(cache_write_duration, 4),
                    cache_size_bytes=len(str(card).encode()),
                    content_hash=content_hash[:8],
                )
        except Exception as e:
            logger.warning(
                "cache_write_error",
                operation="apf_card_generation",
                cache_key=cache_key[:8],
                error=str(e),
                error_type=type(e).__name__,
            )

        # Log generation time
        elapsed = time.time() - start_time
        self._cache_stats["generation_times"].append(elapsed)
        logger.info("card_generated", slug=slug,
                    elapsed_seconds=round(elapsed, 2))

        return card

    def _extract_pipeline_error_message(
        self, result: "AgentPipelineResult"
    ) -> str:
        """Extract a comprehensive and user-friendly error message from pipeline result.

        Args:
            result: The AgentPipelineResult from the failed pipeline

        Returns:
            A detailed error message suitable for user display
        """
        # Validate result object structure
        if not hasattr(result, 'success') or not hasattr(result, 'total_time'):
            return "Pipeline failed: Invalid result structure"

        try:
            # Category 1: Post-validation errors (highest priority - most specific)
            if (hasattr(result, 'post_validation') and result.post_validation and
                hasattr(result.post_validation, 'is_valid') and
                    hasattr(result.post_validation, 'error_type')):

                if not result.post_validation.is_valid:
                    error_type = getattr(
                        result.post_validation, 'error_type', 'unknown')

                    # Use standardized error messages for post-validation too
                    standardized_messages = {
                        'syntax': 'Card syntax error: Generated cards contain invalid formatting or structure.',
                        'factual': 'Card accuracy error: Generated cards contain factual inaccuracies or incorrect information.',
                        'semantic': 'Card semantic error: Generated cards have logical inconsistencies or unclear content.',
                        'template': 'Card template error: Generated cards do not follow the required format specifications.',
                        'none': 'Card validation failed unexpectedly. Review the generated cards for issues.',
                    }

                    # Get the standardized message
                    standardized_message = standardized_messages.get(
                        error_type,
                        f'Card validation failed ({error_type}): Generated cards have quality issues.'
                    )

                    # Include specific error details if available and not too verbose
                    error_details = getattr(
                        result.post_validation, 'error_details', '')
                    if error_details and error_details.strip() and len(error_details) < 150:
                        return f"{standardized_message} Details: {error_details}"
                    else:
                        return standardized_message

            # Category 2: Pre-validation errors (second priority)
            if (hasattr(result, 'pre_validation') and result.pre_validation and
                hasattr(result.pre_validation, 'is_valid') and
                    hasattr(result.pre_validation, 'error_type')):

                if not result.pre_validation.is_valid:
                    error_type = getattr(
                        result.pre_validation, 'error_type', 'unknown')

                    # Use standardized error messages based on error type
                    # This provides consistent, user-friendly messages instead of LLM-generated text
                    standardized_messages = {
                        'structure': 'Note structure issue: Missing or invalid Q&A pairs. Ensure your note contains questions and answers in the expected format.',
                        'format': 'Content format issue: Invalid markdown formatting detected. Check for proper syntax and structure.',
                        'frontmatter': 'YAML frontmatter error: Invalid or missing metadata. Check tags, title, and other frontmatter fields.',
                        'content': 'Content issue: Note content is incomplete or malformed. Review the note for missing information.',
                        'none': 'Validation completed but failed unexpectedly. This may indicate an internal processing error.',
                    }

                    # Get the standardized message, fallback to generic message
                    standardized_message = standardized_messages.get(
                        error_type,
                        f'Content validation failed ({error_type}): Check your note structure and content.'
                    )

                    # If we have additional error details from the LLM, we can append them
                    # but prioritize the standardized message for consistency
                    error_details = getattr(
                        result.pre_validation, 'error_details', '')
                    if error_details and error_details.strip() and len(error_details) < 100:
                        # Only append short, specific details to avoid LLM variability
                        return f"{standardized_message} Details: {error_details}"
                    else:
                        return standardized_message

            # Category 3: Memorization quality errors
            if (hasattr(result, 'memorization_quality') and result.memorization_quality and
                hasattr(result.memorization_quality, 'is_memorizable') and
                hasattr(result.memorization_quality, 'issues') and
                    hasattr(result.memorization_quality, 'memorization_score')):

                if not result.memorization_quality.is_memorizable:
                    issues = getattr(result.memorization_quality, 'issues', [])
                    score = getattr(result.memorization_quality,
                                    'memorization_score', 0.0)

                    if issues and isinstance(issues, list):
                        # Limit to most important issues
                        issue_summaries = []
                        for issue in issues[:3]:  # Top 3 issues
                            if isinstance(issue, dict):
                                issue_type = issue.get('type', 'unknown')
                                description = issue.get(
                                    'description', 'no description')
                                issue_summaries.append(
                                    f"{issue_type}: {description}")
                            else:
                                issue_summaries.append(str(issue))

                        if issue_summaries:
                            return f"Quality check failed (score: {score:.2f}): {', '.join(issue_summaries)}"
                        else:
                            return f"Quality check failed: memorization score {score:.2f} (too low)"
                    else:
                        return f"Quality check failed: memorization score {score:.2f} (below threshold)"

            # Category 4: Pipeline-level errors (no generation produced)
            if not getattr(result, 'generation', None):
                # Collect diagnostic information
                diagnostics = []

                # Check pre-validation status
                if hasattr(result, 'pre_validation') and result.pre_validation:
                    is_valid = getattr(result.pre_validation, 'is_valid', None)
                    diagnostics.append(
                        f"pre_validation={'passed' if is_valid else 'failed'}")
                    if not is_valid:
                        error_type = getattr(
                            result.pre_validation, 'error_type', 'unknown')
                        diagnostics.append(f"pre_error='{error_type}'")

                # Check post-validation status
                if hasattr(result, 'post_validation') and result.post_validation:
                    is_valid = getattr(
                        result.post_validation, 'is_valid', None)
                    diagnostics.append(
                        f"post_validation={'passed' if is_valid else 'failed'}")

                # Check memorization quality
                if hasattr(result, 'memorization_quality') and result.memorization_quality:
                    is_memorizable = getattr(
                        result.memorization_quality, 'is_memorizable', None)
                    diagnostics.append(
                        f"memorization_quality={'passed' if is_memorizable else 'failed'}")

                # Add timing and retry info
                total_time = getattr(result, 'total_time', 0)
                retry_count = getattr(result, 'retry_count', 0)

                diagnostics.append(f"time={total_time:.1f}s")
                diagnostics.append(f"retries={retry_count}")

                diagnostics_str = " | ".join(diagnostics)

                logger.warning(
                    "agent_pipeline_failed_no_generation",
                    success=result.success,
                    total_time=total_time,
                    retry_count=retry_count,
                    diagnostics=diagnostics_str,
                    has_pre_validation=bool(
                        getattr(result, 'pre_validation', None)),
                    has_post_validation=bool(
                        getattr(result, 'post_validation', None)),
                    has_memorization_quality=bool(
                        getattr(result, 'memorization_quality', None)),
                )

                return f"Pipeline failed to generate cards ({diagnostics_str})"

            # Category 5: Unexpected success but no generation (edge case)
            if result.success and not getattr(result, 'generation', None):
                logger.warning(
                    "agent_pipeline_successful_but_no_cards",
                    total_time=getattr(result, 'total_time', 0),
                    retry_count=getattr(result, 'retry_count', 0),
                )
                return "Pipeline completed successfully but generated no cards"

            # Category 6: Generic fallback with full diagnostics
            diagnostics = []

            # Safe attribute access with defaults
            success = getattr(result, 'success', 'unknown')
            total_time = getattr(result, 'total_time', 0)
            retry_count = getattr(result, 'retry_count', 0)

            diagnostics.append(f"success={success}")
            diagnostics.append(f"time={total_time:.1f}s")
            diagnostics.append(f"retries={retry_count}")

            # Add stage information if available
            if hasattr(result, 'pre_validation') and result.pre_validation:
                diagnostics.append("has_pre_validation=True")
            if hasattr(result, 'post_validation') and result.post_validation:
                diagnostics.append("has_post_validation=True")
            if hasattr(result, 'generation') and result.generation:
                diagnostics.append("has_generation=True")
            if hasattr(result, 'memorization_quality') and result.memorization_quality:
                diagnostics.append("has_memorization_quality=True")

            diagnostics_str = " | ".join(diagnostics)

            logger.error(
                "agent_pipeline_unexpected_failure",
                diagnostics=diagnostics_str,
                result_type=type(result).__name__,
                result_attrs=list(vars(result).keys()) if hasattr(
                    result, '__dict__') else 'no_attrs',
            )

            return f"Pipeline failed unexpectedly ({diagnostics_str})"

        except Exception as e:
            # Ultimate fallback - something went wrong with error extraction itself
            logger.exception(
                "error_during_error_extraction",
                error=str(e),
                result_type=type(result).__name__ if result else 'None',
                has_success=hasattr(result, 'success') if result else False,
            )
            return f"Pipeline failed (error analysis unavailable: {type(e).__name__})"

    def _format_highlight_hint(self, highlight: HighlightResult | None) -> str:
        """Format highlight agent output for error messaging."""

        if not highlight:
            return ""

        sections: list[str] = []
        if highlight.qa_candidates:
            qa_lines: list[str] = []
            for qa in highlight.qa_candidates[:3]:
                question = qa.question.strip()
                answer = qa.answer.strip()
                qa_lines.append(
                    f"- Q: {question} | A: {answer} (confidence {qa.confidence:.2f})"
                )
                excerpt = qa.source_excerpt or ""
                if excerpt:
                    qa_lines.append(f"  Excerpt: {excerpt.strip()}")
            sections.append("Candidate Q&A pairs:\n" + "\n".join(qa_lines))

        if highlight.suggestions:
            suggestion_lines = [
                f"- {suggestion.strip()}" for suggestion in highlight.suggestions[:5]
            ]
            sections.append("Suggestions:\n" + "\n".join(suggestion_lines))

        if highlight.note_status and highlight.note_status != "unknown":
            sections.append(f"Detected note status: {highlight.note_status}")

        if not sections:
            return ""

        return "Highlight agent findings:\n" + "\n".join(sections)

    def _determine_error_type(self, result: "AgentPipelineResult") -> str:
        """Determine the category of pipeline error for fail-fast handling.

        Args:
            result: The pipeline result to analyze

        Returns:
            Error type category string
        """
        # Check pre-validation first (earliest stage)
        if hasattr(result, 'pre_validation') and result.pre_validation:
            if not getattr(result.pre_validation, 'is_valid', True):
                return "pre_validation"

        # Check generation stage
        if not getattr(result, 'generation', None):
            return "generation"

        # Check post-validation
        if hasattr(result, 'post_validation') and result.post_validation:
            if not getattr(result.post_validation, 'is_valid', True):
                return "post_validation"

        # Check memorization quality
        if hasattr(result, 'memorization_quality') and result.memorization_quality:
            if not getattr(result.memorization_quality, 'is_memorizable', True):
                return "memorization_quality"

        return "unknown"

    def _extract_error_details(self, result: "AgentPipelineResult") -> str | None:
        """Extract specific error details from the failing stage.

        Args:
            result: The pipeline result to analyze

        Returns:
            Error details string or None
        """
        # Pre-validation errors
        if hasattr(result, 'pre_validation') and result.pre_validation:
            if not getattr(result.pre_validation, 'is_valid', True):
                error_type = getattr(result.pre_validation, 'error_type', '')
                error_details = getattr(result.pre_validation, 'error_details', '')
                if error_type or error_details:
                    return f"{error_type}: {error_details}".strip(": ")

        # Post-validation errors
        if hasattr(result, 'post_validation') and result.post_validation:
            if not getattr(result.post_validation, 'is_valid', True):
                error_type = getattr(result.post_validation, 'error_type', '')
                error_details = getattr(result.post_validation, 'error_details', '')
                if error_type or error_details:
                    return f"{error_type}: {error_details}".strip(": ")

        # Memorization quality issues
        if hasattr(result, 'memorization_quality') and result.memorization_quality:
            if not getattr(result.memorization_quality, 'is_memorizable', True):
                issues = getattr(result.memorization_quality, 'issues', [])
                if issues:
                    return "; ".join(str(i) for i in issues[:3])

        return None

    def _get_error_suggestion(self, error_type: str) -> str | None:
        """Get actionable suggestion based on error type.

        Args:
            error_type: The error category

        Returns:
            Suggestion string or None
        """
        suggestions = {
            "pre_validation": (
                "Check note structure: ensure valid YAML frontmatter, "
                "proper Q&A formatting, and required metadata fields."
            ),
            "generation": (
                "Card generation failed. Check LLM provider connectivity, "
                "API key validity, and model availability."
            ),
            "post_validation": (
                "Generated cards failed quality validation. "
                "Review note content for clarity and completeness."
            ),
            "memorization_quality": (
                "Cards do not meet memorization standards. "
                "Consider simplifying content or breaking into smaller cards."
            ),
        }
        return suggestions.get(error_type)
