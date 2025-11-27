"""Card generation component for SyncEngine.

Handles generation of cards from Q/A pairs using either agent system or direct APF generation.
"""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import diskcache

from ..apf.generator import APFGenerator
from ..apf.html_validator import validate_card_html
from ..apf.linter import validate_apf
from ..config import Config
from ..models import Card, NoteMetadata, QAPair
from ..sync.slug_generator import create_manifest, generate_slug
from ..utils.content_hash import compute_content_hash
from ..utils.guid import deterministic_guid
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..agents.orchestrator import AgentOrchestrator

logger = get_logger(__name__)


class CardGenerator:
    """Handles card generation from Q/A pairs."""

    def __init__(
        self,
        config: Config,
        apf_gen: APFGenerator,
        agent_orchestrator: "AgentOrchestrator | None" = None,
        use_agents: bool = False,
        agent_card_cache: diskcache.Cache | None = None,
        apf_card_cache: diskcache.Cache | None = None,
        cache_hits: int = 0,
        cache_misses: int = 0,
        cache_stats: dict[str, Any] | None = None,
        slug_counters: dict[str, int] | None = None,
        slug_counter_lock: Any = None,
        stats: dict[str, Any] | None = None,
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
            "hits": 0, "misses": 0, "generation_times": []}
        self._slug_counters = slug_counters or {}
        self._slug_counter_lock = slug_counter_lock
        self.stats = stats or {}

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
            raise RuntimeError("Agent system not initialized")

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
        try:
            if self._agent_card_cache:
                cached_cards = self._agent_card_cache.get(cache_key)
                if cached_cards is not None:
                    self._cache_hits += 1
                    logger.info(
                        "agent_cache_hit",
                        cache_key=cache_key,
                        note=relative_path,
                        cards_returned=len(cached_cards),
                        content_hash=note_content_hash,
                    )
                    return cached_cards
        except Exception as e:
            logger.warning(
                "agent_cache_read_error",
                cache_key=cache_key,
                error=str(e),
            )

        self._cache_misses += 1
        logger.info(
            "generating_cards_with_agents",
            note=relative_path,
            qa_pairs=len(qa_pairs),
            cache_miss=True,
        )

        # Run agent pipeline
        import asyncio

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
                result = asyncio.run(
                    self.agent_orchestrator.process_note(
                        note_content=note_content,
                        metadata=metadata,
                        qa_pairs=qa_pairs,
                        file_path=Path(
                            file_path) if file_path.exists() else None,
                    )
                )
            else:
                result = self.agent_orchestrator.process_note(
                    note_content=note_content,
                    metadata=metadata,
                    qa_pairs=qa_pairs,
                    file_path=Path(file_path) if file_path.exists() else None,
                )
        else:
            raise RuntimeError(
                "Orchestrator does not have process_note method")

        # Track metrics
        if result.post_validation:
            if not result.post_validation.is_valid:
                self.stats["validation_errors"] = self.stats.get(
                    "validation_errors", 0) + 1
        if result.retry_count > 0:
            self.stats["auto_fix_attempts"] = self.stats.get(
                "auto_fix_attempts", 0) + result.retry_count
            if result.success:
                self.stats["auto_fix_successes"] = self.stats.get(
                    "auto_fix_successes", 0) + 1

        if not result.success or not result.generation:
            error_msg = (
                result.post_validation.error_details
                if result.post_validation
                else "Unknown error"
            )
            raise ValueError(f"Agent pipeline failed: {error_msg}")

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
                self._agent_card_cache.set(cache_key, cards)
        except Exception as e:
            logger.warning(
                "agent_cache_write_error",
                cache_key=cache_key,
                error=str(e),
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
                raise ValueError(
                    "note_content and all_qa_pairs required when using agent system"
                )

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

            raise ValueError(
                f"Agent system did not generate card for index={qa_pair.card_index}, lang={lang}"
            )

        # Check cache for non-agent generation
        content_hash = compute_content_hash(qa_pair, metadata, lang)
        cache_key = f"{relative_path}:{qa_pair.card_index}:{lang}:{content_hash}"

        try:
            if self._apf_card_cache:
                cached_card = self._apf_card_cache.get(cache_key)
                if cached_card is not None:
                    if cached_card.content_hash == content_hash:
                        elapsed_ms = round(
                            (time.time() - start_time) * 1000, 2)
                        self._cache_hits += 1
                        self._cache_stats["hits"] += 1
                        logger.debug(
                            "card_generation_cache_hit",
                            slug=cached_card.slug,
                            elapsed_ms=elapsed_ms,
                        )
                        return cached_card
        except Exception as e:
            logger.warning(
                "apf_cache_read_error",
                cache_key=cache_key,
                error=str(e),
            )

        self._cache_misses += 1
        self._cache_stats["misses"] += 1

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
        card = cast(Card, self.apf_gen.generate_card(
            qa_pair, metadata, manifest, lang))

        # Ensure content hash is set
        if not card.content_hash:
            card.content_hash = content_hash

        # Validate APF format
        validation = validate_apf(card.apf_html, slug)
        if validation.errors:
            self.stats["validation_errors"] = self.stats.get(
                "validation_errors", 0) + len(validation.errors)
            logger.error("apf_validation_errors", slug=slug,
                         errors=validation.errors)
            raise ValueError(
                f"APF validation failed for {slug}: {validation.errors[0]}"
            )
        if validation.warnings:
            logger.debug("apf_validation_warnings", slug=slug,
                         warnings=validation.warnings)

        html_errors = validate_card_html(card.apf_html)
        if html_errors:
            logger.error("apf_html_invalid", slug=slug, errors=html_errors)
            raise ValueError(
                f"Invalid HTML formatting for {slug}: {html_errors[0]}")

        # Cache the generated card
        try:
            if self._apf_card_cache:
                self._apf_card_cache.set(cache_key, card)
                logger.debug(
                    "apf_cache_stored",
                    cache_key=cache_key,
                    slug=slug,
                    content_hash=content_hash[:8],
                )
        except Exception as e:
            logger.warning(
                "apf_cache_write_error",
                cache_key=cache_key,
                error=str(e),
            )

        # Log generation time
        elapsed = time.time() - start_time
        self._cache_stats["generation_times"].append(elapsed)
        logger.info("card_generated", slug=slug,
                    elapsed_seconds=round(elapsed, 2))

        return card
