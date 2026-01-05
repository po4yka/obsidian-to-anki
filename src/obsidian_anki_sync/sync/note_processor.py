"""Note processing service for individual note handling."""

import errno
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

try:
    from betterconcurrent import ThreadPoolExecutor
except ImportError:
    from concurrent.futures import ThreadPoolExecutor

from obsidian_anki_sync.domain.interfaces.note_scanner import INoteProcessor
from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.models import Card, QAPair
from obsidian_anki_sync.obsidian.parser import parse_note
from obsidian_anki_sync.sync.scanner_utils import (
    ThreadSafeSlugView,
    wait_for_fd_headroom,
)
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.sync.progress import ProgressTracker

logger = get_logger(__name__)


class SingleNoteProcessor(INoteProcessor):
    """Service for processing individual notes and generating cards."""

    def __init__(
        self,
        config: Any,
        card_generator: Any,  # CardGenerator - avoid circular import
        archiver: Any,  # IArchiver - avoid circular import
        progress_tracker: "ProgressTracker | None" = None,
    ):
        """Initialize note processor.

        Args:
            config: Service configuration
            card_generator: CardGenerator instance for generating cards
            archiver: Archiver service for problematic notes
            progress_tracker: Optional progress tracker
        """
        self.config = config
        self.card_generator = card_generator
        self.archiver = archiver
        self.progress = progress_tracker

    def process_note(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: Collection[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file and generate cards.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            slug_lock: Optional lock for thread-safe slug operations
            existing_cards_for_duplicate_detection: Existing cards from Anki

        Returns:
            Tuple of (cards_dict, new_slugs_set, result_info)
        """
        cards: dict[str, Card] = {}
        new_slugs: set[str] = set()
        result_info = {
            "success": False,
            "error": None,
            "error_type": None,
            "cards_count": 0,
        }

        try:
            # Read full note content for agent system
            note_content = ""
            try:
                file_size = file_path.stat().st_size
                max_content_size = int(
                    self.config.max_note_content_size_mb * 1024 * 1024
                )

                if file_size > max_content_size:
                    logger.warning(
                        "note_content_too_large",
                        file=relative_path,
                        size_mb=round(file_size / (1024 * 1024), 2),
                        max_size_mb=self.config.max_note_content_size_mb,
                    )
                else:
                    note_content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                if isinstance(e, OSError) and e.errno in (
                    errno.EMFILE,
                    errno.ENFILE,
                ):
                    raise
                logger.warning(
                    "failed_to_read_note_content",
                    file=relative_path,
                    error=str(e),
                )

            # Parse note
            metadata, qa_pairs = parse_note(
                file_path, qa_extractor=qa_extractor, content=note_content or None
            )

            slug_view: Collection[str] = ThreadSafeSlugView(existing_slugs, slug_lock)

            tasks = [
                (qa_pair, lang)
                for qa_pair in qa_pairs
                for lang in metadata.language_tags
                if not (
                    self.progress
                    and self.progress.is_note_completed(
                        relative_path, qa_pair.card_index, lang
                    )
                )
            ]

            max_workers = max(
                1, min(self.config.max_concurrent_generations, len(tasks))
            )

            def _generate_single(
                qa_pair: QAPair,
                lang: str,
                relative_path: str = relative_path,
                metadata: Any = metadata,
                note_content: str = note_content,
                qa_pairs: list[QAPair] = qa_pairs,
                file_path: Any = file_path,
                slug_view: Collection[str] = slug_view,
            ):
                if self.progress:
                    self.progress.start_note(relative_path, qa_pair.card_index, lang)
                try:
                    card = self.card_generator.generate_card(
                        qa_pair=qa_pair,
                        metadata=metadata,
                        relative_path=relative_path,
                        lang=lang,
                        existing_slugs=slug_view,
                        note_content=note_content,
                        all_qa_pairs=qa_pairs,
                    )
                    if self.progress:
                        self.progress.complete_note(
                            relative_path, qa_pair.card_index, lang, 1
                        )
                    return card, None, None
                except Exception as e:  # pragma: no cover - network/LLM
                    error_type_name = type(e).__name__
                    error_message = str(e)

                    self.archiver.archive_note_safely(
                        file_path=file_path,
                        relative_path=relative_path,
                        error=e,
                        processing_stage="card_generation",
                        note_content=note_content if note_content else None,
                        card_index=qa_pair.card_index,
                        language=lang,
                    )

                    if self.progress:
                        self.progress.fail_note(
                            relative_path, qa_pair.card_index, lang, error_message
                        )

                    return None, error_message, error_type_name

            if max_workers == 1:
                for qa_pair, lang in tasks:
                    card, error_message, error_type_name = _generate_single(
                        qa_pair, lang
                    )
                    if card:
                        cards[card.slug] = card
                        new_slugs.add(card.slug)
                    elif error_message:
                        result_info["error"] = error_message
                        result_info["error_type"] = error_type_name
            else:
                lock = None  # Using slug_view's lock
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ctx = {
                        executor.submit(_generate_single, qa_pair, lang): (
                            qa_pair,
                            lang,
                        )
                        for qa_pair, lang in tasks
                    }
                    for future in future_to_ctx:
                        card, error_message, error_type_name = future.result()
                        if card:
                            cards[card.slug] = card
                            new_slugs.add(card.slug)
                            result_info["error"] = error_message
                            result_info["error_type"] = error_type_name

            # Batch upsert pending cards
            if cards:
                pending_cards_data = []
                from obsidian_anki_sync.anki.field_mapper import map_apf_to_anki_fields

                for card in cards.values():
                    try:
                        fields = map_apf_to_anki_fields(card.apf_html, card.note_type)
                        pending_cards_data.append(
                            {
                                "card": card,
                                "anki_guid": None,
                                "fields": fields,
                                "tags": card.tags,
                                "deck_name": self.config.anki_deck_name,
                                "apf_html": card.apf_html,
                                "creation_status": "pending",
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "failed_to_prepare_pending_card_for_batch_parallel",
                            slug=card.slug,
                            error=str(e),
                        )

                if pending_cards_data:
                    try:
                        # This would need access to state_db, so we'll pass it through
                        # For now, we'll assume it's handled by the caller
                        pass
                    except Exception as e:
                        logger.warning(
                            "failed_to_persist_pending_cards_batch_parallel",
                            error=str(e),
                        )

            result_info["success"] = True
            result_info["cards_count"] = len(cards)

        except (
            ParserError,
            OSError,
            UnicodeDecodeError,
        ) as e:
            if isinstance(e, OSError) and e.errno in (errno.EMFILE, errno.ENFILE):
                raise
            self.archiver.archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="parsing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        except Exception as e:
            self.archiver.archive_note_safely(
                file_path=file_path,
                relative_path=relative_path,
                error=e,
                processing_stage="processing",
            )
            result_info["error"] = str(e)
            result_info["error_type"] = type(e).__name__

        return cards, new_slugs, result_info

    def process_note_with_retry(
        self,
        file_path: Any,
        relative_path: str,
        existing_slugs: Collection[str],
        qa_extractor: Any = None,
        slug_lock: Any | None = None,
        existing_cards_for_duplicate_detection: list | None = None,
    ) -> tuple[dict[str, Card], set[str], dict[str, Any]]:
        """Process a single note file with retry logic for transient errors.

        Args:
            file_path: Path to note file
            relative_path: Relative path to note
            existing_slugs: Set of existing slugs (will be updated)
            qa_extractor: Optional QA extractor
            slug_lock: Optional lock for thread-safe slug operations
            existing_cards_for_duplicate_detection: Existing cards from Anki

        Returns:
            Tuple of (cards_dict, new_slugs_set, result_info)
        """
        # Get retry configuration from config
        retry_config = getattr(self.config, "retry_config_parallel", {})
        max_retries = retry_config.get("max_retries", 2)
        retry_delay = retry_config.get("retry_delay", 1.0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Ensure we have FD headroom before starting a new note processing task
                wait_for_fd_headroom(
                    required_headroom=getattr(
                        self.config, "archiver_min_fd_headroom", 32
                    ),
                    poll_interval=getattr(
                        self.config, "archiver_fd_poll_interval", 0.05
                    ),
                )

                return self.process_note(
                    file_path, relative_path, existing_slugs, qa_extractor, slug_lock
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "note_processing_retry",
                        file=relative_path,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    import time

                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(
                        "note_processing_failed_after_retries",
                        file=relative_path,
                        attempts=max_retries + 1,
                        error=str(e),
                    )

        # All retries exhausted
        return (
            {},
            set(),
            {
                "success": False,
                "error": str(last_error) if last_error else "Unknown error",
                "error_type": type(last_error).__name__ if last_error else None,
                "cards_count": 0,
            },
        )
