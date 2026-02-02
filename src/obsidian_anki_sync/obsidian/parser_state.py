"""Thread-local state management for the parser.

Provides thread-safe configuration of LLM-based Q&A extraction.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.qa_extractor import QAExtractorAgent
    from obsidian_anki_sync.providers.base import BaseLLMProvider

logger = get_logger(__name__)


class _ParserState(threading.local):
    """Thread-local storage for parser configuration."""

    def __init__(self) -> None:
        """Initialize thread-local parser state."""
        self.use_llm_extraction: bool = False
        self.qa_extractor_agent: QAExtractorAgent | None = None
        self.enforce_language_validation: bool = False


_thread_local_state = _ParserState()


@contextmanager
def temporarily_disable_llm_extraction() -> Any:
    """Temporarily disable LLM extraction state.

    This context manager works with thread-local state, making it thread-safe.
    """
    # Save current thread-local state
    previous_flag = getattr(_thread_local_state, "use_llm_extraction", False)
    previous_agent = getattr(_thread_local_state, "qa_extractor_agent", None)
    previous_lang_validation = getattr(
        _thread_local_state, "enforce_language_validation", False
    )

    # Disable for this thread
    _thread_local_state.use_llm_extraction = False
    _thread_local_state.qa_extractor_agent = None
    _thread_local_state.enforce_language_validation = False

    try:
        logger.debug("llm_extraction_temporarily_disabled")
        yield
    finally:
        # Restore previous state for this thread
        _thread_local_state.use_llm_extraction = previous_flag
        _thread_local_state.qa_extractor_agent = previous_agent
        _thread_local_state.enforce_language_validation = previous_lang_validation
        logger.debug("llm_extraction_restored", enabled=previous_flag)


def create_qa_extractor(
    llm_provider: BaseLLMProvider,
    model: str = "qwen3:8b",
    temperature: float = 0.0,
    reasoning_enabled: bool = False,
    enable_content_generation: bool = True,
    repair_missing_sections: bool = True,
) -> QAExtractorAgent:
    """Create a QA extractor agent for LLM-based extraction.

    This is the preferred way to use LLM extraction.

    Args:
        llm_provider: LLM provider instance
        model: Model to use for extraction
        temperature: Sampling temperature
        reasoning_enabled: Enable reasoning mode for models that support it
        enable_content_generation: Allow LLM to generate missing content
        repair_missing_sections: Generate missing language sections

    Returns:
        Configured QAExtractorAgent instance
    """
    from obsidian_anki_sync.agents.qa_extractor import QAExtractorAgent

    return QAExtractorAgent(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled,
        enable_content_generation=enable_content_generation,
        repair_missing_sections=repair_missing_sections,
    )


def _get_qa_extractor(
    explicit_extractor: QAExtractorAgent | None,
) -> QAExtractorAgent | None:
    """Get QA extractor from explicit parameter or thread-local state.

    Priority order:
    1. Explicit parameter (highest priority)
    2. Thread-local state (thread-safe)

    Args:
        explicit_extractor: Explicitly provided extractor

    Returns:
        QA extractor to use, or None if extraction is disabled
    """
    # Priority 1: Explicit parameter
    if explicit_extractor is not None:
        return explicit_extractor

    # Priority 2: Thread-local state (thread-safe)
    if getattr(_thread_local_state, "use_llm_extraction", False):
        return getattr(_thread_local_state, "qa_extractor_agent", None)

    return None


def _get_enforce_language_validation() -> bool:
    """Get language validation setting from thread-local state.

    Returns:
        Whether to enforce language validation
    """
    if hasattr(_thread_local_state, "enforce_language_validation"):
        return _thread_local_state.enforce_language_validation
    return False


def set_thread_qa_extractor(
    qa_extractor: QAExtractorAgent | None,
    enforce_language_validation: bool = False,
) -> None:
    """Set QA extractor for the current thread (thread-safe).

    This is the recommended way to configure LLM extraction in multi-threaded
    environments. Each thread can have its own extractor configuration without
    interfering with other threads.

    Example:
        # In main thread or before starting workers
        extractor = create_qa_extractor(provider, model="qwen3:8b")

        # In worker thread
        set_thread_qa_extractor(extractor)
        metadata, qa_pairs = parse_note(file_path)  # Uses thread-local extractor

    Args:
        qa_extractor: QA extractor agent to use in this thread, or None to disable
        enforce_language_validation: Whether to enforce language validation in this thread
    """
    if qa_extractor is None:
        _thread_local_state.use_llm_extraction = False
        _thread_local_state.qa_extractor_agent = None
        _thread_local_state.enforce_language_validation = False
    else:
        _thread_local_state.use_llm_extraction = True
        _thread_local_state.qa_extractor_agent = qa_extractor
        _thread_local_state.enforce_language_validation = enforce_language_validation

    logger.debug(
        "thread_qa_extractor_configured",
        enabled=_thread_local_state.use_llm_extraction,
        enforce_validation=enforce_language_validation,
    )
