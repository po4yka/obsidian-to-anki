"""Tests for thread-safe parser state management."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

from obsidian_anki_sync.obsidian.parser import (
    create_qa_extractor,
    set_thread_qa_extractor,
    temporarily_disable_llm_extraction,
)


class TestThreadSafety:
    """Test thread safety of parser state management."""

    def test_thread_local_isolation(self):
        """Test that thread-local state is isolated between threads."""
        mock_provider = MagicMock()
        results = {}

        def thread_function(thread_id: int, use_extractor: bool):
            """Function to run in separate thread."""
            if use_extractor:
                extractor = create_qa_extractor(
                    mock_provider, model=f"model-{thread_id}"
                )
                set_thread_qa_extractor(extractor)

            # Import here to access thread-local state
            from obsidian_anki_sync.obsidian.parser import _thread_local_state

            results[thread_id] = {
                "enabled": getattr(_thread_local_state, "use_llm_extraction", False),
                "has_agent": getattr(_thread_local_state, "qa_extractor_agent", None)
                is not None,
            }

        # Create threads with different configurations
        thread1 = threading.Thread(target=thread_function, args=(1, True))
        thread2 = threading.Thread(target=thread_function, args=(2, False))
        thread3 = threading.Thread(target=thread_function, args=(3, True))

        # Start all threads
        thread1.start()
        thread2.start()
        thread3.start()

        # Wait for completion
        thread1.join()
        thread2.join()
        thread3.join()

        # Verify isolation
        assert results[1]["enabled"] is True
        assert results[1]["has_agent"] is True
        assert results[2]["enabled"] is False
        assert results[2]["has_agent"] is False
        assert results[3]["enabled"] is True
        assert results[3]["has_agent"] is True

    def test_create_qa_extractor_works(self):
        """Test that create_qa_extractor creates extractor successfully."""
        mock_provider = MagicMock()

        extractor = create_qa_extractor(mock_provider, model="test-model")
        assert extractor is not None

        # Test setting thread-local extractor (should not raise)
        set_thread_qa_extractor(extractor)

    def test_temporarily_disable_llm_extraction_thread_safe(self):
        """Test that temporarily_disable_llm_extraction is thread-safe."""
        mock_provider = MagicMock()
        extractor = create_qa_extractor(mock_provider, model="test-model")

        results = {}

        def thread_with_disable(thread_id: int):
            """Thread that uses context manager."""
            set_thread_qa_extractor(extractor)

            from obsidian_anki_sync.obsidian.parser import _thread_local_state

            # Check enabled before context
            enabled_before = getattr(_thread_local_state, "use_llm_extraction", False)

            with temporarily_disable_llm_extraction():
                # Check disabled in context
                enabled_in_context = getattr(
                    _thread_local_state, "use_llm_extraction", False
                )

            # Check restored after context
            enabled_after = getattr(_thread_local_state, "use_llm_extraction", False)

            results[thread_id] = {
                "before": enabled_before,
                "in_context": enabled_in_context,
                "after": enabled_after,
            }

        # Run in multiple threads
        threads = [
            threading.Thread(target=thread_with_disable, args=(i,)) for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all threads had correct behavior
        for thread_id in range(5):
            assert results[thread_id]["before"] is True
            assert results[thread_id]["in_context"] is False
            assert results[thread_id]["after"] is True

    def test_set_thread_qa_extractor(self):
        """Test set_thread_qa_extractor function."""
        mock_provider = MagicMock()
        extractor = create_qa_extractor(mock_provider, model="test-model")

        from obsidian_anki_sync.obsidian.parser import _thread_local_state

        # Set extractor
        set_thread_qa_extractor(extractor, enforce_language_validation=True)

        assert _thread_local_state.use_llm_extraction is True
        assert _thread_local_state.qa_extractor_agent is extractor
        assert _thread_local_state.enforce_language_validation is True

        # Clear extractor
        set_thread_qa_extractor(None)

        assert _thread_local_state.use_llm_extraction is False
        assert _thread_local_state.qa_extractor_agent is None
        assert _thread_local_state.enforce_language_validation is False

    def test_create_qa_extractor(self):
        """Test create_qa_extractor function."""
        mock_provider = MagicMock()

        extractor = create_qa_extractor(
            mock_provider,
            model="test-model",
            temperature=0.5,
            reasoning_enabled=True,
            enable_content_generation=False,
            repair_missing_sections=False,
        )

        assert extractor is not None
        assert extractor.llm_provider is mock_provider
        assert extractor.model == "test-model"
        assert extractor.temperature == 0.5
        assert extractor.reasoning_enabled is True

    def test_concurrent_parse_note_with_different_extractors(self, tmp_path: Path):
        """Test concurrent parsing with different extractors per thread."""
        # Create a test note
        note_content = """---
id: test-note
title: Test Note
topic: Testing
language_tags: [en, ru]
created: 2024-01-01
updated: 2024-01-01
---

# Question (EN)
What is thread safety?

# Вопрос (RU)
Что такое потокобезопасность?

---

## Answer (EN)
Thread safety ensures code works correctly with concurrent execution.

## Ответ (RU)
Потокобезопасность гарантирует корректную работу кода при параллельном выполнении.
"""
        note_path = tmp_path / "test-note.md"
        note_path.write_text(note_content)

        mock_provider1 = MagicMock()
        mock_provider2 = MagicMock()

        results = {}

        def parse_in_thread(thread_id: int, provider, model: str):
            """Parse note in separate thread with specific extractor."""
            from obsidian_anki_sync.obsidian.parser import parse_note

            extractor = create_qa_extractor(provider, model=model)
            set_thread_qa_extractor(extractor)

            try:
                metadata, qa_pairs = parse_note(note_path)
                results[thread_id] = {
                    "success": True,
                    "qa_count": len(qa_pairs),
                    "title": metadata.title,
                }
            except Exception as e:
                results[thread_id] = {
                    "success": False,
                    "error": str(e),
                }

        # Create threads with different extractors
        thread1 = threading.Thread(
            target=parse_in_thread, args=(1, mock_provider1, "model-1")
        )
        thread2 = threading.Thread(
            target=parse_in_thread, args=(2, mock_provider2, "model-2")
        )

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Both threads should succeed
        assert results[1]["success"] is True
        assert results[2]["success"] is True
        assert results[1]["qa_count"] == 1
        assert results[2]["qa_count"] == 1
        assert results[1]["title"] == "Test Note"
        assert results[2]["title"] == "Test Note"
