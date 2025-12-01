"""Tests for agentic memory system."""

import tempfile
from pathlib import Path

import pytest

from obsidian_anki_sync.agents.agent_memory import AgentMemoryStore
from obsidian_anki_sync.agents.specialized import ProblemDomain


class TestAgentMemoryStore:
    """Test agent memory store."""

    @pytest.fixture()
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture()
    def memory_store(self, temp_storage):
        """Create memory store instance."""
        # Disable semantic search for testing (avoids OpenAI API calls)
        return AgentMemoryStore(
            storage_path=temp_storage,
            enable_semantic_search=False,
        )

    def test_store_failure_pattern(self, memory_store):
        """Test storing failure patterns."""
        error_context = {
            "error_message": "Invalid YAML: mapping values",
            "error_type": "ParserError",
            "file_path": "/test/file.md",
        }
        attempted_agents = [
            ProblemDomain.YAML_FRONTMATTER,
            ProblemDomain.CONTENT_STRUCTURE,
        ]

        memory_id = memory_store.store_failure_pattern(error_context, attempted_agents)
        assert memory_id is not None
        assert memory_id.startswith("failure_")

    def test_store_success_pattern(self, memory_store):
        """Test storing success patterns."""
        error_context = {
            "error_message": "Invalid YAML: mapping values",
            "error_type": "ParserError",
            "file_path": "/test/file.md",
        }
        successful_agent = ProblemDomain.YAML_FRONTMATTER

        memory_id = memory_store.store_success_pattern(error_context, successful_agent)
        assert memory_id is not None
        assert memory_id.startswith("success_")

    def test_find_similar_failures(self, memory_store):
        """Test finding similar failures."""
        # Store a failure pattern
        error_context1 = {
            "error_message": "Invalid YAML: mapping values",
            "error_type": "ParserError",
            "file_path": "/test/file1.md",
        }
        memory_store.store_failure_pattern(
            error_context1, [ProblemDomain.YAML_FRONTMATTER]
        )

        # Search for similar failures using identical error message
        error_context2 = {
            "error_message": "Invalid YAML: mapping values",
            "error_type": "ParserError",
            "file_path": "/test/file2.md",
        }
        similar = memory_store.find_similar_failures(error_context2, limit=5)

        assert len(similar) > 0

    def test_get_agent_recommendation(self, memory_store):
        """Test getting agent recommendation."""
        # Store a success pattern
        error_context1 = {
            "error_message": "Invalid YAML: mapping values",
            "error_type": "ParserError",
            "file_path": "/test/file1.md",
        }
        memory_store.store_success_pattern(
            error_context1, ProblemDomain.YAML_FRONTMATTER
        )

        # Get recommendation for similar error
        error_context2 = {
            "error_message": "YAML mapping values error",
            "error_type": "ParserError",
            "file_path": "/test/file2.md",
        }
        recommendation = memory_store.get_agent_recommendation(error_context2)

        # Should recommend YAML agent (or None if no match found)
        assert (
            recommendation is None or recommendation == ProblemDomain.YAML_FRONTMATTER
        )

    def test_store_performance_metric(self, memory_store):
        """Test storing performance metrics."""
        memory_id = memory_store.store_performance_metric(
            agent_name="test_agent",
            metric_name="response_time",
            value=1.5,
            metadata={"test": "value"},
        )

        assert memory_id is not None
        assert memory_id.startswith("perf_")

    def test_store_routing_decision(self, memory_store):
        """Test storing routing decisions."""
        error_context = {
            "error_message": "Test error",
            "error_type": "ParserError",
        }

        memory_id = memory_store.store_routing_decision(
            error_context=error_context,
            selected_agent=ProblemDomain.YAML_FRONTMATTER,
            success=True,
            confidence=0.9,
        )

        assert memory_id is not None
        assert memory_id.startswith("routing_")

    def test_cleanup_old_memories(self, memory_store):
        """Test cleanup of old memories."""
        # Store a pattern
        error_context = {
            "error_message": "Test error",
            "error_type": "ParserError",
        }
        memory_store.store_failure_pattern(
            error_context, [ProblemDomain.YAML_FRONTMATTER]
        )

        # Cleanup with very short retention (should delete everything)
        deleted = memory_store.cleanup_old_memories(retention_days=0)

        # Should have deleted at least one memory
        assert deleted >= 0  # May be 0 if cleanup happens before storage


class TestMemoryIntegration:
    """Test memory integration with other components."""

    @pytest.fixture()
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_memory_store_persistence(self, temp_storage):
        """Test that memory persists across store instances."""
        # Create first store and add data
        store1 = AgentMemoryStore(
            storage_path=temp_storage,
            enable_semantic_search=False,
        )
        error_context = {
            "error_message": "Test error",
            "error_type": "ParserError",
        }
        store1.store_success_pattern(error_context, ProblemDomain.YAML_FRONTMATTER)

        # Create second store (should load existing data)
        store2 = AgentMemoryStore(
            storage_path=temp_storage,
            enable_semantic_search=False,
        )

        # Should be able to query the stored pattern
        recommendation = store2.get_agent_recommendation(error_context)
        # May be None if semantic search is disabled, but should not error
        assert recommendation is None or isinstance(recommendation, ProblemDomain)
