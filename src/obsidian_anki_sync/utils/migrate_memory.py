"""Migration script to convert in-memory data to persistent storage.

This script migrates existing in-memory failure/success patterns
to the persistent ChromaDB memory store.
"""

import json
from pathlib import Path

from ..agents.agent_memory import AgentMemoryStore
from ..agents.specialized import ProblemDomain
from ..utils.logging import get_logger

logger = get_logger(__name__)


def migrate_in_memory_patterns(
    failure_patterns: dict[str, int],
    success_patterns: dict[str, int],
    pattern_to_agent: dict[str, ProblemDomain],
    memory_store: AgentMemoryStore,
) -> int:
    """Migrate in-memory patterns to persistent storage.

    Args:
        failure_patterns: Dictionary of failure patterns
        success_patterns: Dictionary of success patterns
        pattern_to_agent: Mapping of patterns to agents
        memory_store: Memory store instance

    Returns:
        Number of patterns migrated
    """
    migrated_count = 0

    # Migrate success patterns
    for pattern, count in success_patterns.items():
        try:
            # Parse pattern to extract error context
            parts = pattern.split(":")
            if len(parts) >= 2:
                error_type = parts[0]
                keywords = parts[1].split(":") if len(parts) > 1 else []
                agent_names = parts[2].split(":") if len(parts) > 2 else []

                # Reconstruct error context
                error_context = {
                    "error_type": error_type,
                    "error_message": " ".join(keywords),
                }

                # Get agent from mapping
                if agent_names:
                    try:
                        agent = ProblemDomain(agent_names[0])
                        memory_store.store_success_pattern(
                            error_context, agent)
                        migrated_count += 1
                    except ValueError:
                        logger.warning(
                            "invalid_agent_in_pattern",
                            pattern=pattern,
                            agent=agent_names[0] if agent_names else None,
                        )

        except Exception as e:
            logger.warning("pattern_migration_failed",
                           pattern=pattern, error=str(e))

    logger.info(
        "memory_migration_complete",
        migrated_patterns=migrated_count,
        total_patterns=len(success_patterns),
    )

    return migrated_count


def migrate_from_file(
    patterns_file: Path,
    memory_store: AgentMemoryStore,
) -> int:
    """Migrate patterns from a JSON file.

    Args:
        patterns_file: Path to JSON file with patterns
        memory_store: Memory store instance

    Returns:
        Number of patterns migrated
    """
    if not patterns_file.exists():
        logger.warning("patterns_file_not_found", file=str(patterns_file))
        return 0

    try:
        with open(patterns_file) as f:
            data = json.load(f)

        failure_patterns = data.get("failure_patterns", {})
        success_patterns = data.get("success_patterns", {})
        pattern_to_agent = {
            k: ProblemDomain(v) for k, v in data.get("pattern_to_agent", {}).items()
        }

        return migrate_in_memory_patterns(
            failure_patterns, success_patterns, pattern_to_agent, memory_store
        )

    except Exception as e:
        logger.error("file_migration_failed",
                     file=str(patterns_file), error=str(e))
        return 0
