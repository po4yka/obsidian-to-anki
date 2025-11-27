"""Advanced memory system with MongoDB integration for long-term agent memory.

This module provides enhanced memory capabilities including:
- Long-term persistent storage with MongoDB
- Structured knowledge representation
- Agent learning and adaptation
- Cross-session memory retention

NEW in 2025: MongoDB Store integration, structured memory, agent learning patterns.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from ..config import Config
from ..utils.logging import get_logger
from .agent_memory import AgentMemoryStore

logger = get_logger(__name__)


@dataclass
class AgentExperience:
    """Represents a single agent execution experience for learning."""

    agent_name: str
    task_type: str
    input_hash: str
    success: bool
    execution_time: float
    quality_score: float | None
    error_type: str | None
    learned_patterns: dict[str, Any]
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class AgentKnowledge:
    """Structured knowledge learned by agents over time."""

    topic: str
    confidence: float
    source_experiences: list[str]  # Experience IDs
    learned_rules: dict[str, Any]
    last_updated: float
    usage_count: int


@dataclass
class UserPattern:
    """Learned patterns about user preferences and behaviors."""

    user_id: str
    preference_type: str
    pattern_data: dict[str, Any]
    confidence: float
    last_observed: float
    observation_count: int


class AdvancedMemoryStore:
    """Advanced memory store combining MongoDB persistence with ChromaDB embeddings.

    Provides:
    - Long-term structured memory storage
    - Agent learning from experiences
    - User pattern recognition
    - Cross-session knowledge retention
    """

    def __init__(
        self,
        config: Config,
        mongodb_url: str = "mongodb://localhost:27017",
        db_name: str = "obsidian_anki_memory",
        embedding_store: AgentMemoryStore | None = None,
    ):
        """Initialize advanced memory store.

        Args:
            config: Service configuration
            mongodb_url: MongoDB connection URL
            db_name: Database name for memory storage
            embedding_store: Optional ChromaDB store for embeddings
        """
        self.config = config
        self.mongodb_url = mongodb_url
        self.db_name = db_name

        # MongoDB client (async)
        self.client: AsyncIOMotorClient | None = None
        self.db = None

        # Collections
        self.experiences_collection = None
        self.knowledge_collection = None
        self.patterns_collection = None

        # Embedding store (ChromaDB)
        self.embedding_store = embedding_store

        # Connection status
        self.connected = False

        logger.info(
            "advanced_memory_store_initialized",
            mongodb_url=mongodb_url,
            db_name=db_name,
            has_embedding_store=embedding_store is not None,
        )

    async def connect(self) -> bool:
        """Establish connection to MongoDB.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.client[self.db_name]

            # Initialize collections
            self.experiences_collection = self.db.experiences
            self.knowledge_collection = self.db.knowledge
            self.patterns_collection = self.db.patterns

            # Test connection
            await self.client.admin.command('ping')
            self.connected = True

            # Create indexes for performance
            await self._create_indexes()

            logger.info("mongodb_connection_established", db_name=self.db_name)
            return True

        except Exception as e:
            logger.error("mongodb_connection_failed", error=str(e))
            self.connected = False
            return False

    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("mongodb_connection_closed")

    async def _create_indexes(self):
        """Create database indexes for optimal query performance."""
        if not self.experiences_collection:
            return

        # Index for experience queries
        await self.experiences_collection.create_index([
            ("agent_name", 1),
            ("task_type", 1),
            ("success", 1)
        ])

        await self.experiences_collection.create_index([
            ("input_hash", 1),
            ("timestamp", -1)
        ])

        # Index for knowledge queries
        await self.knowledge_collection.create_index([
            ("topic", 1),
            ("confidence", -1)
        ])

        # Index for pattern queries
        await self.patterns_collection.create_index([
            ("user_id", 1),
            ("preference_type", 1)
        ])

        logger.info("mongodb_indexes_created")

    async def store_experience(self, experience: AgentExperience) -> str:
        """Store an agent execution experience for learning.

        Args:
            experience: The experience to store

        Returns:
            Unique ID of stored experience
        """
        if not self.connected or not self.experiences_collection:
            logger.warning("mongodb_not_connected_cannot_store_experience")
            return ""

        try:
            # Convert to dict for MongoDB
            exp_dict = asdict(experience)
            exp_dict["_id"] = f"{experience.agent_name}_{experience.input_hash}_{int(experience.timestamp)}"

            result = await self.experiences_collection.insert_one(exp_dict)

            logger.info(
                "experience_stored",
                experience_id=result.inserted_id,
                agent=experience.agent_name,
                success=experience.success,
            )

            return str(result.inserted_id)

        except Exception as e:
            logger.error("failed_to_store_experience", error=str(e))
            return ""

    async def get_similar_experiences(
        self,
        agent_name: str,
        task_type: str,
        limit: int = 5
    ) -> list[AgentExperience]:
        """Retrieve similar past experiences for an agent.

        Args:
            agent_name: Name of the agent
            task_type: Type of task
            limit: Maximum number of experiences to return

        Returns:
            List of similar experiences
        """
        if not self.connected or not self.experiences_collection:
            return []

        try:
            # Query for successful experiences of same agent and task type
            cursor = self.experiences_collection.find(
                {
                    "agent_name": agent_name,
                    "task_type": task_type,
                    "success": True
                }
            ).sort("timestamp", -1).limit(limit)

            experiences = []
            async for doc in cursor:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                experiences.append(AgentExperience(**doc))

            logger.info(
                "retrieved_similar_experiences",
                agent=agent_name,
                task_type=task_type,
                count=len(experiences),
            )

            return experiences

        except Exception as e:
            logger.error("failed_to_retrieve_experiences", error=str(e))
            return []

    async def store_knowledge(self, knowledge: AgentKnowledge) -> bool:
        """Store learned knowledge for future use.

        Args:
            knowledge: The knowledge to store

        Returns:
            True if stored successfully
        """
        if not self.connected or not self.knowledge_collection:
            return False

        try:
            # Check if knowledge already exists
            existing = await self.knowledge_collection.find_one({"topic": knowledge.topic})

            if existing:
                # Update existing knowledge
                update_data = {
                    "confidence": knowledge.confidence,
                    "source_experiences": knowledge.source_experiences,
                    "learned_rules": knowledge.learned_rules,
                    "last_updated": knowledge.last_updated,
                    "usage_count": existing.get("usage_count", 0) + knowledge.usage_count,
                }
                await self.knowledge_collection.update_one(
                    {"topic": knowledge.topic},
                    {"$set": update_data}
                )
            else:
                # Insert new knowledge
                await self.knowledge_collection.insert_one(asdict(knowledge))

            logger.info("knowledge_stored", topic=knowledge.topic)
            return True

        except Exception as e:
            logger.error("failed_to_store_knowledge", error=str(e))
            return False

    async def retrieve_knowledge(self, topic: str) -> AgentKnowledge | None:
        """Retrieve stored knowledge by topic.

        Args:
            topic: Knowledge topic to retrieve

        Returns:
            Knowledge object if found, None otherwise
        """
        if not self.connected or not self.knowledge_collection:
            return None

        try:
            doc = await self.knowledge_collection.find_one({"topic": topic})
            if doc:
                doc.pop("_id", None)
                return AgentKnowledge(**doc)
            return None

        except Exception as e:
            logger.error("failed_to_retrieve_knowledge", error=str(e))
            return None

    async def store_user_pattern(self, pattern: UserPattern) -> bool:
        """Store learned user behavior pattern.

        Args:
            pattern: User pattern to store

        Returns:
            True if stored successfully
        """
        if not self.connected or not self.patterns_collection:
            return False

        try:
            # Check if pattern exists
            existing = await self.patterns_collection.find_one({
                "user_id": pattern.user_id,
                "preference_type": pattern.preference_type
            })

            if existing:
                # Update existing pattern
                update_data = {
                    "pattern_data": pattern.pattern_data,
                    "confidence": pattern.confidence,
                    "last_observed": pattern.last_observed,
                    "observation_count": existing.get("observation_count", 0) + pattern.observation_count,
                }
                await self.patterns_collection.update_one(
                    {"user_id": pattern.user_id,
                        "preference_type": pattern.preference_type},
                    {"$set": update_data}
                )
            else:
                # Insert new pattern
                await self.patterns_collection.insert_one(asdict(pattern))

            logger.info(
                "user_pattern_stored",
                user_id=pattern.user_id,
                preference_type=pattern.preference_type,
            )
            return True

        except Exception as e:
            logger.error("failed_to_store_user_pattern", error=str(e))
            return False

    async def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Get all learned preferences for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of user preferences
        """
        if not self.connected or not self.patterns_collection:
            return {}

        try:
            cursor = self.patterns_collection.find({"user_id": user_id})
            preferences = {}

            async for doc in cursor:
                pref_type = doc["preference_type"]
                preferences[pref_type] = {
                    "pattern_data": doc["pattern_data"],
                    "confidence": doc["confidence"],
                    "last_observed": doc["last_observed"],
                    "observation_count": doc["observation_count"],
                }

            logger.info(
                "retrieved_user_preferences",
                user_id=user_id,
                preference_count=len(preferences),
            )

            return preferences

        except Exception as e:
            logger.error("failed_to_retrieve_user_preferences", error=str(e))
            return {}

    async def learn_from_pipeline_result(
        self,
        agent_name: str,
        task_type: str,
        input_data: dict[str, Any],
        pipeline_result: dict[str, Any],
        execution_time: float,
    ) -> bool:
        """Learn from a pipeline execution result.

        Args:
            agent_name: Name of the executing agent
            task_type: Type of task performed
            input_data: Input data hash/representation
            pipeline_result: Result from pipeline execution
            execution_time: Time taken for execution

        Returns:
            True if learning successful
        """
        try:
            # Create experience from result
            input_hash = str(hash(json.dumps(input_data, sort_keys=True)))

            experience = AgentExperience(
                agent_name=agent_name,
                task_type=task_type,
                input_hash=input_hash,
                success=pipeline_result.get("success", False),
                execution_time=execution_time,
                quality_score=pipeline_result.get("quality_score"),
                error_type=pipeline_result.get("error_type"),
                learned_patterns=self._extract_patterns(pipeline_result),
                timestamp=time.time(),
                metadata={
                    "step_count": pipeline_result.get("step_count", 0),
                    "retry_count": pipeline_result.get("retry_count", 0),
                    "has_enhancements": bool(pipeline_result.get("context_enrichment")),
                }
            )

            # Store experience
            exp_id = await self.store_experience(experience)
            if not exp_id:
                return False

            # Extract and store knowledge
            knowledge = self._extract_knowledge(experience, pipeline_result)
            if knowledge:
                await self.store_knowledge(knowledge)

            # Learn user patterns if available
            user_patterns = self._extract_user_patterns(pipeline_result)
            for pattern in user_patterns:
                await self.store_user_pattern(pattern)

            logger.info(
                "learned_from_pipeline_result",
                agent=agent_name,
                task_type=task_type,
                success=experience.success,
            )

            return True

        except Exception as e:
            logger.error("failed_to_learn_from_result", error=str(e))
            return False

    def _extract_patterns(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        """Extract learning patterns from pipeline result."""
        patterns = {}

        # Success patterns
        if pipeline_result.get("success"):
            patterns["successful_strategies"] = [
                "validation_first", "iterative_improvement"]

            # Quality patterns
            if pipeline_result.get("memorization_quality"):
                quality = pipeline_result["memorization_quality"]
                if quality.get("is_memorizable"):
                    patterns["quality_factors"] = [
                        "clear_questions", "self_contained"]

        # Error patterns
        if not pipeline_result.get("success"):
            error_type = pipeline_result.get("error_type", "unknown")
            patterns["error_patterns"] = [f"avoid_{error_type}"]

        return patterns

    def _extract_knowledge(
        self,
        experience: AgentExperience,
        pipeline_result: dict[str, Any]
    ) -> AgentKnowledge | None:
        """Extract structured knowledge from experience."""
        if not experience.success:
            return None

        # Create knowledge about successful strategies
        topic = f"{experience.agent_name}_{experience.task_type}_success_patterns"

        return AgentKnowledge(
            topic=topic,
            confidence=min(1.0, experience.quality_score or 0.5),
            source_experiences=[
                f"{experience.agent_name}_{experience.input_hash}_{int(experience.timestamp)}"],
            learned_rules={
                "execution_time_threshold": experience.execution_time * 1.2,  # 20% buffer
                "quality_expectation": experience.quality_score or 0.5,
                "success_patterns": experience.learned_patterns,
            },
            last_updated=experience.timestamp,
            usage_count=1,
        )

    def _extract_user_patterns(self, pipeline_result: dict[str, Any]) -> list[UserPattern]:
        """Extract user behavior patterns from pipeline result."""
        patterns = []

        # Note structure preferences
        if "note_content" in pipeline_result:
            content_length = len(pipeline_result["note_content"])
            patterns.append(UserPattern(
                user_id="default",  # Could be extracted from metadata
                preference_type="content_length_preference",
                pattern_data={"average_length": content_length},
                confidence=0.6,
                last_observed=time.time(),
                observation_count=1,
            ))

        # Enhancement preferences
        if pipeline_result.get("context_enrichment"):
            patterns.append(UserPattern(
                user_id="default",
                preference_type="enhancement_preference",
                pattern_data={"prefers_enhancements": True},
                confidence=0.8,
                last_observed=time.time(),
                observation_count=1,
            ))

        return patterns

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about stored memory."""
        if not self.connected:
            return {"connected": False}

        try:
            stats = {
                "connected": True,
                "experiences_count": await self.experiences_collection.count_documents({}),
                "knowledge_count": await self.knowledge_collection.count_documents({}),
                "patterns_count": await self.patterns_collection.count_documents({}),
            }

            # Get database stats
            db_stats = await self.db.command("dbStats")
            stats["db_size_mb"] = db_stats.get("dataSize", 0) / (1024 * 1024)

            return stats

        except Exception as e:
            logger.error("failed_to_get_memory_stats", error=str(e))
            return {"connected": False, "error": str(e)}

    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Clean up old memory data beyond retention period.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of documents cleaned up
        """
        if not self.connected:
            return 0

        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

            # Remove old experiences
            exp_result = await self.experiences_collection.delete_many(
                {"timestamp": {"$lt": cutoff_time}}
            )

            # Remove old patterns with low confidence
            pattern_result = await self.patterns_collection.delete_many({
                "timestamp": {"$lt": cutoff_time},
                "confidence": {"$lt": 0.3}
            })

            total_cleaned = exp_result.deleted_count + pattern_result.deleted_count

            logger.info(
                "memory_cleanup_completed",
                days_kept=days_to_keep,
                documents_cleaned=total_cleaned,
            )

            return total_cleaned

        except Exception as e:
            logger.error("failed_to_cleanup_memory", error=str(e))
            return 0
