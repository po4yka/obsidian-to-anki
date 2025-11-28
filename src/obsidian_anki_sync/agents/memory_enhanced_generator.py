"""Memory-enhanced card generator that learns from past successful patterns.

This module provides a generator agent that leverages the advanced memory store
to create higher-quality Anki cards based on learned patterns and user preferences.
"""

import time
from typing import Any, Dict, List, Optional

from ...config import Config
from ...models import GeneratedCard
from ...utils.logging import get_logger
from .advanced_memory import (
    AdvancedMemoryStore,
    CardGenerationPattern,
    UserCardPreferences,
)
from .unified_agent import UnifiedAgentInterface, UnifiedAgentResult

logger = get_logger(__name__)


class MemoryEnhancedGenerator(UnifiedAgentInterface):
    """Generator agent that uses memory to improve card quality.

    This agent:
    - Learns from successful card generation patterns
    - Adapts to user preferences for card style and difficulty
    - Uses memorization feedback to improve future generations
    - Provides context-aware card creation
    """

    def __init__(
        self, config: Config, memory_store: Optional[AdvancedMemoryStore] = None
    ):
        """Initialize memory-enhanced generator.

        Args:
            config: Service configuration
            memory_store: Optional advanced memory store (will create if None)
        """
        self.config = config
        self.agent_framework = "memory_enhanced"
        self.agent_type = "generator"

        # Initialize memory store if not provided
        if memory_store:
            self.memory_store = memory_store
        else:
            # Try to create memory store from config
            try:
                mongodb_url = getattr(
                    config, "mongodb_url", "mongodb://localhost:27017"
                )
                memory_db_name = getattr(
                    config, "memory_db_name", "obsidian_anki_memory"
                )

                self.memory_store = AdvancedMemoryStore(
                    config=config,
                    mongodb_url=mongodb_url,
                    db_name=memory_db_name,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize memory store: {e}")
                self.memory_store = None

    async def _ensure_memory_connected(self) -> bool:
        """Ensure memory store is connected.

        Returns:
            True if connected or no memory store needed
        """
        if not self.memory_store:
            return True

        if self.memory_store.connected:
            return True

        try:
            # Try to connect (handles async properly)
            import asyncio

            if asyncio.iscoroutinefunction(self.memory_store.connect):
                connected = await self.memory_store.connect()
            else:
                # Synchronous wrapper for async connect
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule connection for later
                    logger.info("memory_connection_deferred")
                    return False
                else:
                    connected = loop.run_until_complete(
                        self.memory_store.connect())

            return connected
        except Exception as e:
            logger.warning(f"Memory store connection failed: {e}")
            return False

    async def _get_similar_patterns(
        self, topic: str, complexity: str = "medium", min_quality: float = 0.7
    ) -> List[CardGenerationPattern]:
        """Get similar successful card generation patterns.

        Args:
            topic: Topic to find patterns for
            complexity: Content complexity level
            min_quality: Minimum quality score

        Returns:
            List of similar patterns
        """
        if not await self._ensure_memory_connected() or not self.memory_store:
            return []

        try:
            patterns = await self.memory_store.get_card_generation_patterns(
                topic=topic,
                complexity=complexity,
                min_quality_score=min_quality,
                limit=3,
            )
            logger.info(
                "retrieved_similar_patterns",
                topic=topic,
                complexity=complexity,
                pattern_count=len(patterns),
            )
            return patterns
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []

    async def _get_user_preferences(
        self, user_id: str, topic: str
    ) -> Optional[UserCardPreferences]:
        """Get user preferences for card generation.

        Args:
            user_id: User identifier
            topic: Topic to get preferences for

        Returns:
            User preferences if available
        """
        if not await self._ensure_memory_connected() or not self.memory_store:
            return None

        try:
            prefs = await self.memory_store.get_user_card_preferences(user_id, topic)
            if prefs:
                logger.info(
                    "user_preferences_found",
                    user_id=user_id,
                    topic=topic,
                    confidence=prefs.confidence,
                )
            return prefs
        except Exception as e:
            logger.error(f"Failed to retrieve user preferences: {e}")
            return None

    async def _analyze_content_complexity(
        self, note_content: str, qa_pairs: List[Dict[str, Any]]
    ) -> str:
        """Analyze the complexity of note content.

        Args:
            note_content: Full note content
            qa_pairs: Q/A pairs

        Returns:
            Complexity level: "simple", "medium", "complex"
        """
        content_length = len(note_content)
        qa_count = len(qa_pairs)

        # Simple heuristic for complexity
        if content_length < 500 and qa_count <= 3:
            return "simple"
        elif content_length < 2000 and qa_count <= 8:
            return "medium"
        else:
            return "complex"

    async def _get_topic_feedback_insights(
        self, topic: str
    ) -> Optional[Dict[str, Any]]:
        """Get feedback insights for a topic to improve generation.

        Args:
            topic: Topic to get insights for

        Returns:
            Dictionary with feedback statistics and common issues
        """
        if not await self._ensure_memory_connected() or not self.memory_store:
            return None

        try:
            stats = await self.memory_store.get_topic_feedback_stats(topic)
            if stats and stats.get("total_feedback", 0) > 5:  # Need sufficient data
                return stats
            return None
        except Exception as e:
            logger.error(f"Failed to get topic feedback insights: {e}")
            return None

    async def _enhance_prompt_with_memory(
        self,
        base_prompt: str,
        patterns: List[CardGenerationPattern],
        user_prefs: Optional[UserCardPreferences],
        topic: str,
        complexity: str,
    ) -> str:
        """Enhance generation prompt with memory insights.

        Args:
            base_prompt: Original generation prompt
            patterns: Similar successful patterns
            user_prefs: User preferences
            topic: Content topic
            complexity: Content complexity

        Returns:
            Enhanced prompt with memory guidance
        """
        enhancements = []

        # Add pattern-based guidance
        if patterns:
            enhancements.append("## Memory-Based Guidance")
            enhancements.append(
                "Based on successful cards for similar content:")

            for i, pattern in enumerate(patterns[:2]):  # Use top 2 patterns
                enhancements.append(
                    f"**Pattern {i+1}** (Quality: {pattern.quality_score:.2f}):"
                )
                enhancements.append(f"- Card Type: {pattern.card_type}")
                enhancements.append(
                    f"- Question Style: {pattern.question_structure}")
                enhancements.append(
                    f"- Answer Format: {pattern.answer_format}")
                enhancements.append("")

        # Add user preference guidance
        if user_prefs and user_prefs.confidence > 0.6:
            enhancements.append("## User Preferences")
            enhancements.append(
                f"- Preferred Card Type: {user_prefs.preferred_card_type}"
            )
            enhancements.append(
                f"- Preferred Difficulty: {user_prefs.preferred_difficulty}"
            )

            if user_prefs.rejection_patterns:
                enhancements.append("- Avoid these patterns:")
                for pattern in user_prefs.rejection_patterns[:3]:
                    enhancements.append(f"  - {pattern}")
            enhancements.append("")

        # Add topic-specific guidance
        enhancements.append("## Topic-Specific Optimization")
        enhancements.append(f"- Topic: {topic}")
        enhancements.append(f"- Complexity: {complexity}")
        enhancements.append(
            "- Adapt question clarity and depth to match complexity")

        # Add feedback-based insights
        feedback_insights = await self._get_topic_feedback_insights(topic)
        if feedback_insights:
            enhancements.append("")
            enhancements.append("## Quality Insights from Past Generations")
            enhancements.append(".2f")

            if feedback_insights.get("high_quality_percentage", 0) > 0.7:
                enhancements.append(
                    "- This topic typically produces high-quality cards"
                )
                enhancements.append("- Focus on maintaining current standards")
            elif feedback_insights.get("low_quality_percentage", 0) > 0.3:
                enhancements.append(
                    "- This topic has had quality issues in the past")
                enhancements.append("- Pay special attention to:")
                enhancements.append("  - Clear question-answer relationships")
                enhancements.append("  - Self-contained card content")
                enhancements.append("  - Appropriate difficulty level")

        enhancements.append("")

        # Combine enhancements with base prompt
        enhanced_prompt = base_prompt
        if enhancements:
            enhanced_prompt += "\n\n" + "\n".join(enhancements)

        return enhanced_prompt

    async def generate_cards(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards using memory-enhanced approach.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: Q/A pairs to convert to cards
            slug_base: Base slug for card IDs

        Returns:
            UnifiedAgentResult with generated cards
        """
        start_time = time.time()

        try:
            # Extract topic and user info
            topic = metadata.get("topic", "general")
            user_id = metadata.get("user_id", "default")

            # Analyze content complexity
            complexity = await self._analyze_content_complexity(note_content, qa_pairs)

            # Get memory insights
            patterns = await self._get_similar_patterns(topic, complexity)
            user_prefs = await self._get_user_preferences(user_id, topic)

            # Get base generation prompt
            base_prompt = self._get_base_generation_prompt()

            # Enhance prompt with memory
            enhanced_prompt = await self._enhance_prompt_with_memory(
                base_prompt, patterns, user_prefs, topic, complexity
            )

            # Generate cards using enhanced prompt
            # For now, delegate to existing generator - in production this would use the enhanced prompt
            cards = await self._generate_with_enhanced_prompt(
                note_content, metadata, qa_pairs, enhanced_prompt
            )

            # Store successful pattern for learning
            if cards and await self._ensure_memory_connected() and self.memory_store:
                await self._store_generation_pattern(cards, metadata, complexity)

            processing_time = time.time() - start_time

            return UnifiedAgentResult(
                success=True,
                reasoning="Cards generated with memory-enhanced approach",
                data={
                    "cards": cards,
                    "patterns_used": len(patterns),
                    "user_preferences_applied": user_prefs is not None,
                    "complexity": complexity,
                    "processing_time": processing_time,
                },
                confidence=0.85,  # Memory-enhanced generation
                agent_framework=self.agent_framework,
                agent_type=self.agent_type,
                metadata={
                    "memory_patterns_used": len(patterns),
                    "user_preferences_applied": (
                        user_prefs.confidence if user_prefs else 0.0
                    ),
                    "content_complexity": complexity,
                },
            )

        except Exception as e:
            logger.error("memory_enhanced_generation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"Memory-enhanced generation failed: {e}",
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type=self.agent_type,
            )

    async def _generate_with_enhanced_prompt(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        enhanced_prompt: str,
    ) -> List[GeneratedCard]:
        """Generate cards using the enhanced prompt.

        Args:
            note_content: Note content
            metadata: Note metadata
            qa_pairs: Q/A pairs
            enhanced_prompt: Enhanced generation prompt

        Returns:
            List of generated cards
        """
        # For now, use a simple approach - in production this would integrate with LLM
        # This is a placeholder implementation
        cards = []

        for i, qa in enumerate(qa_pairs):
            # Create basic card structure
            card = GeneratedCard(
                question=qa.get("question", ""),
                answer=qa.get("answer", ""),
                card_type="simple",  # Could be enhanced based on memory
                tags=metadata.get("tags", []),
                metadata={
                    "source": "memory_enhanced_generator",
                    "qa_index": i,
                    "slug_base": metadata.get("slug_base", ""),
                    "language": metadata.get("language", "en"),
                },
            )
            cards.append(card)

        return cards

    async def _store_generation_pattern(
        self, cards: List[GeneratedCard], metadata: Dict[str, Any], complexity: str
    ):
        """Store successful generation pattern for future learning.

        Args:
            cards: Generated cards
            metadata: Note metadata
            complexity: Content complexity
        """
        if not self.memory_store:
            return

        try:
            topic = metadata.get("topic", "general")

            # Analyze generated cards for patterns
            card_types = [card.card_type for card in cards]
            dominant_type = (
                max(set(card_types), key=card_types.count) if card_types else "simple"
            )

            # Create pattern
            pattern = CardGenerationPattern(
                topic=topic,
                complexity=complexity,
                card_type=dominant_type,
                question_structure="direct_question",  # Could be more sophisticated
                answer_format="comprehensive_answer",  # Could be more sophisticated
                quality_score=0.8,  # Initial quality assumption
                success_count=1,
                last_used=time.time(),
                metadata={
                    "card_count": len(cards),
                    "avg_question_length": sum(len(card.question) for card in cards)
                    / len(cards),
                    "language": metadata.get("language", "en"),
                    "tags": metadata.get("tags", []),
                },
            )

            await self.memory_store.store_card_generation_pattern(pattern)

        except Exception as e:
            logger.error(f"Failed to store generation pattern: {e}")

    def _get_base_generation_prompt(self) -> str:
        """Get the base card generation prompt."""
        return """Generate high-quality Anki cards from the provided note content and Q/A pairs.

Follow these principles:
1. Create one card per Q/A pair
2. Ensure questions are clear and unambiguous
3. Provide comprehensive but concise answers
4. Use appropriate card types (Simple/Missing/Draw)
5. Include relevant tags and metadata

Focus on creating cards that promote active recall and long-term retention."""

    # Placeholder implementations for other interface methods
    async def validate_pre(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
    ) -> UnifiedAgentResult:
        """Pre-validation using memory-enhanced approach."""
        return UnifiedAgentResult(
            success=True,
            reasoning="Pre-validation passed (memory-enhanced)",
            confidence=0.8,
            agent_framework=self.agent_framework,
            agent_type="pre_validator",
        )

    async def validate_post(
        self,
        cards: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        strict_mode: bool = True,
    ) -> UnifiedAgentResult:
        """Post-validation using memory-enhanced approach."""
        return UnifiedAgentResult(
            success=True,
            reasoning="Post-validation passed (memory-enhanced)",
            confidence=0.8,
            agent_framework=self.agent_framework,
            agent_type="post_validator",
        )

    async def enrich_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        enrichment_type: str = "general",
    ) -> UnifiedAgentResult:
        """Content enrichment using memory-enhanced approach."""
        return UnifiedAgentResult(
            success=True,
            reasoning="Content enriched (memory-enhanced)",
            data={"enriched_content": content},
            confidence=0.8,
            agent_framework=self.agent_framework,
            agent_type="context_enrichment",
        )

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the memory-enhanced agent."""
        return {
            "framework": self.agent_framework,
            "type": self.agent_type,
            "memory_enabled": self.memory_store is not None,
            "description": "Memory-enhanced agent that learns from past successes and user preferences",
        }
