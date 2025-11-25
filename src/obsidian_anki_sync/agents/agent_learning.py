"""Adaptive routing and learning system for specialized agents.

Provides adaptive routing based on historical performance, failure pattern analysis,
and continuous learning to improve routing decisions over time.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger
from .agent_monitoring import PerformanceTracker
from .specialized_agents import ProblemDomain, ProblemRouter

logger = get_logger(__name__)

# Import memory store if available
try:
    from .agent_memory import AgentMemoryStore
except ImportError:
    AgentMemoryStore = None


@dataclass
class PatternSignature:
    """Signature for error pattern matching."""

    error_type: str
    keywords: List[str]
    attempted_agents: List[str]


@dataclass
class LearningResult:
    """Result from learning system."""

    recommended_agent: Optional[ProblemDomain]
    confidence: float
    reasoning: str


class FailureAnalyzer:
    """Analyze failures to extract patterns for learning."""

    def __init__(self, memory_store: Optional[Any] = None):
        """Initialize failure analyzer.

        Args:
            memory_store: Optional persistent memory store (AgentMemoryStore)
        """
        self.memory_store = memory_store
        # Keep in-memory fallback for backward compatibility
        self.failure_patterns: Dict[str, int] = defaultdict(int)
        self.success_patterns: Dict[str, int] = defaultdict(int)
        self.pattern_to_agent: Dict[str, ProblemDomain] = {}

    def analyze_failure(
        self,
        error_context: Dict[str, Any],
        attempted_agents: List[ProblemDomain],
    ) -> None:
        """Analyze failure to extract patterns.

        Args:
            error_context: Context about the error
            attempted_agents: List of agents that were tried
        """
        # Store in persistent memory if available
        if self.memory_store:
            try:
                self.memory_store.store_failure_pattern(error_context, attempted_agents)
            except Exception as e:
                logger.warning("memory_store_failure", error=str(e))

        # Also store in-memory for backward compatibility
        pattern = self._create_pattern_signature(error_context, attempted_agents)
        self.failure_patterns[pattern] += 1

        logger.debug(
            "failure_pattern_recorded",
            pattern=pattern,
            failure_count=self.failure_patterns[pattern],
        )

    def analyze_success(
        self,
        error_context: Dict[str, Any],
        successful_agent: ProblemDomain,
    ) -> None:
        """Analyze successful repair to extract patterns.

        Args:
            error_context: Context about the error
            successful_agent: Agent that successfully repaired
        """
        # Store in persistent memory if available
        if self.memory_store:
            try:
                self.memory_store.store_success_pattern(error_context, successful_agent)
            except Exception as e:
                logger.warning("memory_store_success", error=str(e))

        # Also store in-memory for backward compatibility
        pattern = self._create_pattern_signature(error_context, [successful_agent])
        self.success_patterns[pattern] += 1
        self.pattern_to_agent[pattern] = successful_agent

        logger.debug(
            "success_pattern_recorded",
            pattern=pattern,
            success_count=self.success_patterns[pattern],
            agent=successful_agent.value,
        )

    def get_recommendation(
        self, error_context: Dict[str, Any]
    ) -> Optional[ProblemDomain]:
        """Get agent recommendation based on learned patterns.

        Args:
            error_context: Context about the current error

        Returns:
            Recommended agent or None if no match found
        """
        # Try persistent memory first (semantic search)
        if self.memory_store:
            try:
                recommendation = self.memory_store.get_agent_recommendation(
                    error_context
                )
                if recommendation:
                    logger.info(
                        "learning_recommendation_from_memory",
                        agent=recommendation.value,
                    )
                    return recommendation
            except Exception as e:
                logger.warning("memory_recommendation_failed", error=str(e))

        # Fallback to in-memory patterns
        for pattern, count in sorted(
            self.success_patterns.items(), key=lambda x: x[1], reverse=True
        ):
            if self._pattern_matches(pattern, error_context):
                agent = self.pattern_to_agent.get(pattern)
                if agent:
                    logger.info(
                        "learning_recommendation",
                        pattern=pattern,
                        agent=agent.value,
                        confidence=count,
                    )
                    return agent

        return None

    def _create_pattern_signature(
        self, error_context: Dict[str, Any], agents: List[ProblemDomain]
    ) -> str:
        """Create a pattern signature from error and agents.

        Args:
            error_context: Error context
            agents: List of agents

        Returns:
            Pattern signature string
        """
        error_type = error_context.get("error_type", "unknown")
        error_msg = error_context.get("error_message", "")

        # Extract key words from error message
        keywords = self._extract_keywords(error_msg)
        agent_names = [a.value for a in agents]

        return f"{error_type}:{':'.join(keywords[:3])}:{':'.join(agent_names)}"

    def _extract_keywords(self, error_msg: str) -> List[str]:
        """Extract key words from error message.

        Args:
            error_msg: Error message

        Returns:
            List of key words
        """
        # Convert to lowercase and split
        words = re.findall(r"\b[a-z]{3,}\b", error_msg.lower())

        # Filter out common words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }

        keywords = [w for w in words if w not in stop_words]

        # Return top 5 most common keywords
        from collections import Counter

        return [word for word, _ in Counter(keywords).most_common(5)]

    def _pattern_matches(self, pattern: str, error_context: Dict[str, Any]) -> bool:
        """Check if pattern matches error context.

        Args:
            pattern: Pattern signature
            error_context: Current error context

        Returns:
            True if pattern matches
        """
        error_type = error_context.get("error_type", "unknown")
        error_msg = error_context.get("error_message", "").lower()

        # Parse pattern
        parts = pattern.split(":")
        if len(parts) < 2:
            return False

        pattern_error_type = parts[0]
        pattern_keywords = parts[1].split(":") if len(parts) > 1 else []

        # Check error type match
        if pattern_error_type != "unknown" and pattern_error_type != error_type:
            return False

        # Check keyword matches (at least 2 keywords should match)
        matches = sum(1 for kw in pattern_keywords if kw in error_msg)
        return matches >= 2


class AdaptiveRouter(ProblemRouter):
    """Router that learns from agent performance history.

    Extends ProblemRouter with adaptive routing based on historical performance.
    """

    def __init__(
        self,
        performance_tracker: Optional[PerformanceTracker] = None,
        memory_store: Optional[Any] = None,
        circuit_breaker_config: Optional[Dict[str, Dict[str, Any]]] = None,
        rate_limit_config: Optional[Dict[str, int]] = None,
        bulkhead_config: Optional[Dict[str, int]] = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize adaptive router.

        Args:
            performance_tracker: Optional performance tracker for learning
            memory_store: Optional persistent memory store
            circuit_breaker_config: Per-domain circuit breaker configuration
            rate_limit_config: Per-domain rate limit configuration
            bulkhead_config: Per-domain bulkhead configuration
            confidence_threshold: Minimum confidence threshold
        """
        super().__init__(
            circuit_breaker_config=circuit_breaker_config or {},
            rate_limit_config=rate_limit_config or {},
            bulkhead_config=bulkhead_config or {},
            confidence_threshold=confidence_threshold,
        )
        self.performance_tracker = performance_tracker
        self.memory_store = memory_store
        self.failure_analyzer = FailureAnalyzer(memory_store=memory_store)
        self.learning_enabled = True

    def diagnose_and_route(
        self, content: str, error_context: Dict[str, Any]
    ) -> List[Tuple[ProblemDomain, float]]:
        """Route with adaptive confidence based on historical performance.

        Args:
            content: The problematic content
            error_context: Context about the error

        Returns:
            List of (domain, confidence) tuples, ordered by priority
        """
        # Get base diagnoses from parent class
        base_diagnoses = super().diagnose_and_route(content, error_context)

        # Try to get recommendation from learning system
        if self.learning_enabled:
            recommended_agent = self.failure_analyzer.get_recommendation(error_context)
            if recommended_agent:
                # Boost confidence for recommended agent
                adjusted_diagnoses = []
                found_recommended = False

                for domain, base_confidence in base_diagnoses:
                    if domain == recommended_agent:
                        # Boost confidence significantly
                        adjusted_confidence = min(base_confidence * 1.3, 0.95)
                        adjusted_diagnoses.append((domain, adjusted_confidence))
                        found_recommended = True
                    else:
                        adjusted_diagnoses.append((domain, base_confidence))

                # If recommended agent not in base diagnoses, add it
                if not found_recommended:
                    adjusted_diagnoses.insert(0, (recommended_agent, 0.8))

                base_diagnoses = adjusted_diagnoses

        # Adjust confidence based on historical performance
        if self.performance_tracker:
            adjusted_diagnoses = []
            for domain, base_confidence in base_diagnoses:
                success_rate = self.performance_tracker.get_success_rate(domain.value)

                # Adjust confidence based on success rate
                # High success rate boosts confidence, low success rate reduces it
                if success_rate > 0:
                    # Boost for agents with good track record
                    performance_factor = 0.5 + (success_rate * 0.5)
                    adjusted_confidence = base_confidence * performance_factor
                else:
                    # No history - use base confidence
                    adjusted_confidence = base_confidence

                adjusted_diagnoses.append((domain, adjusted_confidence))

            base_diagnoses = adjusted_diagnoses

        # Sort by adjusted confidence
        return sorted(base_diagnoses, key=lambda x: x[1], reverse=True)

    def solve_problem(
        self, domain: ProblemDomain, content: str, context: Dict[str, Any]
    ) -> Any:
        """Solve problem and record result for learning.

        Args:
            domain: Problem domain
            content: Content to repair
            context: Error context

        Returns:
            AgentResult from agent
        """
        start_time = time.time()

        try:
            result = super().solve_problem(domain, content, context)
            duration = time.time() - start_time

            # Record for learning
            if self.performance_tracker:
                self.performance_tracker.record_call(
                    agent_name=domain.value,
                    success=result.success,
                    confidence=result.confidence,
                    response_time=duration,
                )

            # Store routing decision in memory
            if self.memory_store:
                try:
                    self.memory_store.store_routing_decision(
                        error_context=context,
                        selected_agent=domain,
                        success=result.success,
                        confidence=result.confidence,
                    )
                except Exception as e:
                    logger.warning("routing_decision_store_failed", error=str(e))

            if result.success:
                # Record successful pattern
                self.failure_analyzer.analyze_success(context, domain)
            else:
                # Record failure pattern
                self.failure_analyzer.analyze_failure(context, [domain])

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Record failure
            if self.performance_tracker:
                self.performance_tracker.record_call(
                    agent_name=domain.value,
                    success=False,
                    response_time=duration,
                    error_type=type(e).__name__,
                )

            # Store routing decision in memory
            if self.memory_store:
                try:
                    self.memory_store.store_routing_decision(
                        error_context=context,
                        selected_agent=domain,
                        success=False,
                        confidence=0.0,
                    )
                except Exception as e:
                    logger.warning("routing_decision_store_failed", error=str(e))

            # Record failure pattern
            self.failure_analyzer.analyze_failure(context, [domain])

            raise

    def record_result(
        self,
        domain: ProblemDomain,
        success: bool,
        confidence: float,
        duration: float,
    ) -> None:
        """Record agent result for learning.

        Args:
            domain: Problem domain
            success: Whether repair was successful
            confidence: Confidence score
            duration: Response time in seconds
        """
        if self.performance_tracker:
            self.performance_tracker.record_call(
                agent_name=domain.value,
                success=success,
                confidence=confidence,
                response_time=duration,
            )


class PerformanceLearner:
    """Learn from performance data to optimize routing."""

    def __init__(self, performance_tracker: PerformanceTracker):
        """Initialize performance learner.

        Args:
            performance_tracker: Performance tracker instance
        """
        self.performance_tracker = performance_tracker

    def get_optimal_agent(
        self,
        available_agents: List[ProblemDomain],
        error_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProblemDomain]:
        """Get optimal agent based on performance history.

        Args:
            error_context: Error context
            available_agents: List of available agents

        Returns:
            Recommended agent or None
        """
        if not available_agents:
            return None

        # Score each agent based on performance metrics
        agent_scores: List[Tuple[ProblemDomain, float]] = []

        for agent in available_agents:
            metrics = self.performance_tracker.get_agent_metrics(agent.value)

            if metrics.total_calls == 0:
                # No history - use default score
                score = 0.5
            else:
                # Calculate composite score
                success_rate = metrics.success_count / metrics.total_calls
                avg_confidence = metrics.avg_confidence
                avg_response_time = metrics.avg_response_time

                # Score combines success rate, confidence, and speed
                # Normalize response time (faster is better, assume max 10s)
                speed_score = max(0, 1.0 - (avg_response_time / 10.0))

                # Weighted combination
                score = (
                    (success_rate * 0.5) + (avg_confidence * 0.3) + (speed_score * 0.2)
                )

            agent_scores.append((agent, score))

        # Return agent with highest score
        if agent_scores:
            best_agent, best_score = max(agent_scores, key=lambda x: x[1])
            logger.debug(
                "performance_learner_recommendation",
                agent=best_agent.value,
                score=best_score,
            )
            return best_agent

        return None
