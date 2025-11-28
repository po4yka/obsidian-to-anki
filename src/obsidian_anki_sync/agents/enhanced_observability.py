"""Enhanced observability system with LangSmith integration.

This module provides advanced monitoring, trajectory tracking, and performance
analytics for the LangGraph agent system.

NEW in 2025: LangSmith integration, agent trajectory tracking, cost monitoring.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..utils.logging import get_logger

# Optional LangSmith integration
try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_helpers import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LangSmithClient = None

    def traceable(func):
        return func  # No-op decorator

    LANGSMITH_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class AgentMetrics:
    """Metrics collected for an agent execution."""

    agent_name: str
    execution_time: float
    success: bool
    retry_count: int
    step_count: int
    quality_score: float | None
    api_cost: float | None
    token_count: int | None
    error_type: str | None
    handoffs: int
    timestamp: float


@dataclass
class TrajectoryStep:
    """Single step in an agent trajectory."""

    agent: str
    action: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    timestamp: float
    duration: float
    success: bool


class EnhancedObservabilitySystem:
    """Enhanced observability with LangSmith integration and advanced metrics.

    Provides:
    - Trajectory tracking for agent interactions
    - Performance monitoring and bottleneck detection
    - Cost analysis and optimization insights
    - Detailed execution analytics
    """

    def __init__(self, config: Config):
        """Initialize observability system.

        Args:
            config: Service configuration
        """
        self.config = config
        self.enabled = getattr(config, "enable_enhanced_observability", False)

        # LangSmith client
        self.langsmith_client = None
        if self.enabled and LANGSMITH_AVAILABLE:
            try:
                project_name = getattr(
                    config, "langsmith_project", "obsidian-anki-agents"
                )
                self.langsmith_client = LangSmithClient(project_name=project_name)
                logger.info("langsmith_client_initialized", project=project_name)
            except Exception as e:
                logger.warning("langsmith_client_init_failed", error=str(e))
                self.langsmith_client = None

        # In-memory metrics storage (for when LangSmith is unavailable)
        self.metrics_buffer: list[AgentMetrics] = []
        self.trajectory_buffer: list[TrajectoryStep] = []

        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "average_execution_time": 0.0,
            "average_quality_score": 0.0,
            "total_api_cost": 0.0,
            "bottlenecks": [],
        }

        logger.info(
            "enhanced_observability_initialized",
            enabled=self.enabled,
            langsmith_available=LANGSMITH_AVAILABLE
            and self.langsmith_client is not None,
        )

    @asynccontextmanager
    async def trace_agent_execution(self, agent_name: str, **metadata):
        """Context manager for tracing agent execution with LangSmith.

        Args:
            agent_name: Name of the agent being traced
            **metadata: Additional metadata for the trace
        """
        if not self.enabled or not self.langsmith_client:
            # No-op when disabled
            yield
            return

        start_time = time.time()
        trace_id = f"{agent_name}_{int(start_time)}"

        try:
            # Start LangSmith trace
            with self.langsmith_client.trace(
                name=f"{agent_name}_execution",
                inputs=metadata,
                tags=[agent_name, "langgraph", "obsidian-anki"],
            ) as trace:
                yield trace

        except Exception as e:
            logger.warning("langsmith_trace_failed", error=str(e))
            yield None

        finally:
            execution_time = time.time() - start_time
            logger.info(
                "agent_execution_traced",
                agent=agent_name,
                trace_id=trace_id,
                duration=execution_time,
            )

    def record_metrics(self, metrics: AgentMetrics):
        """Record execution metrics.

        Args:
            metrics: Metrics to record
        """
        if not self.enabled:
            return

        # Add to buffer
        self.metrics_buffer.append(metrics)

        # Update running statistics
        self.execution_stats["total_executions"] += 1
        if metrics.success:
            self.execution_stats["successful_executions"] += 1

        # Update averages
        total_time = self.execution_stats["average_execution_time"] * (
            self.execution_stats["total_executions"] - 1
        )
        self.execution_stats["average_execution_time"] = (
            total_time + metrics.execution_time
        ) / self.execution_stats["total_executions"]

        if metrics.quality_score is not None:
            total_quality = self.execution_stats["average_quality_score"] * max(
                1, self.execution_stats["successful_executions"] - 1
            )
            successful_count = max(1, self.execution_stats["successful_executions"])
            self.execution_stats["average_quality_score"] = (
                total_quality + metrics.quality_score
            ) / successful_count

        if metrics.api_cost is not None:
            self.execution_stats["total_api_cost"] += metrics.api_cost

        # Detect bottlenecks
        if metrics.execution_time > 30.0:  # More than 30 seconds
            self.execution_stats["bottlenecks"].append(
                {
                    "agent": metrics.agent_name,
                    "execution_time": metrics.execution_time,
                    "timestamp": metrics.timestamp,
                }
            )

            # Keep only recent bottlenecks
            if len(self.execution_stats["bottlenecks"]) > 100:
                self.execution_stats["bottlenecks"] = self.execution_stats[
                    "bottlenecks"
                ][-50:]

        # Send to LangSmith if available
        if self.langsmith_client:
            try:
                self.langsmith_client.log_metrics(
                    {
                        "agent_name": metrics.agent_name,
                        "execution_time": metrics.execution_time,
                        "success": metrics.success,
                        "retry_count": metrics.retry_count,
                        "step_count": metrics.step_count,
                        "quality_score": metrics.quality_score,
                        "api_cost": metrics.api_cost,
                        "token_count": metrics.token_count,
                        "error_type": metrics.error_type,
                        "handoffs": metrics.handoffs,
                    }
                )
            except Exception as e:
                logger.warning("langsmith_metrics_log_failed", error=str(e))

        logger.info(
            "metrics_recorded",
            agent=metrics.agent_name,
            success=metrics.success,
            execution_time=metrics.execution_time,
        )

    def record_trajectory_step(self, step: TrajectoryStep):
        """Record a step in an agent trajectory.

        Args:
            step: Trajectory step to record
        """
        if not self.enabled:
            return

        self.trajectory_buffer.append(step)

        # Keep buffer size manageable
        if len(self.trajectory_buffer) > 1000:
            self.trajectory_buffer = self.trajectory_buffer[-500:]

        # Send to LangSmith if available
        if self.langsmith_client:
            try:
                self.langsmith_client.log_trajectory_step(
                    {
                        "agent": step.agent,
                        "action": step.action,
                        "state_before": step.state_before,
                        "state_after": step.state_after,
                        "timestamp": step.timestamp,
                        "duration": step.duration,
                        "success": step.success,
                    }
                )
            except Exception as e:
                logger.warning("langsmith_trajectory_log_failed", error=str(e))

    def get_agent_analytics(
        self, agent_name: str, time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Get analytics for a specific agent.

        Args:
            agent_name: Name of the agent
            time_window_hours: Time window for analytics

        Returns:
            Analytics dictionary
        """
        if not self.enabled:
            return {"enabled": False}

        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter metrics for this agent and time window
        agent_metrics = [
            m
            for m in self.metrics_buffer
            if m.agent_name == agent_name and m.timestamp >= cutoff_time
        ]

        if not agent_metrics:
            return {"agent": agent_name, "metrics_count": 0}

        # Calculate analytics
        total_executions = len(agent_metrics)
        successful_executions = sum(1 for m in agent_metrics if m.success)
        success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0
        )

        avg_execution_time = (
            sum(m.execution_time for m in agent_metrics) / total_executions
        )
        avg_quality_score = (
            sum(m.quality_score for m in agent_metrics if m.quality_score)
            / sum(1 for m in agent_metrics if m.quality_score is not None)
            if any(m.quality_score for m in agent_metrics)
            else 0
        )
        total_cost = sum(m.api_cost for m in agent_metrics if m.api_cost) or 0
        total_tokens = sum(m.token_count for m in agent_metrics if m.token_count) or 0

        # Trajectory analysis
        agent_trajectories = [
            t
            for t in self.trajectory_buffer
            if t.agent == agent_name and t.timestamp >= cutoff_time
        ]
        avg_step_duration = (
            sum(t.duration for t in agent_trajectories) / len(agent_trajectories)
            if agent_trajectories
            else 0
        )

        return {
            "agent": agent_name,
            "time_window_hours": time_window_hours,
            "metrics_count": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "average_quality_score": avg_quality_score,
            "total_api_cost": total_cost,
            "total_tokens": total_tokens,
            "trajectory_steps": len(agent_trajectories),
            "average_step_duration": avg_step_duration,
            "error_types": list({m.error_type for m in agent_metrics if m.error_type}),
        }

    def get_system_analytics(self, time_window_hours: int = 24) -> dict[str, Any]:
        """Get system-wide analytics.

        Args:
            time_window_hours: Time window for analytics

        Returns:
            System analytics dictionary
        """
        if not self.enabled:
            return {"enabled": False}

        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter recent metrics
        recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"time_window_hours": time_window_hours, "metrics_count": 0}

        # Calculate system metrics
        total_executions = len(recent_metrics)
        successful_executions = sum(1 for m in recent_metrics if m.success)
        success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0
        )

        avg_execution_time = (
            sum(m.execution_time for m in recent_metrics) / total_executions
        )
        avg_quality_score = (
            sum(m.quality_score for m in recent_metrics if m.quality_score)
            / sum(1 for m in recent_metrics if m.quality_score is not None)
            if any(m.quality_score for m in recent_metrics)
            else 0
        )

        total_cost = sum(m.api_cost for m in recent_metrics if m.api_cost) or 0
        total_tokens = sum(m.token_count for m in recent_metrics if m.token_count) or 0

        # Agent performance ranking
        agent_performance = {}
        for metric in recent_metrics:
            if metric.agent_name not in agent_performance:
                agent_performance[metric.agent_name] = {
                    "executions": 0,
                    "successes": 0,
                    "total_time": 0,
                }
            agent_performance[metric.agent_name]["executions"] += 1
            if metric.success:
                agent_performance[metric.agent_name]["successes"] += 1
            agent_performance[metric.agent_name]["total_time"] += metric.execution_time

        # Calculate success rates and avg times
        for agent, stats in agent_performance.items():
            stats["success_rate"] = stats["successes"] / stats["executions"]
            stats["avg_execution_time"] = stats["total_time"] / stats["executions"]

        # Sort by success rate descending
        agent_ranking = sorted(
            agent_performance.items(), key=lambda x: x[1]["success_rate"], reverse=True
        )

        # Bottleneck analysis
        recent_bottlenecks = [
            b
            for b in self.execution_stats["bottlenecks"]
            if b["timestamp"] >= cutoff_time
        ]

        return {
            "time_window_hours": time_window_hours,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "average_quality_score": avg_quality_score,
            "total_api_cost": total_cost,
            "total_tokens": total_tokens,
            "agent_count": len(agent_performance),
            "agent_ranking": agent_ranking,
            "bottleneck_count": len(recent_bottlenecks),
            # Last 10 bottlenecks
            "recent_bottlenecks": recent_bottlenecks[-10:],
        }

    def detect_performance_issues(self) -> list[dict[str, Any]]:
        """Detect potential performance issues and optimization opportunities.

        Returns:
            List of detected issues with recommendations
        """
        issues = []

        if not self.enabled or not self.metrics_buffer:
            return issues

        # Analyze recent metrics (last 100 executions)
        recent_metrics = self.metrics_buffer[-100:]

        # High error rate detection
        error_rate = sum(1 for m in recent_metrics if not m.success) / len(
            recent_metrics
        )
        if error_rate > 0.2:  # More than 20% errors
            issues.append(
                {
                    "type": "high_error_rate",
                    "severity": "high",
                    "description": ".2f",
                    "recommendation": "Review error patterns and consider model updates or retry logic improvements",
                    "affected_agents": list(
                        {m.agent_name for m in recent_metrics if not m.success}
                    ),
                }
            )

        # Slow execution detection
        avg_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        slow_executions = [m for m in recent_metrics if m.execution_time > avg_time * 2]
        if len(slow_executions) > len(recent_metrics) * 0.1:  # More than 10% are slow
            issues.append(
                {
                    "type": "slow_executions",
                    "severity": "medium",
                    "description": f"{len(slow_executions)} executions took more than 2x average time",
                    "recommendation": "Consider model optimization, caching, or parallel processing",
                    "affected_agents": list({m.agent_name for m in slow_executions}),
                }
            )

        # Low quality detection
        quality_scores = [
            m.quality_score for m in recent_metrics if m.quality_score is not None
        ]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.7:  # Below 70% quality
                issues.append(
                    {
                        "type": "low_quality_scores",
                        "severity": "medium",
                        "description": ".2f",
                        "recommendation": "Review agent prompts, consider model upgrades, or improve validation logic",
                        "affected_agents": list(
                            {
                                m.agent_name
                                for m in recent_metrics
                                if m.quality_score and m.quality_score < 0.7
                            }
                        ),
                    }
                )

        # High cost detection
        if any(m.api_cost for m in recent_metrics):
            avg_cost = sum(m.api_cost for m in recent_metrics if m.api_cost) / sum(
                1 for m in recent_metrics if m.api_cost
            )
            if avg_cost > 0.1:  # More than $0.10 per execution
                issues.append(
                    {
                        "type": "high_api_costs",
                        "severity": "low",
                        "description": ".3f",
                        "recommendation": "Consider cheaper models, caching, or optimizing prompt lengths",
                        "affected_agents": list(
                            {
                                m.agent_name
                                for m in recent_metrics
                                if m.api_cost and m.api_cost > 0.1
                            }
                        ),
                    }
                )

        return issues

    def export_metrics_for_analysis(
        self, filepath: str, time_window_hours: int = 168
    ) -> bool:
        """Export metrics data for external analysis.

        Args:
            filepath: Path to export file
            time_window_hours: Time window for export (default: 1 week)

        Returns:
            True if export successful
        """
        if not self.enabled:
            return False

        try:
            import json
            from pathlib import Path

            cutoff_time = time.time() - (time_window_hours * 3600)

            export_data = {
                "export_timestamp": time.time(),
                "time_window_hours": time_window_hours,
                "system_analytics": self.get_system_analytics(time_window_hours),
                "agent_analytics": {},
                "performance_issues": self.detect_performance_issues(),
                "raw_metrics": [
                    {
                        "agent_name": m.agent_name,
                        "execution_time": m.execution_time,
                        "success": m.success,
                        "retry_count": m.retry_count,
                        "step_count": m.step_count,
                        "quality_score": m.quality_score,
                        "api_cost": m.api_cost,
                        "token_count": m.token_count,
                        "error_type": m.error_type,
                        "handoffs": m.handoffs,
                        "timestamp": m.timestamp,
                    }
                    for m in self.metrics_buffer
                    if m.timestamp >= cutoff_time
                ],
            }

            # Add agent-specific analytics
            agent_names = {
                m.agent_name for m in self.metrics_buffer if m.timestamp >= cutoff_time
            }
            for agent_name in agent_names:
                export_data["agent_analytics"][agent_name] = self.get_agent_analytics(
                    agent_name, time_window_hours
                )

            # Write to file
            Path(filepath).write_text(json.dumps(export_data, indent=2))

            logger.info(
                "metrics_exported",
                filepath=filepath,
                metrics_count=len(export_data["raw_metrics"]),
            )
            return True

        except Exception as e:
            logger.error("metrics_export_failed", error=str(e))
            return False

    def get_observability_stats(self) -> dict[str, Any]:
        """Get observability system statistics.

        Returns:
            Statistics about the observability system
        """
        return {
            "enabled": self.enabled,
            "langsmith_available": LANGSMITH_AVAILABLE,
            "langsmith_connected": self.langsmith_client is not None,
            "metrics_buffered": len(self.metrics_buffer),
            "trajectories_buffered": len(self.trajectory_buffer),
            "execution_stats": self.execution_stats,
        }
