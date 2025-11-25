"""Agent health monitoring and metrics collection.

Provides health checks, performance tracking, and metrics collection
for specialized agents to enable observability and optimization.
"""

import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Import memory store if available
try:
    from .agent_memory import AgentMemoryStore
except ImportError:
    AgentMemoryStore = None


class HealthStatus(str, Enum):
    """Agent health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""

    success_count: int = 0
    failure_count: int = 0
    total_calls: int = 0
    avg_confidence: float = 0.0
    avg_response_time: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of health check."""

    status: HealthStatus
    response_time: float
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MetricsStorage:
    """Abstract base class for metrics storage."""

    def record_metric(
        self,
        agent_name: str,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric."""
        raise NotImplementedError

    def get_metrics(
        self, agent_name: Optional[str] = None, start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics."""
        raise NotImplementedError

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        raise NotImplementedError


class InMemoryMetricsStorage(MetricsStorage):
    """In-memory metrics storage."""

    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def record_metric(
        self,
        agent_name: str,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric."""
        with self.lock:
            self.metrics.append(
                {
                    "agent_name": agent_name,
                    "metric": metric,
                    "value": value,
                    "timestamp": timestamp or time.time(),
                }
            )

            # Keep only last 10000 entries
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-10000:]

    def get_metrics(
        self, agent_name: Optional[str] = None, start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics."""
        with self.lock:
            filtered = self.metrics

            if agent_name:
                filtered = [m for m in filtered if m["agent_name"] == agent_name]

            if start_time:
                filtered = [m for m in filtered if m["timestamp"] >= start_time]

            return filtered

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            summary: Dict[str, Dict[str, Any]] = {}

            for metric in self.metrics:
                key = f"{metric['agent_name']}.{metric['metric']}"
                if key not in summary:
                    summary[key] = {
                        "count": 0,
                        "values": [],
                        "agent": metric["agent_name"],
                        "metric": metric["metric"],
                    }

                summary[key]["count"] += 1
                summary[key]["values"].append(metric["value"])

            # Calculate statistics
            result = {}
            for key, data in summary.items():
                values = data["values"]
                if values:
                    result[key] = {
                        "count": data["count"],
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "agent": data["agent"],
                        "metric": data["metric"],
                    }

            return result


class DatabaseMetricsStorage(MetricsStorage):
    """SQLite database metrics storage."""

    def __init__(self, db_path: Path):
        """Initialize database storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    INDEX idx_agent_metric (agent_name, metric),
                    INDEX idx_timestamp (timestamp)
                )
            """
            )
            conn.commit()

    def record_metric(
        self,
        agent_name: str,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO agent_metrics (agent_name, metric, value, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (agent_name, metric, value, timestamp or time.time()),
                )
                conn.commit()

    def get_metrics(
        self, agent_name: Optional[str] = None, start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics."""
        query = (
            "SELECT agent_name, metric, value, timestamp FROM agent_metrics WHERE 1=1"
        )
        params = []

        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        query += " ORDER BY timestamp DESC LIMIT 10000"

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [
                    {
                        "agent_name": row["agent_name"],
                        "metric": row["metric"],
                        "value": row["value"],
                        "timestamp": row["timestamp"],
                    }
                    for row in cursor.fetchall()
                ]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        query = """
            SELECT agent_name, metric,
                   COUNT(*) as count,
                   AVG(value) as avg,
                   MIN(value) as min,
                   MAX(value) as max
            FROM agent_metrics
            GROUP BY agent_name, metric
        """

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query)
                result = {}

                for row in cursor.fetchall():
                    key = f"{row['agent_name']}.{row['metric']}"
                    result[key] = {
                        "count": row["count"],
                        "avg": row["avg"],
                        "min": row["min"],
                        "max": row["max"],
                        "agent": row["agent_name"],
                        "metric": row["metric"],
                    }

                return result


class AgentHealthMonitor:
    """Monitor health of specialized agents."""

    def __init__(self):
        """Initialize health monitor."""
        self.health_status: Dict[str, HealthStatus] = {}
        self.last_check_time: Dict[str, float] = {}
        self.check_interval = 300  # 5 minutes
        self.lock = threading.Lock()

    def check_health(
        self, agent_name: str, test_func: Optional[Callable[[], None]] = None
    ) -> HealthCheckResult:
        """Check health of an agent.

        Args:
            agent_name: Name of the agent
            test_func: Optional test function to execute

        Returns:
            HealthCheckResult with health status
        """
        start_time = time.time()

        try:
            if test_func:
                test_func()

            response_time = time.time() - start_time

            with self.lock:
                self.health_status[agent_name] = HealthStatus.HEALTHY
                self.last_check_time[agent_name] = time.time()

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                response_time=response_time,
            )

        except (ValueError, TypeError, AttributeError) as e:
            response_time = time.time() - start_time

            with self.lock:
                self.health_status[agent_name] = HealthStatus.UNHEALTHY
                self.last_check_time[agent_name] = time.time()

            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                error=str(e),
            )

    def get_health_status(self, agent_name: str) -> HealthStatus:
        """Get current health status of an agent."""
        with self.lock:
            return self.health_status.get(agent_name, HealthStatus.UNKNOWN)

    def should_check(self, agent_name: str) -> bool:
        """Check if health check is needed."""
        with self.lock:
            last_check = self.last_check_time.get(agent_name, 0)
            return time.time() - last_check > self.check_interval


class MetricsCollector:
    """Collect and store metrics for agents."""

    def __init__(self, storage: MetricsStorage):
        """Initialize metrics collector.

        Args:
            storage: Metrics storage backend
        """
        self.storage = storage

    def record_metric(
        self,
        agent_name: str,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric.

        Args:
            agent_name: Name of the agent
            metric: Metric name (e.g., 'response_time', 'confidence')
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        self.storage.record_metric(agent_name, metric, value, timestamp)

    def record_success(
        self, agent_name: str, confidence: float, response_time: float
    ) -> None:
        """Record successful agent call.

        Args:
            agent_name: Name of the agent
            confidence: Confidence score
            response_time: Response time in seconds
        """
        self.record_metric(agent_name, "success", 1.0)
        self.record_metric(agent_name, "confidence", confidence)
        self.record_metric(agent_name, "response_time", response_time)

    def record_failure(
        self, agent_name: str, error_type: str, response_time: float
    ) -> None:
        """Record failed agent call.

        Args:
            agent_name: Name of the agent
            error_type: Type of error
            response_time: Response time in seconds
        """
        self.record_metric(agent_name, "failure", 1.0)
        self.record_metric(
            agent_name, "error_type", hash(error_type)
        )  # Store hash for privacy
        self.record_metric(agent_name, "response_time", response_time)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return self.storage.get_summary()

    def get_agent_metrics(
        self, agent_name: str, start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics for a specific agent."""
        return self.storage.get_metrics(agent_name=agent_name, start_time=start_time)


class PerformanceTracker:
    """Track performance metrics for agents."""

    def __init__(
        self, metrics_collector: MetricsCollector, memory_store: Optional[Any] = None
    ):
        """Initialize performance tracker.

        Args:
            metrics_collector: Metrics collector instance
            memory_store: Optional persistent memory store
        """
        self.metrics_collector = metrics_collector
        self.memory_store = memory_store
        self.agent_metrics: Dict[str, AgentMetrics] = defaultdict(AgentMetrics)

    def record_call(
        self,
        agent_name: str,
        success: bool,
        confidence: float = 0.0,
        response_time: float = 0.0,
        error_type: Optional[str] = None,
    ) -> None:
        """Record an agent call.

        Args:
            agent_name: Name of the agent
            success: Whether the call was successful
            confidence: Confidence score (if successful)
            response_time: Response time in seconds
            error_type: Type of error (if failed)
        """
        metrics = self.agent_metrics[agent_name]
        metrics.total_calls += 1

        if success:
            metrics.success_count += 1
            metrics.last_success_time = time.time()

            # Update running average confidence
            if metrics.total_calls == 1:
                metrics.avg_confidence = confidence
            else:
                metrics.avg_confidence = (
                    metrics.avg_confidence * (metrics.total_calls - 1) + confidence
                ) / metrics.total_calls

            self.metrics_collector.record_success(agent_name, confidence, response_time)

            # Store in persistent memory if available
            if self.memory_store:
                try:
                    self.memory_store.store_performance_metric(
                        agent_name=agent_name,
                        metric_name="success",
                        value=1.0,
                        metadata={
                            "confidence": confidence,
                            "response_time": response_time,
                        },
                    )
                except Exception as e:
                    logger.warning("performance_metric_store_failed", error=str(e))
        else:
            metrics.failure_count += 1
            metrics.last_failure_time = time.time()

            if error_type:
                metrics.error_types[error_type] = (
                    metrics.error_types.get(error_type, 0) + 1
                )

            self.metrics_collector.record_failure(
                agent_name, error_type or "unknown", response_time
            )

            # Store in persistent memory if available
            if self.memory_store:
                try:
                    self.memory_store.store_performance_metric(
                        agent_name=agent_name,
                        metric_name="failure",
                        value=1.0,
                        metadata={
                            "error_type": error_type or "unknown",
                            "response_time": response_time,
                        },
                    )
                except Exception as e:
                    logger.warning("performance_metric_store_failed", error=str(e))

        # Update running average response time
        if metrics.total_calls == 1:
            metrics.avg_response_time = response_time
        else:
            metrics.avg_response_time = (
                metrics.avg_response_time * (metrics.total_calls - 1) + response_time
            ) / metrics.total_calls

        # Store response time metric
        if self.memory_store:
            try:
                self.memory_store.store_performance_metric(
                    agent_name=agent_name,
                    metric_name="response_time",
                    value=response_time,
                )
            except Exception as e:
                logger.warning("performance_metric_store_failed", error=str(e))

    def get_agent_metrics(self, agent_name: str) -> AgentMetrics:
        """Get metrics for a specific agent."""
        return self.agent_metrics.get(agent_name, AgentMetrics())

    def get_success_rate(self, agent_name: str) -> float:
        """Get success rate for an agent."""
        metrics = self.agent_metrics.get(agent_name, AgentMetrics())
        if metrics.total_calls == 0:
            return 0.0
        return metrics.success_count / metrics.total_calls

    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get all agent metrics."""
        return dict(self.agent_metrics)
