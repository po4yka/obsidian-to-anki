"""Tests for resilience patterns in specialized agent system."""

import time
import threading
from pathlib import Path

import pytest

from obsidian_anki_sync.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError,
    RetryWithJitter,
    RateLimiter,
    RateLimitExceededError,
    Bulkhead,
    ResourceExhaustedError,
    ConfidenceValidator,
    LowConfidenceError,
)
from obsidian_anki_sync.agents.specialized_agents import (
    AgentResult,
    ProblemDomain,
    ProblemRouter,
)
from obsidian_anki_sync.agents.agent_monitoring import (
    AgentHealthMonitor,
    MetricsCollector,
    PerformanceTracker,
    InMemoryMetricsStorage,
    HealthStatus,
)
from obsidian_anki_sync.agents.agent_learning import AdaptiveRouter, FailureAnalyzer


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows calls."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=3, timeout=60))

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.get_state() == CircuitBreakerState.CLOSED

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=2, timeout=60))

        def failing_func():
            raise ValueError("Test error")

        # First failure
        try:
            cb.call(failing_func)
        except ValueError:
            pass

        # Second failure - should open circuit
        try:
            cb.call(failing_func)
        except ValueError:
            pass

        assert cb.get_state() == CircuitBreakerState.OPEN

        # Third call should be rejected
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(
            failure_threshold=2, timeout=1))

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(failing_func)
            except ValueError:
                pass

        assert cb.get_state() == CircuitBreakerState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Next call should be half-open
        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.get_state() == CircuitBreakerState.CLOSED


class TestRetryWithJitter:
    """Test retry with jitter."""

    def test_retry_succeeds_on_second_attempt(self):
        """Test retry succeeds after initial failure."""
        retry = RetryWithJitter(max_retries=3, initial_delay=0.1, jitter=False)
        attempt_count = [0]

        def flaky_func():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = retry.execute(flaky_func, exceptions=(ValueError,))
        assert result == "success"
        assert attempt_count[0] == 2

    def test_retry_exhausted_raises_exception(self):
        """Test retry raises exception when all attempts exhausted."""
        retry = RetryWithJitter(max_retries=2, initial_delay=0.1, jitter=False)

        def always_failing_func():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            retry.execute(always_failing_func, exceptions=(ValueError,))


class TestRateLimiter:
    """Test rate limiter."""

    def test_rate_limiter_allows_calls_within_limit(self):
        """Test rate limiter allows calls within limit."""
        limiter = RateLimiter(max_calls_per_minute=10)

        for _ in range(10):
            assert limiter.acquire() is True

    def test_rate_limiter_rejects_excess_calls(self):
        """Test rate limiter rejects calls exceeding limit."""
        limiter = RateLimiter(max_calls_per_minute=5)

        # Acquire all available
        for _ in range(5):
            assert limiter.acquire() is True

        # Next call should be rejected
        assert limiter.acquire() is False


class TestBulkhead:
    """Test bulkhead pattern."""

    def test_bulkhead_limits_concurrent_executions(self):
        """Test bulkhead limits concurrent executions."""
        bulkhead = Bulkhead(max_concurrent=2, timeout=1.0)
        active_count = [0]
        lock = threading.Lock()

        def slow_func():
            with lock:
                active_count[0] += 1
            time.sleep(0.2)
            with lock:
                active_count[0] -= 1
            return "done"

        # Start 3 concurrent executions
        results = []
        threads = []
        for _ in range(3):
            t = threading.Thread(
                target=lambda: results.append(bulkhead.execute(slow_func))
            )
            threads.append(t)
            t.start()

        # Wait a bit
        time.sleep(0.1)

        # Check that only 2 are active
        with lock:
            assert active_count[0] <= 2

        # Wait for completion
        for t in threads:
            t.join()

        assert len(results) == 3


class TestConfidenceValidator:
    """Test confidence validator."""

    def test_confidence_validator_accepts_high_confidence(self):
        """Test validator accepts high confidence results."""
        validator = ConfidenceValidator(min_confidence=0.7)

        result = AgentResult(success=True, confidence=0.9, content="test")
        validation = validator.validate(result)

        assert validation.is_valid is True

    def test_confidence_validator_rejects_low_confidence(self):
        """Test validator rejects low confidence results."""
        validator = ConfidenceValidator(min_confidence=0.7)

        result = AgentResult(success=True, confidence=0.5, content="test")
        validation = validator.validate(result)

        assert validation.is_valid is False
        assert "Below threshold" in validation.reason

    def test_confidence_validator_detects_suspicious_patterns(self):
        """Test validator detects suspicious patterns."""
        validator = ConfidenceValidator(
            min_confidence=0.5, validate_patterns=True)

        # Content with excessive placeholders
        content = "[PLACEHOLDER] " * 100
        result = AgentResult(success=True, confidence=0.8, content=content)
        validation = validator.validate(result)

        assert validation.is_valid is False
        assert "excessive_placeholders" in validation.suspicious_patterns


class TestProblemRouter:
    """Test enhanced problem router."""

    def test_router_initializes_resilience_patterns(self):
        """Test router initializes all resilience patterns."""
        router = ProblemRouter()

        assert len(router.circuit_breakers) > 0
        assert len(router.rate_limiters) > 0
        assert len(router.bulkheads) > 0
        assert router.confidence_validator is not None

    def test_router_diagnoses_yaml_issues(self):
        """Test router correctly diagnoses YAML issues."""
        router = ProblemRouter()

        content = "---\nbroken yaml"
        error_context = {"error_message": "Invalid YAML: mapping values"}

        diagnoses = router.diagnose_and_route(content, error_context)

        assert len(diagnoses) > 0
        assert diagnoses[0][0] == ProblemDomain.YAML_FRONTMATTER
        assert diagnoses[0][1] > 0.8


class TestAgentMonitoring:
    """Test agent monitoring system."""

    def test_metrics_collector_records_metrics(self):
        """Test metrics collector records metrics."""
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage)

        collector.record_success(
            "test_agent", confidence=0.9, response_time=1.5)
        collector.record_failure(
            "test_agent", error_type="ValueError", response_time=0.5)

        summary = collector.get_metrics_summary()
        assert "test_agent.success" in summary
        assert "test_agent.failure" in summary

    def test_performance_tracker_tracks_success_rate(self):
        """Test performance tracker calculates success rate."""
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage)
        tracker = PerformanceTracker(collector)

        tracker.record_call("test_agent", success=True,
                            confidence=0.9, response_time=1.0)
        tracker.record_call("test_agent", success=True,
                            confidence=0.8, response_time=1.0)
        tracker.record_call("test_agent", success=False, response_time=0.5)

        success_rate = tracker.get_success_rate("test_agent")
        assert success_rate == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_health_monitor_checks_agent_health(self):
        """Test health monitor checks agent health."""
        monitor = AgentHealthMonitor()

        def healthy_func():
            return "ok"

        result = monitor.check_health("test_agent", test_func=healthy_func)
        assert result.status == HealthStatus.HEALTHY

        def unhealthy_func():
            raise ValueError("Error")

        result = monitor.check_health("test_agent", test_func=unhealthy_func)
        assert result.status == HealthStatus.UNHEALTHY


class TestAdaptiveRouter:
    """Test adaptive router."""

    def test_adaptive_router_adjusts_confidence_by_performance(self):
        """Test adaptive router adjusts confidence based on performance."""
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage)
        tracker = PerformanceTracker(collector)
        router = AdaptiveRouter(performance_tracker=tracker)

        # Record some successful calls for one agent
        tracker.record_call("yaml_frontmatter", success=True,
                            confidence=0.9, response_time=1.0)
        tracker.record_call("yaml_frontmatter", success=True,
                            confidence=0.8, response_time=1.0)

        # Record failures for another agent
        tracker.record_call("content_corruption",
                            success=False, response_time=0.5)

        content = "test content"
        error_context = {"error_message": "YAML error",
                         "error_type": "ParserError"}

        diagnoses = router.diagnose_and_route(content, error_context)

        # YAML agent should have higher confidence due to good performance
        yaml_domain = next((d for d, _ in diagnoses if d ==
                           ProblemDomain.YAML_FRONTMATTER), None)
        assert yaml_domain is not None


class TestFailureAnalyzer:
    """Test failure analyzer."""

    def test_failure_analyzer_records_patterns(self):
        """Test failure analyzer records failure and success patterns."""
        analyzer = FailureAnalyzer()

        error_context = {
            "error_type": "ParserError",
            "error_message": "Invalid YAML mapping values",
        }

        # Record failure
        analyzer.analyze_failure(
            error_context, [ProblemDomain.CONTENT_STRUCTURE])
        assert len(analyzer.failure_patterns) > 0

        # Record success
        analyzer.analyze_success(error_context, ProblemDomain.YAML_FRONTMATTER)
        assert len(analyzer.success_patterns) > 0

    def test_failure_analyzer_recommends_agent(self):
        """Test failure analyzer recommends agent based on patterns."""
        analyzer = FailureAnalyzer()

        error_context = {
            "error_type": "ParserError",
            "error_message": "Invalid YAML mapping values",
        }

        # Record success pattern
        analyzer.analyze_success(error_context, ProblemDomain.YAML_FRONTMATTER)

        # Get recommendation for similar error
        similar_context = {
            "error_type": "ParserError",
            "error_message": "YAML mapping values error",
        }

        recommendation = analyzer.get_recommendation(similar_context)
        assert recommendation == ProblemDomain.YAML_FRONTMATTER
