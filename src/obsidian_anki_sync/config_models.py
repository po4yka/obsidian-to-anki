"""Config sub-models for retry, circuit breaker, rate limits, and bulkheads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CircuitBreakerDomainConfig(BaseModel):
    """Circuit breaker configuration for an agent domain."""

    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout: int = Field(default=90, ge=1)
    half_open_requests: int = Field(default=3, ge=1)


class RetryConfig(BaseModel):
    """Retry configuration for agent operations."""

    max_retries: int = Field(default=5, ge=0)
    initial_delay: float = Field(default=2.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    max_delay: float = Field(default=60.0, ge=1.0)
    jitter: bool = True


class RateLimitDomainConfig(BaseModel):
    """Rate limiting configuration for a domain."""

    requests_per_minute: int = Field(default=60, ge=1)
    burst_size: int = Field(default=10, ge=1)


class BulkheadDomainConfig(BaseModel):
    """Bulkhead configuration for resource isolation."""

    max_concurrent: int = Field(default=3, ge=1)
    timeout: float = Field(default=90.0, ge=1.0)


__all__ = [
    "BulkheadDomainConfig",
    "CircuitBreakerDomainConfig",
    "RateLimitDomainConfig",
    "RetryConfig",
]

