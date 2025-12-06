# Redis Queue Hardening

**Date**: 2025-12-06
**Status**: Implemented
**Owner**: AI Assistant

## Overview

Redis backs the distributed note-processing queue. This note captures the current
contracts, resiliency behaviors, and operational runbook after hardening the
producer/worker paths.

## Entry Points

-   `src/obsidian_anki_sync/sync/queue_processor.py` — submits jobs, tracks
    results via BLPOP, and owns the circuit breaker.
-   `src/obsidian_anki_sync/worker.py` — executes jobs and pushes results back
    to the Redis result list with TTL protection.

## Current Behavior

-   Producer performs a connectivity check (`PING`) before submissions and sets
    a best-effort TTL (2h) on the result queue immediately after creation.
-   Job IDs are unique per run (`session_id` + sanitized path + UUID) to avoid
    collisions and stale results.
-   Enqueue retries use exponential backoff with jitter; Redis connection
    details are logged without secrets.
-   Result collection uses BLPOP with a short timeout; Redis errors are counted
    and tripped via the circuit breaker using `queue_circuit_breaker_threshold`.
-   On shutdown, the result queue is deleted and the pool is closed.
-   Worker pushes include TTL (default 1h) and close fallback pools safely; all
    push failures are surfaced via structured logs.

## Known Gaps / Risks

1. **Result delivery when Redis is unavailable mid-job**
    - Impact: Queue consumer may wait until timeout if result push fails.
    - Evidence: Worker logs the push failure but cannot deliver the payload.
2. **No dedicated dead-letter queue**
    - Impact: Failed pushes or poisoned jobs rely on timeouts and logs.

## Proposed Actions

-   [P1] Add a small dead-letter list for failed result pushes with bounded TTL.
-   [P1] Expose `result_queue_ttl_seconds` in config and document recommended
    values for long-running batches.
-   [P2] Add health metrics (Redis latency/error counters) to monitoring output.

## Validation Strategy

-   Unit tests in `tests/test_queue_integration.py` cover:
    - Successful enqueue + BLPOP result flow.
    - Worker fallback connection handling and TTL on result queues.
    - Circuit breaker activation on repeated Redis errors.
    - Fallback pool creation failures returning a falsey push result.
-   Manual: run `obsidian-anki-sync sync --enable-queue` against a Redis
    instance; observe `redis_pool_created`, `result_queue_deleted`, and absence
    of lingering `obsidian_anki_sync:results:*` keys after completion.

## References

-   Code: `src/obsidian_anki_sync/sync/queue_processor.py`
-   Code: `src/obsidian_anki_sync/worker.py`
-   Tests: `tests/test_queue_integration.py`

