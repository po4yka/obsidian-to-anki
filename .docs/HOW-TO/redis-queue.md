# Redis Queue for Parallel Card Generation

## When to use

Use the Redis queue when you have 100+ notes, need resilient parallel
processing, or want to decouple card generation from sync orchestration.

## Prerequisites

Install Redis and confirm it is running:

```bash
redis-cli ping
# Expected output: PONG
```

Install the Python worker dependency (ships with the project):

```bash
uv sync
```

## Configuration

Add the following to your `config.yaml`. All values shown are defaults unless
noted otherwise.

```yaml
enable_queue: true                       # default: false -- must opt in
redis_url: "redis://localhost:6379"
queue_max_retries: 3
redis_socket_connect_timeout: 5.0        # seconds

queue_max_wait_time_seconds: 18000       # 5 hours (min 60)
queue_job_timeout_seconds: 10800         # 3 hours (min 30)
queue_poll_interval: 0.5                 # seconds (range 0.1-10.0)
queue_poll_max_interval: 5.0             # seconds, adaptive backoff
queue_circuit_breaker_threshold: 3       # consecutive failures before open
queue_circuit_breaker_timeout: 60        # seconds before half-open

result_queue_ttl_seconds: 3600           # 1 hour
result_dead_letter_ttl_seconds: 3600
result_dead_letter_max_length: 100

worker_generation_timeout_seconds: 2700.0   # 45 min (min 60)
worker_validation_timeout_seconds: 2700.0   # 45 min (min 30)
```

## Starting the worker

The worker uses `arq` (async Redis queue) and processes `process_note_job`
jobs. Each worker handles up to 50 concurrent generations.

Worker module: `src/obsidian_anki_sync/worker.py`

```bash
# Start one worker (foreground)
uv run arq obsidian_anki_sync.worker.WorkerSettings
```

Start additional workers in separate terminals to increase throughput.

## Running sync with the queue

Option A -- CLI flag:

```bash
obsidian-anki-sync sync --use-queue
# or with an explicit Redis URL:
obsidian-anki-sync sync --use-queue --redis-url redis://localhost:6379
```

Option B -- config file (set `enable_queue: true` as shown above):

```bash
obsidian-anki-sync sync
```

## SLA monitoring

The system tracks per-stage timing against internal SLA targets:

| Stage        | SLA target | Notes                          |
|--------------|------------|--------------------------------|
| Generation   | ~900 s     | Per-note card generation       |
| Validation   | ~900 s     | Per-note card validation       |
| Total job    | ~5580 s    | ~93 min end-to-end per job     |

SLA violations are logged with detailed stage timing so you can identify
bottlenecks.

## Result handling

- Results are pushed to a Redis result queue and serialized as JSON containing
  `cards`, `success`, `job_id`, and `errors` fields.
- Failed results are routed to a dead letter queue (`<queue_name>:dlq`).
- The dead letter queue retains up to `result_dead_letter_max_length` entries
  for the configured TTL.

## Monitoring and troubleshooting

```bash
# General Redis health
redis-cli info

# Queue depth (replace <queue_name> with your configured queue)
redis-cli llen <queue_name>

# Inspect dead letter queue for failed jobs
redis-cli lrange <queue_name>:dlq 0 -1
```

**Circuit breaker** -- after 3 consecutive failures the circuit opens and the
worker stops dispatching new jobs for 60 seconds. Both thresholds are
configurable via `queue_circuit_breaker_threshold` and
`queue_circuit_breaker_timeout`.

Common issues:

1. **Worker exits immediately** -- verify Redis is reachable
   (`redis-cli ping`).
2. **Jobs stuck in queue** -- confirm at least one worker is running and check
   logs for timeout or SLA messages.
3. **High dead letter count** -- inspect DLQ entries for repeated errors;
   often caused by malformed notes or provider outages.

## See also

- [first-sync.md](first-sync.md) -- initial setup and first run
- [performance-tuning.md](performance-tuning.md) -- tuning throughput and
  resource usage
