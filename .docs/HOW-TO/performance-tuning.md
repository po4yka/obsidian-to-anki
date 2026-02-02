# Performance Tuning

LLM inference dominates sync time. Network latency and Anki card creation
are secondary. The levers below are ordered from highest to lowest impact.

## 1. Use incremental sync

Content hashing (`compute_content_hash()`) detects changed notes so only
those are re-processed. Use this on every run after your
[first sync](first-sync.md):

```bash
obsidian-anki-sync sync --incremental
```

## 2. Choose a faster model

```yaml
model_preset: "fast"            # default: "balanced"
default_llm_model: "qwen3:8b"  # smaller = faster, lower quality
# default_llm_model: "qwen3:32b"  # larger = slower, higher quality
```

Defaults: `default_llm_model: "deepseek/deepseek-v3.2"`,
`fallback_llm_model: "qwen/qwen3-max"`. Per-task overrides via
`model_overrides` dict (cascade: per-task > preset > `default_llm_model`):

```yaml
model_overrides:
  generator: "qwen3:8b"
  validator: "deepseek/deepseek-v3.2"
```

See [switching-providers.md](switching-providers.md) for provider setup.

## 3. Disable optional agents

Each adds an LLM call per note. Disable in `config.yaml`:

```yaml
enable_card_splitting: false        # default: true
enable_context_enrichment: false    # default: true
enable_memorization_quality: false  # default: true
enable_highlight_agent: false       # default: true
enable_agent_memory: false          # default: true
# Already disabled by default:
# enable_self_reflection: false
# enable_cot_reasoning: false
# rag_enabled: false
```

## 4. Tune concurrency

```yaml
max_concurrent_generations: 5   # default; range 1-500
batch_size: 50                  # notes per batch
enable_batch_operations: true   # default
auto_adjust_workers: false      # set true to auto-tune concurrency
generation_timeout_seconds: 300.0   # default; min 30
generation_max_retries: 3           # default; range 1-5
```

Local Ollama on one GPU: 1-3 concurrent. Remote API (OpenRouter): 10-20.

## 5. Tune LLM parameters

```yaml
llm_temperature: 0.2    # default; lower = faster, more deterministic
llm_max_tokens: 8192    # default; reduce for shorter/faster responses
llm_timeout: 3600.0     # default (1 hour)
```

## 6. Use caching

**Validation cache:**

```bash
obsidian-anki-sync validate stats         # check cache stats
obsidian-anki-sync validate --incremental # use cached results
obsidian-anki-sync validate clear-cache   # reset
```

**RAG cache:** 500 MB DiskCache with LRU eviction (automatic when RAG is
enabled). **Agent memory cache:** `max_agent_memory_size_mb: 500`
(default; 0 = unlimited).

## 7. Large vaults (100+ notes)

Test with a subset first, then scale up:

```bash
obsidian-anki-sync sync --sample 10
```

Use the Redis queue for parallel processing across multiple workers
(see [redis-queue.md](redis-queue.md)):

```bash
obsidian-anki-sync sync --use-queue
```

**File descriptor limits** -- for constrained environments (~128
descriptors):

```yaml
archiver_batch_size: 16             # default: 64
archiver_min_fd_headroom: 8         # default: 32
archiver_fd_poll_interval: 0.05     # seconds, backoff interval
```

## 8. Monitoring

```bash
obsidian-anki-sync progress                    # session progress
obsidian-anki-sync analyze-logs --days 7       # identify slow stages
obsidian-anki-sync sync --incremental --verbose  # real-time details
```

Preflight blocking thresholds (`obsidian-anki-sync check`):

| Resource        | Threshold |
|-----------------|-----------|
| Free memory     | < 4 GB    |
| Free disk       | < 0.5 GB  |
| Network latency | > 250 ms  |

## Quick-start: maximum speed config

```yaml
model_preset: "fast"
default_llm_model: "qwen3:8b"
max_concurrent_generations: 10
batch_size: 50
enable_card_splitting: false
enable_context_enrichment: false
enable_memorization_quality: false
enable_highlight_agent: false
enable_agent_memory: false
llm_max_tokens: 4096
```

```bash
obsidian-anki-sync sync --incremental --verbose
```

## See also

- [First sync walkthrough](first-sync.md)
- [Redis queue for parallel processing](redis-queue.md)
- [Switching LLM providers](switching-providers.md)
