# Timeout Configuration Fixes

## Quick Fix Summary

The 120-second timeout is too short for qwen3:32b and qwen3:14b models. This document shows the exact changes needed.

---

## Fix #1: config.py - Update Config Loading Default

**File**: `src/obsidian_anki_sync/config.py:213`

**Current Code**:
```python
llm_timeout=get_float("llm_timeout", 120.0),
```

**Fixed Code**:
```python
llm_timeout=get_float("llm_timeout", 900.0),
```

**Reason**: The fallback default should match the dataclass definition (line 46) which specifies 900.0 seconds (15 minutes) for large models.

---

## Fix #2: config.yaml - Update Explicit Timeout Value

**File**: `config.yaml:65`

**Current Code**:
```yaml
llm_timeout: 120.0
llm_max_tokens: 2048
```

**Fixed Code (Option A - Recommended for balance)**:
```yaml
llm_timeout: 600.0  # 10 minutes - good for qwen3:32b and qwen3:14b
llm_max_tokens: 2048
```

**Fixed Code (Option B - Conservative, matching dataclass)**:
```yaml
llm_timeout: 900.0  # 15 minutes - original intent for large models
llm_max_tokens: 2048
```

**Reason**: Current value of 120 seconds is insufficient for:
- qwen3:32b (typical: 150-300+ seconds)
- qwen3:14b (typical: 100-180+ seconds)

600 seconds provides a good balance between responsiveness and accommodating slow requests.

---

## Fix #3: config.providers.example.yaml - Update Example Config

**File**: `config.providers.example.yaml:13`

**Current Code**:
```yaml
# Common LLM settings (apply to all providers)
llm_temperature: 0.2      # Sampling temperature (0.0-1.0)
llm_top_p: 0.3           # Top-p sampling
llm_timeout: 120.0       # Request timeout in seconds
llm_max_tokens: 2048     # Maximum tokens in response
```

**Fixed Code**:
```yaml
# Common LLM settings (apply to all providers)
llm_temperature: 0.2      # Sampling temperature (0.0-1.0)
llm_top_p: 0.3           # Top-p sampling
llm_timeout: 600.0       # Request timeout in seconds (10 min for large models like qwen3:32b)
llm_max_tokens: 2048     # Maximum tokens in response
```

**Reason**: Example config should demonstrate best practices for large model deployments.

---

## Fix #4: lm_studio.py - Update LM Studio Default

**File**: `src/obsidian_anki_sync/providers/lm_studio.py:30`

**Current Code**:
```python
def __init__(
    self,
    base_url: str = "http://localhost:1234/v1",
    timeout: float = 120.0,
    max_tokens: int = 2048,
    **kwargs: Any,
):
```

**Fixed Code**:
```python
def __init__(
    self,
    base_url: str = "http://localhost:1234/v1",
    timeout: float = 600.0,  # 10 minutes for local models
    max_tokens: int = 2048,
    **kwargs: Any,
):
```

Also update the docstring (line 23):
```python
# Current:
timeout: Request timeout in seconds (default: 120.0)

# Fixed:
timeout: Request timeout in seconds (default: 600.0 - 10 minutes for large local models)
```

**Reason**: Local LM Studio instances with large models also need extended timeout.

---

## Fix #5: openrouter.py - Update OpenRouter Default

**File**: `src/obsidian_anki_sync/providers/openrouter.py:35`

**Current Code**:
```python
def __init__(
    self,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 120.0,
    max_tokens: int = 2048,
    site_url: str | None = None,
    site_name: str | None = None,
    **kwargs: Any,
):
```

**Fixed Code**:
```python
def __init__(
    self,
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 180.0,  # 3 minutes - cloud API is faster than local
    max_tokens: int = 2048,
    site_url: str | None = None,
    site_name: str | None = None,
    **kwargs: Any,
):
```

Also update the docstring (line 25):
```python
# Current:
timeout: Request timeout in seconds (default: 120.0)

# Fixed:
timeout: Request timeout in seconds (default: 180.0 - cloud API is faster than local)
```

**Reason**: Cloud APIs are typically faster than local inference. 3 minutes should be sufficient. Keep it lower than local timeouts to avoid long waits for cloud API issues.

---

## Testing These Fixes

After applying the changes above, test with:

```bash
# Test with qwen3:32b (large model)
python -m obsidian_anki_sync.cli --model qwen3:32b

# Test with qwen3:14b (medium model)  
python -m obsidian_anki_sync.cli --model qwen3:14b

# Test with qwen3:8b (small model) - should still work
python -m obsidian_anki_sync.cli --model qwen3:8b
```

Monitor logs for:
- Successful completion without timeout errors
- Actual request durations logged
- No retry attempts due to timeouts

---

## Environment Variable Override

If you want to use environment variables instead of modifying config files:

```bash
# Set 10-minute timeout via environment variable
export LLM_TIMEOUT=600.0

# Then run the sync
python -m obsidian_anki_sync.cli
```

This will override the config.yaml value.

---

## Performance Impact

| Model | Old Timeout | New Timeout | Improvement |
|-------|------------|------------|-------------|
| qwen3:8b | 120s | 600s | More stable (still completes quickly) |
| qwen3:14b | 120s (TIMEOUT) | 600s (OK) | Fixes timeout issue |
| qwen3:32b | 120s (TIMEOUT) | 600s (OK) | Fixes timeout issue |

---

## Commit Message (if committing these changes)

```
fix: increase timeout defaults to support large models (qwen3:32b, qwen3:14b)

- Increase config.py fallback default from 120s to 900s
- Update config.yaml explicit timeout from 120s to 600s (10 minutes)
- Update example config to use 600s for large model deployments
- Increase LM Studio default from 120s to 600s for local models
- Increase OpenRouter default from 120s to 180s for cloud API

The previous 120-second timeout was too short for:
- qwen3:32b: typically needs 150-300+ seconds
- qwen3:14b: typically needs 100-180+ seconds

These changes align with the intended 900s default in the Config dataclass
and accommodate both local and cloud model inference times.

Fixes timeout issues when using agent system with large models.
```

---

## Verification

After applying ALL fixes above, verify:

1. `config.py:213` uses 900.0 as default
2. `config.yaml:65` uses 600.0
3. `config.providers.example.yaml:13` uses 600.0
4. `lm_studio.py:30` uses 600.0
5. `openrouter.py:35` uses 180.0

Run tests:
```bash
pytest tests/test_sync_engine.py -v
pytest tests/test_anki_client.py -v
```

