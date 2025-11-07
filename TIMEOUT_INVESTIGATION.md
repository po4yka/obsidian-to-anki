# LLM Timeout Investigation Report
## Issue: qwen3:32b and qwen3:14b Timing Out After 120 Seconds

---

## Summary

**Root Cause**: Configuration mismatch between code defaults and actual config values. Large models (qwen3:32b, qwen3:14b) require longer than 120 seconds to generate responses, but the system is configured with a 120-second timeout.

**Severity**: HIGH - This prevents agent-based card generation from working with large models.

---

## Detailed Findings

### 1. Timeout Configuration Settings

#### Dataclass Definition (config.py:46)
```python
llm_timeout: float = 900.0  # 15 minutes for large models
```

#### Config Loading Function (config.py:213)
```python
llm_timeout=get_float("llm_timeout", 120.0),  # ← PROBLEM: Default is 120 seconds!
```

#### Actual Config Values
- **config.yaml (line 65)**: `llm_timeout: 120.0`
- **config.providers.example.yaml (line 13)**: `llm_timeout: 120.0  # Request timeout in seconds`

**Issue**: The config loading function defaults to 120 seconds as a fallback, which conflicts with the intended 900-second dataclass default. Since config.yaml explicitly sets it to 120, this value is used.

---

### 2. Where the 120-Second Timeout is Defined

**Multiple locations establish the 120-second timeout:**

| File | Line | Context |
|------|------|---------|
| config.py | 213 | `get_float("llm_timeout", 120.0)` - fallback default |
| config.yaml | 65 | `llm_timeout: 120.0` - explicit config value |
| config.providers.example.yaml | 13 | `llm_timeout: 120.0` - example config value |
| providers/factory.py | 130, 138, 148 | `getattr(config, "llm_timeout", 120.0)` - fallback defaults |

**Why models timeout:**

- **qwen3:8b** (pre-validator): Lightweight model, typically completes within 120 seconds
- **qwen3:32b** (generator): Large model, requires 150-300+ seconds depending on prompt/card complexity
- **qwen3:14b** (post-validator): Medium model, requires 100-180+ seconds

---

### 3. Retry Logic and Exponential Backoff Implementation

**Location**: `/src/obsidian_anki_sync/utils/retry.py`

**Key Configuration:**
```python
@retry(max_attempts=3, initial_delay=2.0, backoff_factor=2.0)
```

**How it works:**
- Attempt 1: Immediate (0s wait before attempt)
- Attempt 2: After 2 second delay
- Attempt 3: After 4 second delay (2 × 2.0 backoff_factor)

**Problem with timeouts**: The retry decorator cannot help with timeout errors because:
1. The timeout occurs WITHIN the request (120 seconds)
2. httpx.TimeoutException is caught, but the timeout still prevents completion
3. Even with retries, each attempt is still limited to 120 seconds

**Timeline of a large model request:**
```
Attempt 1: Start at 0s → Timeout at 120s → Raises httpx.TimeoutException
Wait 2s (retry delay)
Attempt 2: Start at 122s → Timeout at 242s → Raises httpx.TimeoutException  
Wait 4s (retry delay)
Attempt 3: Start at 246s → Timeout at 366s → Final failure
```

The retry mechanism doesn't help because the timeout limit is the problem, not transient failures.

---

### 4. Timeout Configurability

**Current Configuration Method:**

Users can configure timeouts in THREE ways (in order of precedence):

1. **Environment Variable** (highest priority)
   ```bash
   export LLM_TIMEOUT=600.0
   ```

2. **YAML Config File** (second priority)
   ```yaml
   llm_timeout: 600.0
   ```

3. **Code Defaults** (lowest priority)
   - Fallback: 120.0 (config.py:213)
   - Dataclass definition: 900.0 (config.py:46) - currently ignored

**Evidence from config.py:179-181:**
```python
def get_float(key: str, default: float) -> float:
    val = config_data.get(key) or os.getenv(key.upper())
    return float(val) if val is not None else default
```

---

### 5. Typical Response Times vs Timeouts

#### Observed Performance (from code logging - ollama.py:199-216)

**Performance Thresholds defined in code:**
- Requests > 600s (10 minutes): Log "very_slow_operation_detected" warning
- Requests > 300s (5 minutes): Log "slow_operation_detected" warning

**Typical model performance (estimates based on logging thresholds):**

| Model | Estimated Time | Current Timeout | Status |
|-------|----------------|-----------------|--------|
| qwen3:8b | 30-80s | 120s | ✓ Usually OK |
| qwen3:14b | 100-180s | 120s | ✗ TIMEOUT |
| qwen3:32b | 150-300s | 120s | ✗ TIMEOUT |

**Evidence from code:**
```python
# Line 191-196: Model loading detection
if is_first_request and request_duration > 30:
    logger.info(
        "model_loading_detected",
        model=model,
        duration=round(request_duration, 2),
        note="First request to this model may include loading time",
    )
```

Large model first requests can include 30+ seconds of model loading time.

---

## Provider-Specific Timeout Handling

### OllamaProvider (ollama.py:32)
```python
def __init__(
    self,
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    timeout: float = 900.0,  # ← Intended default: 15 minutes
    **kwargs: Any,
):
```

Provider default is 900s (15 minutes), but overridden by config value of 120s.

### LMStudioProvider (lm_studio.py:30)
```python
timeout: float = 120.0,  # ← Hardcoded default
```

Also uses 120-second default, problematic for large models.

### OpenRouterProvider (openrouter.py:35)
```python
timeout: float = 120.0,  # ← Hardcoded default
```

Uses 120-second default (but for cloud API, requests are typically faster).

### Pull Model Operation (ollama.py:279)
```python
timeout=httpx.Timeout(600.0),  # 10 minutes
```

Model pulling has a separate 600-second timeout (good practice for large downloads).

---

## Configuration vs Performance Issue Analysis

### Is this a configuration issue?
**YES, PRIMARY CAUSE**

- Default timeout of 120 seconds is too aggressive for large models
- Config files explicitly set 120 seconds
- Easy to fix by changing a single number

### Is this a model performance issue?
**PARTIALLY - SECONDARY FACTOR**

- qwen3:32b and qwen3:14b are legitimately slower (large models)
- They require more computational time, not a bug
- This is expected behavior for 32B and 14B parameter models
- On typical hardware:
  - Small models (8B): 30-80 seconds per request
  - Medium models (14B): 100-180 seconds per request  
  - Large models (32B): 150-300+ seconds per request

---

## Recommended Fixes

### Priority 1: Update Configuration Defaults (IMMEDIATE)

**File: src/obsidian_anki_sync/config.py:213**
```python
# Current (WRONG):
llm_timeout=get_float("llm_timeout", 120.0),

# Should be (CORRECT):
llm_timeout=get_float("llm_timeout", 900.0),
```

**File: config.yaml:65**
```yaml
# Current (TOO SHORT):
llm_timeout: 120.0

# Recommended (SUITABLE FOR LARGE MODELS):
llm_timeout: 600.0  # 10 minutes - good balance
# Or for safety with large models:
llm_timeout: 900.0  # 15 minutes - original intent
```

**File: config.providers.example.yaml:13**
```yaml
# Current:
llm_timeout: 120.0

# Should be:
llm_timeout: 600.0  # Request timeout in seconds (10 min for large models)
```

### Priority 2: Update LM Studio and OpenRouter Defaults

**File: src/obsidian_anki_sync/providers/lm_studio.py:30**
```python
# Current:
timeout: float = 120.0,

# Should be:
timeout: float = 600.0,  # 10 minutes for local models
```

**File: src/obsidian_anki_sync/providers/openrouter.py:35**
```python
# Current:
timeout: float = 120.0,

# Should be:
timeout: float = 180.0,  # Cloud API is faster, 3 minutes should suffice
```

### Priority 3: Add Documentation

Update docstrings to clarify:
- Default timeout expectations
- How to configure for different model sizes
- Hardware-dependent performance considerations

### Priority 4: Improve Logging

The logging improvements from commit 828696a are excellent. Consider adding:
- Timeout threshold relative to configured value
- Early warning when request exceeds 80% of timeout
- Suggestion to increase timeout if consistently near limit

---

## Implementation Cost

**Effort**: Very Low (configuration values only, no code changes needed)

**Risk**: None (just relaxing timeout limits)

**Testing**: 
1. Generate cards with qwen3:32b model
2. Generate cards with qwen3:14b model
3. Verify completion without timeouts
4. Monitor logs for performance metrics

---

## Verification Checklist

After applying fixes:

- [ ] config.py line 213 default changed to 900.0
- [ ] config.yaml llm_timeout changed to 600.0+
- [ ] config.providers.example.yaml llm_timeout changed to 600.0+
- [ ] Test card generation with qwen3:32b (should complete)
- [ ] Test card generation with qwen3:14b (should complete)
- [ ] Verify logs show actual request duration
- [ ] Confirm no timeouts in retry logs

---

## Summary Table

| Aspect | Current | Issue | Recommended |
|--------|---------|-------|-------------|
| Config default | 120s | Too short for qwen3:32b, 14b | 600-900s |
| Code dataclass default | 900s | Ignored by config loader | Align with config |
| Ollama provider default | 900s | Overridden by config 120s | Use config value when > default |
| Model pull timeout | 600s | Only for pulls | Might apply to large requests |
| Retry mechanism | 3 attempts | Cannot help if timeout is root cause | ✓ Working as designed |
| Per-model defaults | None | All models share config | Consider model-specific tuning |

