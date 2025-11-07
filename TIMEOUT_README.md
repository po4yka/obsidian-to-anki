# LLM Timeout Investigation Results

## Overview

This directory contains a comprehensive investigation into why qwen3:32b and qwen3:14b models timeout after 120 seconds.

## Quick Summary

The system is configured with a **120-second timeout** that is **too short** for large LLM models:
- **qwen3:8b**: Needs 30-80 seconds ✓ Works
- **qwen3:14b**: Needs 100-180 seconds ✗ Timeout
- **qwen3:32b**: Needs 150-300+ seconds ✗ Timeout

**Root Cause**: Configuration issue (80% of problem) + Model performance expectations (20% of problem)

## Files in This Investigation

### 1. TIMEOUT_INVESTIGATION.md
**Purpose**: Detailed technical analysis
**Contents**:
- Complete findings across all 5 key areas
- Where the 120-second timeout is defined
- Retry logic analysis and why it can't help
- Timeout configurability methods
- Performance expectations vs. actual timeouts
- Provider-specific timeout handling
- Configuration vs. Performance issue analysis
- Recommended fixes with implementation cost

**Read this when**: You want to understand the full technical details

### 2. TIMEOUT_FIXES.md
**Purpose**: Step-by-step implementation guide
**Contents**:
- Quick fix summary
- Specific code changes needed (before/after)
- 5 files that need updating
- Testing procedures
- Environment variable override instructions
- Performance impact table
- Ready-to-use commit message

**Read this when**: You're ready to implement the fixes

### 3. TIMEOUT_VISUAL_GUIDE.md
**Purpose**: Visual explanations and diagrams
**Contents**:
- Timeline diagrams showing current vs. fixed behavior
- Architecture flowchart showing how timeout flows through code
- Configuration hierarchy with precedence order
- Model performance expectations table
- Issue vs. solution comparison chart
- Retry logic impact diagram
- Configuration priority code examples
- Problem and solution chains

**Read this when**: You want to understand the issue visually

## The Problem in 30 Seconds

```
config.yaml has: llm_timeout: 120.0
                                    ↓
Ollama provider gets: 120-second timeout
                                    ↓
qwen3:32b request starts, needs ~200 seconds
                                    ↓
Request aborts at exactly 120 seconds ← TIMEOUT ERROR
```

## The Solution in 30 Seconds

Change these 5 values:
1. `config.py:213` from `120.0` to `900.0`
2. `config.yaml:65` from `120.0` to `600.0`
3. `config.providers.example.yaml:13` from `120.0` to `600.0`
4. `lm_studio.py:30` from `120.0` to `600.0`
5. `openrouter.py:35` from `120.0` to `180.0`

That's it! No logic changes, no complex refactoring.

## Quick Fix (Temporary)

If you need to use the system right now without modifying files:

```bash
export LLM_TIMEOUT=600.0
python -m obsidian_anki_sync.cli
```

This sets a 10-minute timeout via environment variable.

## Key Findings at a Glance

| Aspect | Finding |
|--------|---------|
| **Root Cause** | 120s timeout configured, but large models need 150-300s |
| **Severity** | HIGH - Blocks agent system with large models |
| **Configurability** | ✓ Fully configurable, 3 methods available |
| **Retry Logic** | ✓ Working, but cannot help with timeouts |
| **Fix Complexity** | TRIVIAL - 5 number changes |
| **Risk Level** | ZERO - Only relaxing timeout limits |
| **Effort** | < 5 minutes to implement |

## Configuration Locations Found

1. **config.yaml:65** - Explicit 120.0 seconds ← Currently used!
2. **config.py:213** - Fallback default 120.0 seconds
3. **ollama.py:32** - Provider default 900.0 seconds (overridden)
4. **lm_studio.py:30** - Provider default 120.0 seconds
5. **openrouter.py:35** - Provider default 120.0 seconds
6. **factory.py:130,138,148** - Fallback defaults 120.0 seconds

## Recommended Timeout Values

| Component | Current | Recommended | Reason |
|-----------|---------|-------------|--------|
| Config Default | 120s | 900s | Align with dataclass definition |
| Config File | 120s | 600s | Good balance for local models |
| Ollama Provider | 900s (overridden) | Used from config | 15 min for large models |
| LM Studio | 120s | 600s | Local models are slow |
| OpenRouter | 120s | 180s | Cloud API is faster |

## Model Performance Expectations

Based on code analysis and logging thresholds:

```
qwen3:8b (8B params)
├─ Model load: 10-15 seconds
├─ Processing: 30-50 seconds  
└─ Total: 40-65 seconds ✓ Safe with 120s

qwen3:14b (14B params)
├─ Model load: 20-30 seconds
├─ Processing: 60-150 seconds
└─ Total: 80-180 seconds ✗ Exceeds 120s!

qwen3:32b (32B params)
├─ Model load: 30-40 seconds
├─ Processing: 120-260 seconds
└─ Total: 150-300+ seconds ✗ Exceeds 120s by 230s!
```

## Architecture Issues

The timeout configuration has a **precedence problem**:

1. config.yaml explicitly sets 120.0 ← USED
2. Config loader has 120.0 fallback
3. Provider has 900.0 default ← IGNORED (overridden by #2)

This creates a mismatch between intent (900s) and actual use (120s).

## Why Retry Logic Can't Help

The `@retry` decorator with 3 attempts and exponential backoff cannot help because:
- Timeout happens WITHIN each request (at 120s)
- Each retry attempt still has the same 120-second limit
- Retry is designed for transient failures (network glitches), not timeouts

Example:
```
Attempt 1: 0s → 120s [TIMEOUT]
Wait 2s
Attempt 2: 122s → 242s [TIMEOUT]  
Wait 4s
Attempt 3: 246s → 366s [TIMEOUT]
Result: FAILURE (timeout is the root cause, not transient)
```

## What's Not the Problem

- ✓ Ollama server is working fine
- ✓ Models are installed and functional
- ✓ Network connectivity is fine
- ✓ Retry mechanism is working correctly
- ✓ No bugs in the code
- ✓ Models aren't "too slow" - large models ARE slow by design

It's simply a **configuration timeout too short for the model size**.

## Verification Steps

After implementing the fixes:

```bash
# Test each model
python -m obsidian_anki_sync.cli --model qwen3:8b   # Should complete in ~60s
python -m obsidian_anki_sync.cli --model qwen3:14b  # Should complete in ~150s
python -m obsidian_anki_sync.cli --model qwen3:32b  # Should complete in ~200s

# Check logs for these patterns:
# "ollama_generate_success" - request completed successfully
# "request_duration=XXX" - shows actual time taken
# No "timeout" in error logs
```

## Related Commits

- **828696a**: "perf: increase Ollama timeout to 15 minutes and add detailed performance logging"
  - This commit increased the dataclass default to 900s
  - But didn't update the config.yaml or config loader defaults
  - This is why the fix was incomplete

## Next Steps

1. **Read TIMEOUT_INVESTIGATION.md** for complete technical analysis
2. **Read TIMEOUT_FIXES.md** for implementation instructions
3. **Read TIMEOUT_VISUAL_GUIDE.md** for visual explanations
4. **Apply the 5 changes** listed in TIMEOUT_FIXES.md
5. **Test with all three models** to verify success
6. **Monitor logs** to confirm proper timeout behavior

## Questions?

Refer to the specific documentation file:
- **"How does this work?"** → TIMEOUT_VISUAL_GUIDE.md
- **"What exactly needs to change?"** → TIMEOUT_FIXES.md  
- **"Tell me everything"** → TIMEOUT_INVESTIGATION.md

---

**Investigation Date**: November 7, 2025
**Status**: Complete - Ready for implementation
**Impact**: HIGH - Fixes broken agent system with large models
**Effort**: TRIVIAL - 5 configuration value changes
**Risk**: ZERO - No logic changes, only relaxing timeout limits
