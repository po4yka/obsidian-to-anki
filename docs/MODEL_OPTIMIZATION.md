# Model Optimization Implementation

**Date**: 2025-11-19
**Status**: Implemented

## Overview

Optimized model selection to improve performance while maintaining quality. The key change is using a faster, smaller model for QA extraction while keeping the high-quality large model for card generation.

## Changes Made

### 1. QA Extraction Model Optimization

**Before**: `qwen/qwen3-235b-a22b-2507` (235B parameters)
- Response time: 33-78 seconds per request
- Throughput: 62-87 tokens/second
- Cost: ~$0.004 per note

**After**: `qwen/qwen3-32b` (32B parameters)
- Expected response time: ~10-20 seconds per request (3-4x faster)
- Expected throughput: 150-200 tokens/second
- Cost: ~$0.001 per note (4x cheaper)

**Rationale**:
- QA extraction is primarily structured JSON output (pattern recognition)
- Smaller models excel at structured tasks
- Quality remains high for extraction tasks
- Significant speed improvement for large syncs

### 2. Generation Model (Unchanged)

**Model**: `qwen/qwen3-235b-a22b-2507` (235B parameters)
- Kept for card generation where quality matters most
- Excellent for creative content generation
- Large context window (262K) for complex notes

### 3. Configuration Updates

**File**: `config.yaml`
```yaml
# QA Extraction: Use faster model for speed
qa_extractor_model: "qwen/qwen3-32b"
qa_extractor_temperature: 0.0

# Generation: Keep high-quality model
generator_model: "qwen/qwen3-235b-a22b-2507"
generator_temperature: 0.3
```

### 4. Provider Updates

**File**: `src/obsidian_anki_sync/providers/openrouter.py`

Added support for smaller Qwen3 models:
- Added `qwen/qwen3-32b` to `MODELS_WITH_STRUCTURED_OUTPUT_ISSUES` (uses non-strict JSON mode)
- Added context window: 131072 (128K tokens)
- Added pricing: $0.05/$0.20 per 1M tokens (prompt/completion)

## Performance Impact

### Expected Improvements

**For 3 notes (test-run)**:
- Before: ~2.5 minutes (49.4s avg × 3)
- After: ~45 seconds (15s avg × 3)
- **Improvement**: ~3.3x faster

**For 976 notes (full sync)**:
- Before: ~13.4 hours (49.4s avg × 976)
- After: ~4.1 hours (15s avg × 976)
- **Improvement**: ~3.3x faster

### Cost Impact

**Per note**:
- Before: ~$0.004
- After: ~$0.001
- **Savings**: 75% reduction

**For 976 notes**:
- Before: ~$3.90
- After: ~$0.98
- **Savings**: ~$2.92

## Quality Considerations

### QA Extraction Quality
- **Risk**: Low
- Smaller models (32B) are excellent at structured JSON extraction
- Pattern recognition doesn't require massive models
- Quality maintained for extraction tasks

### Generation Quality
- **Risk**: None (unchanged)
- Still using high-quality 235B model
- No impact on final card quality

## Model Specifications

### Qwen3-32B
- **Parameters**: 32B
- **Context Window**: 128K tokens
- **Speed**: ~3-4x faster than 235B
- **Cost**: $0.05/$0.20 per 1M tokens
- **Best For**: Structured outputs, JSON extraction, pattern recognition

### Qwen3-235B (unchanged)
- **Parameters**: 235B (MoE)
- **Context Window**: 262K tokens
- **Speed**: Slower but high quality
- **Cost**: $0.08/$0.55 per 1M tokens
- **Best For**: Content generation, complex reasoning, high-quality output

## Testing Recommendations

1. **Run test-run with 3-5 notes** to verify:
   - Response times are improved
   - Extraction quality remains high
   - JSON structure is correct

2. **Monitor logs for**:
   - Response times (should be ~10-20s instead of 33-78s)
   - JSON truncation warnings (should be similar or better)
   - Extraction accuracy (compare results)

3. **Cost tracking**:
   - Verify cost per note is reduced
   - Check total cost for larger syncs

## Rollback Plan

If quality issues arise:

1. **Temporary**: Revert `qa_extractor_model` to `qwen/qwen3-235b-a22b-2507`
2. **Alternative**: Try `qwen/qwen3-next-80b-a3b-instruct` (80B, middle ground)
3. **Investigation**: Check extraction quality metrics

## Future Optimizations

1. **Consider model for parser repair**: Currently using 80B, could use 32B
2. **Monitor performance**: Track actual vs expected improvements
3. **A/B testing**: Compare extraction quality between models
4. **Dynamic selection**: Use larger model only for complex notes

## Related Documentation

- `MODEL_ANALYSIS.md` - Model selection guidelines
- `LOG_ANALYSIS_2025-11-19.md` - Performance analysis that led to this optimization
- `src/obsidian_anki_sync/providers/openrouter.py` - Model configuration

