# Model Configuration Analysis & Recommendations

**Date**: 2025-11-18
**Current Config**: `config.yaml`

## Current Model Configuration

| Task | Current Model | Recommended Model | Status |
|------|--------------|-------------------|--------|
| **QA Extractor** | `moonshotai/kimi-k2` | `qwen/qwen-2.5-72b-instruct` | ⚠️ Suboptimal |
| **Parser Repair** | `qwen/qwen3-max` | `qwen/qwen-2.5-32b-instruct` | ⚠️ Outdated |
| **Pre-Validator** | `qwen/qwen3-max` | `qwen/qwen-2.5-32b-instruct` | ⚠️ Outdated |
| **Generator** | `moonshotai/kimi-k2` | `qwen/qwen-2.5-72b-instruct` | ⚠️ Suboptimal |
| **Post-Validator** | `deepseek/deepseek-chat-v3.1` | `deepseek/deepseek-chat` | ⚠️ Has Issues |

## Issues Identified

### 1. **QA Extractor: Kimi K2** ⚠️
**Problem**:
- Expensive model for extraction task
- Overkill for structured JSON extraction
- Slower than necessary

**Recommendation**:
- Use `qwen/qwen-2.5-72b-instruct` for better cost/performance
- Excellent JSON schema compliance
- Strong multilingual support (EN/RU)
- Faster and cheaper than Kimi K2

**Why**: QA extraction is primarily about pattern recognition and structured output, not complex reasoning. Qwen 2.5 72B excels at this.

---

### 2. **Generator: Kimi K2** ⚠️
**Problem**:
- Not optimized for content generation
- More expensive than necessary
- Documentation recommends Qwen 2.5 72B for generation

**Recommendation**:
- Use `qwen/qwen-2.5-72b-instruct`
- Specifically optimized for content creation
- Better at APF format generation
- More cost-effective

**Why**: Documentation explicitly states "powerful content creation" for Qwen 2.5 72B in generator role.

---

### 3. **Pre-Validator & Parser Repair: Qwen3-max** ⚠️
**Problem**:
- `qwen/qwen3-max` appears to be outdated model name
- Not in context window registry
- Should use Qwen 2.5 series (2025 optimized)

**Recommendation**:
- Use `qwen/qwen-2.5-32b-instruct` for both
- Faster and more efficient for validation tasks
- Lower cost
- Better structured output support

**Why**: Validation tasks don't need large models. 32B is sufficient and faster.

---

### 4. **Post-Validator: DeepSeek Chat V3.1** ⚠️
**Problem**:
- Listed in `MODELS_WITH_STRUCTURED_OUTPUT_ISSUES`
- May have JSON schema compliance problems
- Documentation recommends `deepseek/deepseek-chat` (without version)

**Recommendation**:
- Use `deepseek/deepseek-chat` (latest stable)
- Better structured output support
- Strong reasoning capabilities maintained
- Avoids known issues

**Why**: The `-v3.1` variant has documented structured output issues. The base `deepseek-chat` model is more reliable.

---

## Recommended Configuration

### Optimized for Performance & Cost

```yaml
# QA Extraction - Pattern recognition, JSON schema
qa_extractor_model: "qwen/qwen-2.5-72b-instruct"
qa_extractor_temperature: 0.0

# Parser Repair - Fast structural fixes
parser_repair_model: "qwen/qwen-2.5-32b-instruct"
parser_repair_temperature: 0.0

# Pre-Validation - Fast structural checks
pre_validator_model: "qwen/qwen-2.5-32b-instruct"
pre_validator_temperature: 0.0

# Generation - High-quality content creation
generator_model: "qwen/qwen-2.5-72b-instruct"
generator_temperature: 0.3

# Post-Validation - Strong reasoning, reliable structured outputs
post_validator_model: "deepseek/deepseek-chat"
post_validator_temperature: 0.0
```

### Cost-Effective Alternative (Lower Quality)

If cost is a primary concern:

```yaml
qa_extractor_model: "qwen/qwen-2.5-32b-instruct"  # Smaller, cheaper
generator_model: "qwen/qwen-2.5-32b-instruct"      # Smaller, cheaper
# Others remain the same
```

### High-Quality Alternative (Higher Cost)

If quality is paramount:

```yaml
qa_extractor_model: "qwen/qwen-2.5-72b-instruct"  # Keep
generator_model: "qwen/qwen-2.5-72b-instruct"      # Keep
post_validator_model: "deepseek/deepseek-chat"     # Keep
# Consider adding reasoning models for complex tasks
```

---

## Model Capabilities Summary

### Qwen 2.5 Series (Recommended)

| Model | Size | Best For | Cost | Speed |
|-------|------|----------|------|-------|
| `qwen/qwen-2.5-72b-instruct` | 72B | Generation, Complex Extraction | Medium | Medium |
| `qwen/qwen-2.5-32b-instruct` | 32B | Validation, Simple Tasks | Low | Fast |

**Strengths**:
- ✅ Excellent JSON schema compliance
- ✅ Strong multilingual support (EN/RU)
- ✅ 2025 optimized models
- ✅ Cost-effective
- ✅ Fast structured outputs

### DeepSeek Chat

| Model | Best For | Notes |
|-------|----------|-------|
| `deepseek/deepseek-chat` | Post-validation, Reasoning | ✅ Reliable structured outputs |
| `deepseek/deepseek-chat-v3.1` | ❌ Avoid | Has structured output issues |

**Strengths**:
- ✅ Strong reasoning capabilities
- ✅ Good for validation tasks
- ⚠️ Avoid `-v3.1` variant

### Kimi K2

| Model | Best For | Notes |
|-------|----------|-------|
| `moonshotai/kimi-k2` | Reasoning, Analysis | Expensive, overkill for extraction |
| `moonshotai/kimi-k2-thinking` | Advanced reasoning | Not compatible with JSON schema |

**Strengths**:
- ✅ Excellent reasoning
- ❌ More expensive than needed
- ❌ Thinking variant incompatible with JSON schema

---

## Task Requirements Analysis

### QA Extraction Requirements
- ✅ Pattern recognition across markdown formats
- ✅ Semantic understanding of Q&A relationships
- ✅ Multilingual support (EN/RU)
- ✅ JSON schema structured output
- ✅ Accurate extraction without hallucination

**Best Model**: `qwen/qwen-2.5-72b-instruct`
- Strong at structured extraction
- Excellent JSON compliance
- Good multilingual support
- Cost-effective

### Generation Requirements
- ✅ APF 2.1 format HTML generation
- ✅ Front/Back/Extra sections
- ✅ Proper metadata
- ✅ Bilingual content
- ✅ Creative formatting

**Best Model**: `qwen/qwen-2.5-72b-instruct`
- Optimized for content creation
- Strong formatting capabilities
- Good multilingual support

### Validation Requirements
- ✅ Fast structural checks
- ✅ Factual accuracy verification
- ✅ Template compliance
- ✅ JSON schema validation

**Best Model**: `qwen/qwen-2.5-32b-instruct` (pre), `deepseek/deepseek-chat` (post)
- Fast and efficient
- Good reasoning for post-validation
- Reliable structured outputs

---

## Migration Steps

1. **Update config.yaml** with recommended models
2. **Test QA extraction** with new model
3. **Test card generation** with new model
4. **Monitor costs** - should decrease significantly
5. **Monitor quality** - should improve or stay same

---

## Expected Improvements

### Cost Reduction
- **QA Extractor**: ~30-50% cheaper (Kimi K2 → Qwen 2.5 72B)
- **Generator**: ~30-50% cheaper (Kimi K2 → Qwen 2.5 72B)
- **Pre-Validator**: ~40-60% cheaper (Qwen3-max → Qwen 2.5 32B)
- **Parser Repair**: ~40-60% cheaper (Qwen3-max → Qwen 2.5 32B)

### Performance Improvements
- **Faster validation**: 32B models are faster than 72B
- **Better structured outputs**: Qwen 2.5 series has better JSON schema support
- **Fewer errors**: Avoid DeepSeek V3.1 structured output issues

### Quality
- **Maintained or improved**: Qwen 2.5 72B is excellent for generation
- **Better extraction**: Optimized for structured tasks
- **More reliable**: Avoid known model issues

---

## Verification Checklist

After updating models:

- [ ] QA extraction works correctly
- [ ] JSON schemas are validated properly
- [ ] Bilingual content (EN/RU) is preserved
- [ ] Card generation quality maintained
- [ ] Validation passes correctly
- [ ] No structured output errors
- [ ] Cost reduction confirmed
- [ ] Performance acceptable

---

## References

- [Agent Summary](.docs/AGENT_SUMMARY.md) - Model recommendations per agent
- [Model Configuration Guide](.docs/MODEL_CONFIGURATION.md) - Detailed model guide
- [Structured Outputs](STRUCTURED_OUTPUTS_IMPLEMENTATION.md) - JSON schema requirements

---

**Recommendation**: Update to recommended models for better cost/performance balance while maintaining or improving quality.

