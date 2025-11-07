# Service Log Analysis Report

**Date:** 2025-11-07
**Session Duration:** ~19 minutes (1147.9 seconds)
**Notes Processed:** 2
**Cards Generated:** 0
**Failure Rate:** 100%

## Executive Summary

The obsidian-to-anki service is experiencing a critical failure in the card generation pipeline. All card generation attempts are failing at the post-validation stage with "Invalid card header format" errors, followed by auto-fix failures where the LLM returns empty JSON responses. This results in a 100% failure rate.

## Critical Issues Identified

### 1. **Invalid Card Header Format (Primary Blocker)**

**Location:** Post-validation syntax check (`src/obsidian_anki_sync/apf/linter.py:160-167`)

**What's Happening:**
- The generator agent (qwen3:32b) produces APF cards
- The post-validator detects invalid card headers
- Error: `Card 1: Invalid card header format`

**Expected Format:**
```html
<!-- Card 1 | slug: lowercase-dash-slug | CardType: Simple | Tags: tag1 tag2 tag3 -->
```

**Validation Regex:**
```python
r"<!-- Card (\d+) \| slug: ([a-z0-9-]+) \| CardType: (Simple|Missing|Draw) \| Tags: (.+?) -->"
```

**Possible Causes:**
1. **Extra/missing spaces** around the pipe separators (`|`)
2. **Case sensitivity** issues (CardType must be exact: Simple, Missing, or Draw)
3. **Tag format** issues (must be space-separated, no commas)
4. **Slug format** violations (only lowercase, numbers, and hyphens)
5. **Model confusion** between the instruction format and actual output

**Evidence from Logs:**
```
2025-11-07T16:06:52.408802Z [warning] post_validation_syntax_failed
  error_breakdown={'Card 1: Invalid': 2} errors_count=2
2025-11-07T16:06:52.408858Z [warning] validation_error_detail
  error='[cs-mvi-pattern-/-паттерн-mvi-(mod-1-en] APF format: Card 1: Invalid card header format'
```

### 2. **Auto-Fix Returns Empty JSON (Secondary Blocker)**

**Location:** Auto-fix agent (`src/obsidian_anki_sync/agents/post_validator.py:439-446`)

**What's Happening:**
- After detecting syntax errors, the system attempts auto-fix
- The qwen3:14b model is called to fix the errors
- The model returns `{}` (empty JSON object)
- This triggers a validation error in `base.py:109-118`

**Error Pattern:**
```
2025-11-07T16:02:05.592589Z [info] ollama_generate_success
  completion_tokens=2 response_length=2
2025-11-07T16:02:05.592904Z [error] empty_json_response
  provider=OllamaProvider response_text={}
2025-11-07T16:02:05.593191Z [error] llm_error
  error_type=empty_response
  message='LLM returned empty or invalid response for auto-fix'
```

**Root Causes:**
1. **Prompt complexity**: The auto-fix prompt may be too complex for qwen3:14b
2. **Model limitations**: qwen3:14b may not have enough context/capability for this task
3. **Token limit hit**: The model might be hitting token limits mid-generation
4. **Temperature setting**: Temperature is 0.0, which can cause deterministic failures
5. **Prompt engineering issue**: The model doesn't understand what's being asked

**Evidence:**
- Consistent `completion_tokens=2` (only returning `{}`)
- Response time ~10-17 seconds, indicating the model completes quickly
- Same failure pattern across all 4 attempts (2 notes × 2 retries)

### 3. **Performance Issues**

**Metrics:**
- Average time per note: **573.96 seconds** (~9.5 minutes)
- Total processing time: **1147.9 seconds** (~19 minutes)
- Token generation speed: 10-40 tokens/second
- Pre-validation: ~30-44 seconds per note
- Card generation: ~210-250 seconds per note (2 cards)
- Post-validation: ~10-17 seconds per attempt

**Contributing Factors:**
1. **Model loading delays**: First request to qwen3:32b includes 115+ second loading time
2. **Large models**: Using qwen3:32b (32B parameters) is slow but provides quality
3. **Sequential processing**: Cards generated one at a time, not in parallel
4. **Multiple retries**: Failed attempts compound the time
5. **Context size**: High context utilization (12-14%) for large prompts

## Pipeline Flow Analysis

```
┌─────────────────────┐
│  1. Index Building  │  ✓ Success (966 notes indexed, 976 discovered)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  2. Pre-Validation  │  ⚠️ Auto-fixing frontmatter issues (working)
│     (qwen3:8b)      │  ~30-44s per note
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 3. Card Generation  │  ✓ Generating cards (producing output)
│    (qwen3:32b)      │  ~105-125s per card
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ 4. Post-Validation  │  ✗ FAILING: Invalid card header format
│    (qwen3:14b)      │  Syntax validation detects format errors
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  5. Auto-Fix        │  ✗ FAILING: Returns empty JSON {}
│    (qwen3:14b)      │  ~10-17s per attempt
└──────────┬──────────┘
           │
           ▼
         FAIL (ValueError: Agent pipeline failed)
```

## Additional Findings

### Parsing Issues During Indexing

**Warnings:** Multiple "incomplete_qa_block" and "no_qa_pairs_found" warnings during vault indexing

**Summary:**
```
- Total discovered: 976 notes
- Successfully indexed: 966 notes
- Errors: 10 notes (ParserError)
- Common issue: "Missing required fields"
```

**Sample Errors:**
```
70-Kotlin/q-debugging-coroutines-techniques--kotlin--medium.md: Missing required fields
70-Kotlin/q-mutex-synchronized-coroutines--kotlin--medium.md: Missing required fields
70-Kotlin/q-coroutine-exception-handler--kotlin--medium.md: Missing required fields
```

**Issue Types:**
- Incomplete Q&A blocks (state=RELATED, FOLLOWUPS, QUESTION_RU, ANSWER_EN)
- Missing Russian content (has_answer=False has_question=True)
- Parsing incomplete (failed=1-4 blocks per note)

These parsing issues indicate data quality problems in the source vault that could affect card generation quality.

### Topic Mismatches

**Pattern:** 966 topic mismatches detected

**Examples:**
- `30-System-Design -> system-design`: 10 notes
- `70-Kotlin -> kotlin`: 237 notes
- `40-Android -> android`: 527 notes

**Assessment:** This appears to be expected behavior (directory name mapping to topic tags), not an error.

## Recommendations

### Immediate Actions (Critical Priority)

1. **Debug Card Header Format**
   - Add debug logging to capture the exact card header produced by the generator
   - Compare against the expected regex pattern character-by-character
   - Check for:
     - Unicode characters in tags (e.g., Cyrillic characters)
     - Extra whitespace or line breaks
     - Special characters in slugs
   - File: `src/obsidian_anki_sync/agents/generator.py`

2. **Fix Auto-Fix Prompt**
   - Simplify the auto-fix prompt for qwen3:14b
   - Provide explicit examples of corrections
   - Consider switching to a larger model (qwen3:32b) for auto-fix
   - Add validation that the model returns the required JSON structure
   - File: `src/obsidian_anki_sync/agents/post_validator.py:383-404`

3. **Add Debug Artifacts**
   - Enable saving of failed card outputs to debug directory
   - Save both generator output and validation errors
   - This will help diagnose the exact format mismatch
   - File: `src/obsidian_anki_sync/agents/debug_artifacts.py`

### Short-Term Fixes

4. **Improve Error Messages**
   - Show the actual card header that failed validation
   - Show what the regex expected vs. what it got
   - Add character-by-character diff for debugging
   - File: `src/obsidian_anki_sync/apf/linter.py:166`

5. **Enhance System Prompt**
   - Make card header format requirements MORE explicit in `CARDS_PROMPT.md`
   - Add a validation checklist at the end
   - Include common mistakes to avoid
   - File: `.docs/CARDS_PROMPT.md:52`

6. **Model Selection for Auto-Fix**
   - Consider using qwen3:32b instead of qwen3:14b for auto-fix
   - The 14B model may not have sufficient capability
   - Alternative: Implement rule-based fixes for common patterns

### Medium-Term Improvements

7. **Performance Optimization**
   - Implement parallel card generation for multiple languages
   - Cache model loading to avoid 115s delays
   - Use streaming responses to detect errors faster
   - Consider smaller models for pre-validation (qwen3:3b)

8. **Validation Strategy**
   - Add format validation BEFORE calling post-validator
   - Implement incremental validation (header → tags → content)
   - Provide specific error messages for each validation failure

9. **Testing Infrastructure**
   - Add unit tests that validate generated cards against real LLM output
   - Create regression tests with the current failing examples
   - Test each agent independently with known inputs

10. **Source Data Quality**
    - Fix the 10 notes with ParserError
    - Add validation for Russian content completeness
    - Implement better error handling for incomplete Q&A blocks

## Diagnostic Commands

To further investigate, run these commands:

```bash
# Enable debug mode to see full card outputs
uv run obsidian-anki-sync test-run --count 1 --use-agents --debug

# Check what the generator is actually producing
grep -r "Card 1 |" .debug_artifacts/ 2>/dev/null

# Test the validation regex directly
python -c "import re; print(re.match(r'<!-- Card (\d+) \| slug: ([a-z0-9-]+) \| CardType: (Simple|Missing|Draw) \| Tags: (.+?) -->', '<!-- Card 1 | slug: test | CardType: Simple | Tags: tag1 tag2 -->'))"

# Check model availability
curl http://localhost:11434/api/tags

# Test a single LLM call
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "prompt": "Return valid JSON: {\"test\": \"value\"}",
  "format": "json"
}'
```

## Conclusion

The service is experiencing a critical failure where:

1. **Generator produces invalid card headers** (exact format unknown without debug output)
2. **Auto-fix agent cannot correct them** (returns empty JSON)
3. **100% failure rate** for all card generation attempts

**Priority:** Immediate investigation required to:
- Capture actual generator output for comparison
- Fix the card header format mismatch
- Improve or replace the auto-fix mechanism

**Impact:** The service is completely non-functional for card generation with agents enabled.

**Next Steps:**
1. Add debug logging to capture generator output
2. Identify the exact format mismatch
3. Either fix generator prompt or relax validation
4. Fix or replace auto-fix agent functionality
5. Retest with the sample notes

## Related Files

- `src/obsidian_anki_sync/apf/linter.py` - Validation logic (line 160-167)
- `src/obsidian_anki_sync/agents/post_validator.py` - Auto-fix logic (line 383-480)
- `src/obsidian_anki_sync/agents/generator.py` - Card generation
- `src/obsidian_anki_sync/providers/base.py` - Empty JSON detection (line 109-118)
- `.docs/CARDS_PROMPT.md` - Generator system prompt (line 52)
