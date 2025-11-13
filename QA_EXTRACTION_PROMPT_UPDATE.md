# Q&A Extraction Prompt Update

## Changes Made

Updated the Q&A extraction prompt to **skip follow-up questions without answers**, preventing validation warnings and failed extractions.

## Problem

Previously, the LLM was extracting questions from "Follow-up Questions" or "Additional Questions" sections that didn't have answers in the note, resulting in:

```
WARNING | missing_en_content_in_extraction | card_index=8 has_question=True has_answer=False
WARNING | skipping_invalid_qa_pair | card_index=8
```

This created noise in the logs and wasted tokens on incomplete Q&A pairs.

## Solution

### Updated Extraction Rules (Rule #7)

**Added explicit instruction:**
```
7. **IMPORTANT**: ONLY extract Q&A pairs where BOTH question AND answer are present
   - Skip sections labeled "Follow-up Questions", "Additional Questions", or similar
   - Skip questions without explicit answers in the note
   - Do NOT create separate Q&A pairs for unanswered questions
```

### Updated Critical Requirements Section

**Before:**
```
IMPORTANT:
- If a language is not in the language_tags, leave that field empty
- Preserve markdown formatting in questions and answers
- Be thorough - extract ALL Q&A pairs you can identify
```

**After:**
```
CRITICAL REQUIREMENTS:
- ONLY extract Q&A pairs where BOTH question AND answer exist in the note
- SKIP any "Follow-up Questions", "Additional Questions", or unanswered question sections
- If a language is not in the language_tags, leave that field empty
- Preserve markdown formatting in questions and answers
- Extract ALL Q&A pairs that have complete answers
- Follow-up questions can be listed in the 'followups' field for context, but NOT as separate Q&A pairs
```

### Updated System Prompt

**Before:**
```python
system_prompt = """You are an expert Q&A extraction system.
Your job is to identify and extract question-answer pairs from educational notes.
Be flexible and intelligent - recognize Q&A patterns regardless of formatting.
Always respond in valid JSON format.
Be thorough and extract all Q&A pairs you can find."""
```

**After:**
```python
system_prompt = """You are an expert Q&A extraction system.
Your job is to identify and extract question-answer pairs from educational notes.
Be flexible and intelligent - recognize Q&A patterns regardless of formatting.
Always respond in valid JSON format.

CRITICAL: ONLY extract Q&A pairs where BOTH the question AND answer are present in the note.
SKIP sections like "Follow-up Questions", "Additional Questions", or any questions without answers.
Do NOT create Q&A pairs for unanswered questions - they can be mentioned in the 'followups' field for context only."""
```

## Expected Behavior After Update

### Before
```
qa_pairs_extracted=10
valid_pairs=7
skipped_pairs=3  ← Questions without answers

⚠️ WARNING: missing_en_content_in_extraction (3 warnings)
⚠️ WARNING: skipping_invalid_qa_pair (3 warnings)
```

### After
```
qa_pairs_extracted=7
valid_pairs=7
skipped_pairs=0  ← LLM now skips these during extraction

✅ No validation warnings
✅ Cleaner logs
✅ Fewer wasted tokens
```

## How It Works

1. **LLM identifies Q&A pairs** in the note content
2. **Checks if both question AND answer exist**
3. **Skips "Follow-up Questions" sections** entirely
4. **Only extracts complete Q&A pairs**
5. **Optionally includes unanswered questions** in the `followups` field for reference

## Example Note Structure

**Note with Follow-up Questions:**
```markdown
# Main Question (EN)
What is binary search?

# Answer (EN)
Binary search is an efficient algorithm...

## Follow-up Questions
- How does binary search handle duplicates?
- What is the difference between lower_bound and upper_bound?
```

**Before the update:**
- Extracts 3 Q&A pairs
- 2 valid (main question + answer)
- 1 invalid (follow-up without answer) → ⚠️ WARNING

**After the update:**
- Extracts 1 Q&A pair
- 1 valid (main question + answer)
- Follow-up questions stored in `followups` field
- No warnings ✅

## Benefits

1. ✅ **Cleaner logs** - No more validation warnings
2. ✅ **Token efficiency** - Don't waste tokens on incomplete pairs
3. ✅ **Better quality** - Only complete Q&A pairs become cards
4. ✅ **Context preserved** - Follow-ups still captured in metadata
5. ✅ **Consistent behavior** - LLM won't try to "guess" answers

## Testing

Run your sync again to verify:
```bash
obsidian-anki-sync sync --dry-run
```

You should see:
- ✅ Fewer "pairs_extracted" for notes with follow-up sections
- ✅ Zero "skipping_invalid_qa_pair" warnings
- ✅ `pairs_extracted == expected_total` (no mismatches)

## File Modified

- `src/obsidian_anki_sync/agents/qa_extractor.py`
  - Updated `_build_extraction_prompt()` method (lines 77-132)
  - Updated system prompt (lines 164-171)
