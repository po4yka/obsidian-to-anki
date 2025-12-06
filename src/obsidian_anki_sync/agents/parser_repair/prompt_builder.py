"""Prompt builders for parser repair agents."""

from __future__ import annotations

import json


def build_repair_prompt(
    content: str,
    error: str,
    enable_content_generation: bool,
    repair_missing_sections: bool,
    content_preview_limit: int = 3000,
) -> str:
    """Build repair prompt for the LLM."""
    content_gen_section = ""
    if enable_content_generation:
        content_gen_section = """
<content_generation_instructions>
When missing language sections are detected, you MUST generate the missing content with HIGH QUALITY:

1. **Translation**: If content exists in one language but not another:
   - Translate questions/answers from existing language to missing language
   - Preserve technical terms, code snippets, and formatting EXACTLY
   - Maintain the same level of detail and structure
   - Use proper grammar and clear, concise language
   - Ensure technical accuracy - verify technical terms are correctly translated
   - Maintain consistency in terminology throughout the note

2. **Inference**: If only partial content exists:
   - Infer missing questions from existing answers (or vice versa)
   - Use context from the note to generate appropriate content
   - Ensure generated content matches the style and depth of existing content
   - Maintain technical accuracy - don't invent technical details
   - Use clear, grammatically correct language

3. **Completion**: If sections are truncated or incomplete:
   - Complete truncated sections based on context
   - Fix unbalanced code fences (``` ) by adding missing closers
   - Complete incomplete markdown structures
   - Ensure completed content is grammatically correct and clear
   - Maintain technical accuracy in completed portions

4. **Quality Requirements**:
   - Generated content must be technically accurate - verify all technical claims
   - Preserve all existing content exactly as-is (no modifications)
   - Maintain markdown formatting and structure
   - Ensure bilingual consistency (same concepts, same level of detail in both languages)
   - Use proper grammar, spelling, and punctuation
   - Write clearly and concisely - avoid ambiguity
   - Maintain consistent terminology (use same technical terms in both languages)
   - Ensure code examples are syntactically correct if present

5. **Grammar and Clarity Improvements**:
   - Fix grammatical errors in existing content (if safe to do so)
   - Improve clarity without changing meaning
   - Ensure consistent verb tenses and voice
   - Use active voice when possible for clarity
   - Break up long sentences for readability

6. **Technical Accuracy Checks**:
   - Verify technical terms are used correctly
   - Ensure code examples are syntactically valid
   - Check that explanations match the technical level of the note
   - Verify consistency in technical terminology

7. **What to Generate**:
   - Missing "# Question (EN)" or "# Вопрос (RU)" sections
   - Missing "## Answer (EN)" or "## Ответ (RU)" sections
   - Incomplete answers that are truncated
   - Missing code fence closers (```)
   - Grammar and clarity improvements (when safe)

8. **What NOT to Generate**:
   - Do NOT invent new Q&A pairs that don't exist
   - Do NOT change existing content meaning
   - Do NOT generate content if the note is completely empty
   - Do NOT make up technical details that aren't inferable from context
</content_generation_instructions>
"""

    repair_constraints = []
    if enable_content_generation and repair_missing_sections:
        repair_constraints.append(
            "- Missing language sections (GENERATE by translating from existing content)"
        )
    if enable_content_generation:
        repair_constraints.extend(
            [
                "- Incomplete Q&A pairs (GENERATE missing questions/answers from context)",
                "- Unbalanced code fences (add missing ``` closers)",
                "- Truncated content (complete based on context)",
            ]
        )

    constraints_text = "\n".join(repair_constraints)

    return f"""<task>
Diagnose and repair an Obsidian note that failed parsing. Think step by step to identify the root cause and provide targeted fixes.

IMPORTANT: You must provide structured error diagnosis and quality scoring:
1. Categorize the error (syntax/structure/content/quality/frontmatter/unknown)
2. Assess severity (low/medium/high/critical)
3. Score quality BEFORE repair (completeness, structure, bilingual consistency, technical accuracy)
4. Perform repairs with priority ranking
5. Score quality AFTER repair to measure improvement
{content_gen_section if content_gen_section else ""}
</task>

<input>
<parsing_error>
{error}
</parsing_error>

<note_content>
{content[:content_preview_limit]}
</note_content>
</input>

<diagnostic_steps>
Step 1: Analyze the parsing error message
- Identify what the parser expected vs. what it found
- Determine which parsing stage failed (frontmatter, Q&A extraction, etc.)
- Categorize error type: syntax/structure/content/quality/frontmatter/unknown
- Assess severity: low (minor formatting) / medium (missing fields) / high (structural issues) / critical (unparseable)

Step 2: Score quality BEFORE repair
- Completeness: Are all required sections present? (0.0-1.0)
- Structure: Is markdown structure correct? (0.0-1.0)
- Bilingual consistency: Do both languages have equivalent content? (0.0-1.0)
- Technical accuracy: Is content technically correct? (0.0-1.0)
- Overall score: Weighted average of above scores
- List specific issues found

Step 3: Inspect the frontmatter
- Check for missing required fields: id, title, topic, language_tags, created, updated
- Validate YAML syntax (proper indentation, array notation, no quotes issues)
- Verify language_tags contains valid values: 'en', 'ru', or both
- Assign repair priority (1=highest priority, 10=lowest)

Step 4: Check Q&A structure
- Verify presence of question headers: # Question (EN), # Вопрос (RU)
- Verify presence of answer headers: ## Answer (EN), ## Ответ (RU)
- Confirm questions and answers have actual content (not empty)
- Check section ordering is logical (both RU-first and EN-first are valid)

Step 5: Validate markdown syntax
- Check for malformed headers (missing #, incorrect spacing)
- Verify proper YAML frontmatter delimiters (---)
- Look for special characters that might break parsing

Step 6: Determine repairability and prioritize repairs
- If fixable: prepare complete repaired content
- Rank repairs by priority (critical syntax errors first, then structure, then content)
- Mark which errors can be auto-fixed vs require manual intervention

Step 7: Score quality AFTER repair
- Re-assess all quality metrics after applying repairs
- Calculate improvement delta
- Verify repairs resolved the issues
</diagnostic_steps>

<common_issues>
Frontmatter issues:
1. Empty or missing language_tags → should be [en, ru] or [en] or [ru]
2. Missing required fields → add: id, title, topic, language_tags, created, updated
3. Invalid YAML syntax → fix indentation, array notation, quote issues
4. Wrong date format → use YYYY-MM-DD format

Content structure issues:
5. Missing section headers → add: # Question (EN), # Вопрос (RU), ## Answer (EN), ## Ответ (RU)
6. Incorrect header levels → Questions use #, Answers use ##
7. Empty question or answer sections → {"GENERATE missing content by translating or inferring from existing content" if enable_content_generation else "flag as unrepairable if truly empty"}
8. Language mismatch → {"GENERATE missing language sections by translating from existing content" if repair_missing_sections else "ensure content exists for all languages in language_tags"}
9. Unbalanced code fences → add missing ``` closers
10. Truncated content → complete based on context

Ordering issues:
9. Section ordering → both RU-first and EN-first are valid, just needs to be consistent
</common_issues>

<expected_format>
Valid note format (both language orderings supported):

---
id: unique-note-id
title: Note Title
topic: topic-name
language_tags: [en, ru]
created: 2025-01-01
updated: 2025-01-01
---

# Вопрос (RU) OR # Question (EN)
> Question text in Russian or English

# Question (EN) OR # Вопрос (RU)
> Question text in English or Russian

## Ответ (RU) OR ## Answer (EN)
Answer text in Russian or English

## Answer (EN) OR ## Ответ (RU)
Answer text in English or Russian
</expected_format>

<output_format>
Respond with valid JSON matching this structure:

{{
    "diagnosis": "Brief description of the root cause",
    "repairs": [
        {{
            "type": "repair_type (e.g., 'yaml_fix', 'header_fix', 'content_generation')",
            "description": "Specific issue found and how it was fixed"
        }}
    ],
    "repaired_content": "FULL repaired content including frontmatter, or null if unrepairable",
    "content_generation_applied": true/false,
    "generated_sections": [
        {{
            "section_type": "question_en|question_ru|answer_en|answer_ru",
            "method": "translation|inference|completion",
            "description": "What was generated and how"
        }}
    ],
    "error_diagnosis": {{
        "error_category": "syntax|structure|content|quality|frontmatter|unknown",
        "severity": "low|medium|high|critical",
        "error_description": "Detailed description of the error",
        "repair_priority": 1-10,
        "can_auto_fix": true/false
    }},
    "quality_before": {{
        "completeness_score": 0.0-1.0,
        "structure_score": 0.0-1.0,
        "bilingual_consistency": 0.0-1.0,
        "technical_accuracy": 0.0-1.0,
        "overall_score": 0.0-1.0,
        "issues_found": ["issue1", "issue2", ...]
    }},
    "quality_after": {{
        "completeness_score": 0.0-1.0,
        "structure_score": 0.0-1.0,
        "bilingual_consistency": 0.0-1.0,
        "technical_accuracy": 0.0-1.0,
        "overall_score": 0.0-1.0,
        "issues_found": ["remaining_issue1", ...]
    }},
    "is_repairable": true/false,
    "repair_time": 0.0
}}

is_repairable:
- true: Issues can be fixed programmatically (including content generation)
- false: Note is fundamentally broken (missing all content, corrupted beyond repair, etc.)

repaired_content:
- If is_repairable is true: provide the COMPLETE repaired note content with all missing sections generated
- If is_repairable is false: set to null

content_generation_applied:
- true: If you generated any missing content (translated, inferred, or completed sections)
- false: If you only fixed structure/formatting without generating new content

generated_sections:
- List all sections you generated (not just repaired)
- Include method used: "translation" (translated from existing language), "inference" (inferred from context), "completion" (completed truncated content)
</output_format>

<examples>
<example_1>
Input: Missing language_tags field

Diagnosis: "Frontmatter missing required language_tags field"

Output:
{{
    "diagnosis": "Frontmatter missing required language_tags field",
    "repairs": [
        {{
            "issue": "Missing language_tags in frontmatter",
            "fix": "Add language_tags: [en, ru] based on detected content"
        }}
    ],
    "repaired_content": "---\\nid: note-1\\ntitle: Example\\ntopic: example\\nlanguage_tags: [en, ru]\\ncreated: 2025-01-01\\nupdated: 2025-01-01\\n---\\n\\n# Question (EN)\\n...",
    "is_repairable": true
}}
</example_1>

<example_2>
Input: Malformed YAML frontmatter

Diagnosis: "Invalid YAML syntax: improper array notation"

Output:
{{
    "diagnosis": "Invalid YAML syntax: improper array notation for language_tags",
    "repairs": [
        {{
            "issue": "language_tags uses (en, ru) instead of [en, ru]",
            "fix": "Change language_tags: (en, ru) to language_tags: [en, ru]"
        }}
    ],
    "repaired_content": "---\\nid: note-1\\ntitle: Example\\nlanguage_tags: [en, ru]\\n...",
    "is_repairable": true
}}
</example_2>

<example_3>
Input: Completely empty file

Output:
{{
    "diagnosis": "Note is completely empty with no content",
    "repairs": [],
    "repaired_content": null,
    "is_repairable": false
}}
</example_3>
</examples>

<constraints>
DO repair:
- Missing frontmatter fields (can infer reasonable defaults)
- YAML syntax errors
- Malformed headers
- Minor formatting issues
{constraints_text}

DO NOT repair (mark as unrepairable):
- Completely empty notes
- Notes with no Q&A content at all
- Corrupted files with no recognizable structure
- Notes missing both questions AND answers (unless content generation can infer them from context)
</constraints>"""


REPAIR_SYSTEM_PROMPT = """<role>
You are a diagnostic and repair agent for Obsidian educational notes. Your expertise includes YAML syntax, markdown structure, parsing error analysis, and programmatic content repair.
</role>

<approach>
Think step by step through the diagnostic process:
1. Carefully analyze the parsing error message to understand what failed
2. Inspect the note content to identify the root cause
3. Determine if the issue is fixable programmatically
4. If fixable, construct the minimal repair needed
5. Provide the complete repaired content

Be methodical and thorough in your diagnosis.
</approach>

<repair_philosophy>
- Fix only what is broken, preserve all valid content
- Infer reasonable defaults for missing required fields
- Be conservative: mark as unrepairable if content is fundamentally missing
- Provide clear explanations of what was fixed and why
</repair_philosophy>

<output_requirements>
- Always respond in valid JSON format with the exact structure requested
- Provide specific, actionable diagnosis of the problem
- List all repairs applied in the repairs array
- Include COMPLETE repaired content (no truncation) if repairable
- Set is_repairable to false only when truly unrepairable
</output_requirements>

<constraints>
NEVER repair by:
- Inventing question or answer content that doesn't exist
- Making up frontmatter values that can't be inferred
- Drastically changing the structure or meaning

ALWAYS:
- Preserve all existing content and semantic meaning
- Only fix syntax, structure, and format issues
- Provide complete repaired content, not partial fixes
</constraints>"""


def build_proactive_analysis_prompt(content: str, preview_limit: int = 3000) -> str:
    """Build prompt for proactive analysis."""
    return f"""<task>
Analyze this Obsidian note for quality issues and potential parsing problems BEFORE parsing occurs.
Identify issues proactively to prevent parse failures.

Think step by step:
1. Check frontmatter structure and required fields
2. Verify markdown structure (headers, sections, formatting)
3. Check for missing language sections (EN/RU)
4. Identify grammar and clarity issues
5. Detect incomplete or truncated content
6. Assess bilingual consistency
7. Score overall quality
</task>

<note_content>
{content[:preview_limit]}
</note_content>

<analysis_requirements>
Provide a comprehensive quality analysis:
- Overall quality score (0.0-1.0)
- List of issues found (be specific)
- Whether correction is needed
- Suggested corrections
- Confidence in analysis (0.0-1.0)
</analysis_requirements>

<output_format>
Respond with valid JSON:
{{
    "quality_score": 0.0-1.0,
    "issues_found": ["issue1", "issue2", ...],
    "needs_correction": true/false,
    "suggested_corrections": ["correction1", "correction2", ...],
    "confidence": 0.0-1.0
}}
</output_format>"""


PROACTIVE_ANALYSIS_SYSTEM_PROMPT = """<role>
You are a proactive quality analysis agent for Obsidian educational notes. Your goal is to identify and fix issues BEFORE they cause parsing failures.
</role>

<approach>
1. Analyze the note structure and content
2. Identify any issues that might cause parsing errors or reduce quality
3. Determine if corrections are needed
4. Provide specific, actionable suggestions
</approach>"""


def build_proactive_correction_prompt(
    content: str,
    issues_found: list[str],
    suggested_corrections: list[str],
) -> str:
    """Build prompt for proactive correction."""
    return f"""<task>
Apply corrections to this Obsidian note based on the identified issues.
Fix the issues while preserving the original meaning and structure.

Issues to fix:
{json.dumps(issues_found, indent=2)}

Suggested corrections:
{json.dumps(suggested_corrections, indent=2)}
</task>

<note_content>
{content}
</note_content>

<output_format>
Respond with valid JSON:
{{
    "corrected_content": "FULL corrected note content",
    "corrections_applied": ["correction1", "correction2", ...],
    "quality_after": {{
        "overall_score": 0.0-1.0,
        "issues_remaining": []
    }}
}}
</output_format>"""


PROACTIVE_CORRECTION_SYSTEM_PROMPT = """<role>
You are a proactive correction agent. Fix quality issues in this note before parsing.
Apply corrections conservatively - only fix real issues, preserve all valid content.
</role>"""

