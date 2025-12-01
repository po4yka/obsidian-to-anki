"""Parser-Repair agent for fixing malformed notes.

This agent activates only when rule-based parsing fails:
- Diagnoses parsing errors
- Suggests fixes for common issues
- Repairs content in-memory (doesn't modify source files)
- Re-attempts parsing with repairs
"""

import contextlib
import json
import re
from pathlib import Path

from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

from .json_schemas import get_parser_repair_schema
from .langgraph.retry_policies import classify_error_category, select_repair_strategy
from .models import (
    NoteCorrectionResult,
    PartialRepairResult,
    RepairDiagnosis,
    RepairQualityScore,
)
from .repair_learning import get_repair_learning_system
from .repair_metrics import get_repair_metrics_collector

logger = get_logger(__name__)


class ParserRepairAgent:
    """Agent for repairing malformed notes that fail parsing.

    Uses lightweight model (qwen3:8b) for fast analysis and repair.
    Only invoked when rule-based parser fails.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:8b",
        temperature: float = 0.0,
        enable_content_generation: bool = True,
        repair_missing_sections: bool = True,
    ):
        """Initialize parser-repair agent.

        Args:
            ollama_client: LLM provider instance
            model: Model to use for repair
            temperature: Sampling temperature (0.0 for deterministic)
            enable_content_generation: Allow LLM to generate missing content
            repair_missing_sections: Generate missing language sections
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.enable_content_generation = enable_content_generation
        self.repair_missing_sections = repair_missing_sections
        logger.info(
            "parser_repair_agent_initialized",
            model=model,
            enable_content_generation=enable_content_generation,
            repair_missing_sections=repair_missing_sections,
        )

    def _build_repair_prompt(
        self, content: str, error: str, enable_content_gen: bool = True
    ) -> str:
        """Build repair prompt for the LLM.

        Args:
            content: Original note content
            error: Parser error message
            enable_content_gen: Whether to enable content generation instructions

        Returns:
            Formatted prompt string
        """
        content_gen_section = ""
        if enable_content_gen and self.enable_content_generation:
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
   - Fix unbalanced code fences (```) by adding missing closers
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
{content[:3000]}
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
7. Empty question or answer sections → {"GENERATE missing content by translating or inferring from existing content" if self.enable_content_generation else "flag as unrepairable if truly empty"}
8. Language mismatch → {"GENERATE missing language sections by translating from existing content" if self.repair_missing_sections else "ensure content exists for all languages in language_tags"}
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
{"- Missing language sections (GENERATE by translating from existing content)" if self.enable_content_generation and self.repair_missing_sections else ""}
{"- Incomplete Q&A pairs (GENERATE missing questions/answers from context)" if self.enable_content_generation else ""}
{"- Unbalanced code fences (add missing ``` closers)" if self.enable_content_generation else ""}
{"- Truncated content (complete based on context)" if self.enable_content_generation else ""}

DO NOT repair (mark as unrepairable):
- Completely empty notes
- Notes with no Q&A content at all
- Corrupted files with no recognizable structure
- Notes missing both questions AND answers (unless content generation can infer them from context)
</constraints>"""

    def repair_and_parse(
        self, file_path: Path, original_error: Exception
    ) -> tuple[NoteMetadata, list[QAPair]] | None:
        """Attempt to repair a note that failed parsing.

        Args:
            file_path: Path to the note file
            original_error: Original parsing error

        Returns:
            Tuple of (metadata, qa_pairs) if successful, None if unrepairable

        Raises:
            ParserError: If repair also fails
        """
        import time

        start_time = time.time()
        metrics_collector = get_repair_metrics_collector()

        # Classify error for metrics
        error_category = classify_error_category(original_error)

        # Try to get suggested strategy from learning system
        learning_system = get_repair_learning_system()
        suggested_strategy = learning_system.suggest_strategy(
            error_category=error_category,
            error_type=type(original_error).__name__,
            error_message=str(original_error),
        )

        # Use suggested strategy if available, otherwise use default selection
        if suggested_strategy:
            from .models import RepairStrategy

            repair_strategy = RepairStrategy(
                strategy_type=suggested_strategy,
                priority=1,  # Learned strategies get high priority
                stages=[suggested_strategy],
                confidence_threshold=0.7,
            )
            logger.info(
                "repair_strategy_from_learning",
                category=error_category,
                strategy=suggested_strategy,
            )
        else:
            repair_strategy = select_repair_strategy(original_error)

        strategy_type = repair_strategy.strategy_type

        logger.info(
            "parser_repair_attempt",
            file=str(file_path),
            error=str(original_error),
            category=error_category,
            strategy=strategy_type,
        )

        # Read original content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("parser_repair_read_failed", file=str(file_path), error=str(e))
            msg = f"Cannot read file for repair: {e}"
            raise ParserError(msg) from e

        # Build repair prompt
        prompt = self._build_repair_prompt(
            content,
            str(original_error),
            enable_content_gen=self.enable_content_generation,
        )

        # System prompt for note repair agent
        system_prompt = """<role>
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

        # Call LLM for repair analysis
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get JSON schema for structured output
                json_schema = get_parser_repair_schema()

                repair_result = self.ollama_client.generate_json(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    temperature=self.temperature,
                    json_schema=json_schema,
                )
                break  # Success, exit retry loop

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "parser_repair_json_retry",
                        file=str(file_path),
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    continue
                logger.error(
                    "parser_repair_invalid_json",
                    file=str(file_path),
                    error=str(e),
                )
                return None
            except Exception as e:
                logger.error(
                    "parser_repair_llm_failed", file=str(file_path), error=str(e)
                )
                return None

        # Check if repairable
        if not repair_result.get("is_repairable", False):
            repair_time = time.time() - start_time
            logger.warning(
                "parser_repair_unrepairable",
                file=str(file_path),
                diagnosis=repair_result.get("diagnosis", "Unknown"),
            )
            # Record failed attempt
            metrics_collector.record_attempt(
                error_category=error_category,
                error_type=type(original_error).__name__,
                strategy_used=strategy_type,
                success=False,
                repair_time=repair_time,
                error_message=str(original_error),
            )
            return None

        # Get repaired content
        repaired_content = repair_result.get("repaired_content")
        if not repaired_content:
            logger.warning(
                "parser_repair_no_content",
                file=str(file_path),
            )
            return None

        # Log repairs applied
        repairs = repair_result.get("repairs", [])
        content_generation_applied = repair_result.get(
            "content_generation_applied", False
        )
        generated_sections = repair_result.get("generated_sections", [])

        # Parse error diagnosis if present
        error_diagnosis = None
        error_diag_data = repair_result.get("error_diagnosis")
        if error_diag_data:
            try:
                error_diagnosis = RepairDiagnosis(**error_diag_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_diagnosis",
                    file=str(file_path),
                    error=str(e),
                )

        # Parse quality scores if present
        quality_before = None
        quality_after = None
        quality_before_data = repair_result.get("quality_before")
        quality_after_data = repair_result.get("quality_after")

        if quality_before_data:
            try:
                quality_before = RepairQualityScore(**quality_before_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_quality_before",
                    file=str(file_path),
                    error=str(e),
                )

        if quality_after_data:
            try:
                quality_after = RepairQualityScore(**quality_after_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_quality_after",
                    file=str(file_path),
                    error=str(e),
                )

        logger.info(
            "parser_repair_applied",
            file=str(file_path),
            diagnosis=repair_result.get("diagnosis", "N/A"),
            repairs_count=len(repairs),
            content_generation_applied=content_generation_applied,
            generated_sections_count=len(generated_sections),
            error_category=error_diagnosis.error_category if error_diagnosis else None,
            severity=error_diagnosis.severity if error_diagnosis else None,
            quality_before=quality_before.overall_score if quality_before else None,
            quality_after=quality_after.overall_score if quality_after else None,
        )

        for repair in repairs:
            logger.debug(
                "parser_repair_detail",
                repair_type=repair.get("type", "unknown"),
                description=repair.get("description", ""),
            )

        for gen_section in generated_sections:
            logger.info(
                "parser_repair_content_generated",
                file=str(file_path),
                section_type=gen_section.get("section_type", "unknown"),
                method=gen_section.get("method", "unknown"),
                description=gen_section.get("description", ""),
            )

            # Validate repaired content against APF/Obsidian requirements
            validation_errors = self._validate_repaired_content(
                repaired_content, file_path
            )
            if validation_errors:
                logger.warning(
                    "parser_repair_validation_warnings",
                    file=str(file_path),
                    errors=validation_errors,
                )

            # Try parsing repaired content
            # Import here to avoid circular dependency
            from obsidian_anki_sync.obsidian.parser import (
                parse_frontmatter,
                parse_qa_pairs,
            )

            try:
                # Write to temporary path for parsing
                temp_content_for_parse = repaired_content

                # Parse frontmatter from repaired content
                metadata = parse_frontmatter(temp_content_for_parse, file_path)

                # Parse Q/A pairs from repaired content
                qa_pairs = parse_qa_pairs(temp_content_for_parse, metadata, file_path)

                if not qa_pairs:
                    logger.warning(
                        "parser_repair_no_qa_pairs",
                        file=str(file_path),
                    )
                    return None

                repair_time = time.time() - start_time
                quality_improvement = (
                    quality_after.overall_score - quality_before.overall_score
                    if quality_before and quality_after
                    else None
                )

                logger.info(
                    "parser_repair_success",
                    file=str(file_path),
                    qa_pairs_count=len(qa_pairs),
                    quality_improvement=quality_improvement,
                )

                # Record successful repair
                metrics_collector.record_attempt(
                    error_category=error_category,
                    error_type=type(original_error).__name__,
                    strategy_used=strategy_type,
                    success=True,
                    quality_before=(
                        quality_before.overall_score if quality_before else None
                    ),
                    quality_after=(
                        quality_after.overall_score if quality_after else None
                    ),
                    repair_time=repair_time,
                )

                # Learn from successful repair
                quality_improvement = (
                    quality_after.overall_score - quality_before.overall_score
                    if quality_before and quality_after
                    else None
                )
                learning_system.learn_from_success(
                    error_category=error_category,
                    error_type=type(original_error).__name__,
                    error_message=str(original_error),
                    strategy_used=strategy_type,
                    quality_improvement=quality_improvement,
                    repair_steps=[repair.get("type", "") for repair in repairs],
                )

                return metadata, qa_pairs

            except ParserError as e:
                repair_time = time.time() - start_time
                logger.error(
                    "parser_repair_reparse_failed",
                    file=str(file_path),
                    error=str(e),
                )
                # Record failed attempt
                metrics_collector.record_attempt(
                    error_category=error_category,
                    error_type=type(original_error).__name__,
                    strategy_used=strategy_type,
                    success=False,
                    repair_time=repair_time,
                    error_message=str(e),
                )
                return None
        return None

    async def analyze_and_correct_proactively_async(
        self, content: str, file_path: Path | None = None
    ) -> NoteCorrectionResult:
        """Proactively analyze note quality and apply corrections before parsing asynchronously.

        This method analyzes notes for common issues before they cause parsing failures,
        enabling early correction and better quality.

        Args:
            content: Note content to analyze
            file_path: Optional file path for context

        Returns:
            NoteCorrectionResult with analysis and corrections
        """
        import time

        start_time = time.time()
        logger.info(
            "proactive_note_analysis_async_start",
            file=str(file_path) if file_path else "unknown",
        )

        # Build proactive analysis prompt
        analysis_prompt = f"""<task>
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
{content[:3000]}
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

        system_prompt = """<role>
You are a proactive quality analysis agent for Obsidian educational notes. Your goal is to identify and fix issues BEFORE they cause parsing failures.
</role>

<approach>
1. Analyze the note structure and content
2. Identify any issues that might cause parsing errors or reduce quality
3. Determine if corrections are needed
4. Provide specific, actionable suggestions
</approach>"""

        try:
            # Call LLM for analysis
            analysis_result = await self.ollama_client.generate_json_async(
                model=self.model,
                prompt=analysis_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )

            needs_correction = analysis_result.get("needs_correction", False)
            quality_score = analysis_result.get("quality_score", 0.0)
            issues_found = analysis_result.get("issues_found", [])
            confidence = analysis_result.get("confidence", 0.0)

            # If correction is needed, apply it
            corrected_content = None
            corrections_applied = []
            quality_after = None

            if needs_correction and self.enable_content_generation:
                logger.info(
                    "proactive_correction_needed_async",
                    file=str(file_path),
                    issues=issues_found,
                )

                # Build correction prompt
                correction_prompt = f"""<task>
Apply corrections to this Obsidian note based on the identified issues.
Fix the issues while preserving the original meaning and structure.

Issues to fix:
{json.dumps(issues_found, indent=2)}

Suggested corrections:
{json.dumps(analysis_result.get("suggested_corrections", []), indent=2)}
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

                # Call LLM for correction
                correction_result_data = await self.ollama_client.generate_json_async(
                    model=self.model,
                    prompt=correction_prompt,
                    system=system_prompt,
                    temperature=self.temperature,
                )

                corrected_content = correction_result_data.get("corrected_content")
                corrections_applied = correction_result_data.get(
                    "corrections_applied", []
                )
                quality_after_data = correction_result_data.get("quality_after")

                if quality_after_data:
                    with contextlib.suppress(Exception):
                        quality_after = RepairQualityScore(**quality_after_data)

            correction_time = time.time() - start_time

            return NoteCorrectionResult(
                needs_correction=needs_correction,
                quality_score=quality_score,
                issues_found=issues_found,
                corrections_applied=corrections_applied,
                corrected_content=corrected_content,
                quality_after=quality_after,
                confidence=confidence,
                correction_time=correction_time,
            )

        except Exception as e:
            logger.error(
                "proactive_analysis_async_failed",
                file=str(file_path),
                error=str(e),
            )
            return NoteCorrectionResult(
                needs_correction=False,
                quality_score=0.0,
                issues_found=[f"Analysis failed: {e!s}"],
                corrections_applied=[],
                confidence=0.0,
                correction_time=time.time() - start_time,
            )

    def analyze_and_correct_proactively(
        self, content: str, file_path: Path | None = None
    ) -> NoteCorrectionResult:
        """Proactively analyze note quality and apply corrections before parsing.

        This method analyzes notes for common issues before they cause parsing failures,
        enabling early correction and better quality.

        Args:
            content: Note content to analyze
            file_path: Optional file path for context

        Returns:
            NoteCorrectionResult with analysis and corrections
        """
        import time

        start_time = time.time()
        logger.info(
            "proactive_note_analysis_start",
            file=str(file_path) if file_path else "unknown",
        )

        # Build proactive analysis prompt
        analysis_prompt = f"""<task>
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
{content[:3000]}
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

        system_prompt = """<role>
You are a proactive quality analysis agent for Obsidian educational notes. Your goal is to identify and fix issues BEFORE they cause parsing failures.
</role>

<approach>
1. Analyze note structure and content comprehensively
2. Identify potential parsing issues early
3. Suggest specific, actionable corrections
4. Score quality objectively
5. Be conservative: only flag real issues
</approach>

<quality_dimensions>
- Completeness: Are all required sections present?
- Structure: Is markdown structure correct?
- Bilingual consistency: Do both languages have equivalent content?
- Technical accuracy: Is content technically correct?
- Grammar and clarity: Is content well-written?
</quality_dimensions>"""

        try:
            # Get JSON schema for structured output
            json_schema = {
                "type": "object",
                "properties": {
                    "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "issues_found": {"type": "array", "items": {"type": "string"}},
                    "needs_correction": {"type": "boolean"},
                    "suggested_corrections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": [
                    "quality_score",
                    "issues_found",
                    "needs_correction",
                    "confidence",
                ],
            }

            analysis_result = self.ollama_client.generate_json(
                model=self.model,
                prompt=analysis_prompt,
                system=system_prompt,
                temperature=self.temperature,
                json_schema=json_schema,
            )

            quality_score = analysis_result.get("quality_score", 0.5)
            issues_found = analysis_result.get("issues_found", [])
            needs_correction = analysis_result.get("needs_correction", False)
            suggested_corrections = analysis_result.get("suggested_corrections", [])
            confidence = analysis_result.get("confidence", 0.5)

            corrected_content = None
            corrections_applied = []
            quality_after = None

            # If correction is needed and we have suggestions, attempt repair
            if needs_correction and suggested_corrections:
                logger.info(
                    "proactive_correction_needed",
                    issues_count=len(issues_found),
                    suggestions_count=len(suggested_corrections),
                )

                # Create a synthetic error for repair
                synthetic_error = f"Proactive correction needed. Issues: {', '.join(issues_found[:3])}"

                # Attempt repair using existing repair logic
                repair_prompt = self._build_repair_prompt(
                    content,
                    synthetic_error,
                    enable_content_gen=self.enable_content_generation,
                )

                repair_system_prompt = """<role>
You are a proactive correction agent. Fix quality issues in this note before parsing.
Apply corrections conservatively - only fix real issues, preserve all valid content.
</role>"""

                try:
                    repair_json_schema = get_parser_repair_schema()
                    repair_result = self.ollama_client.generate_json(
                        model=self.model,
                        prompt=repair_prompt,
                        system=repair_system_prompt,
                        temperature=self.temperature,
                        json_schema=repair_json_schema,
                    )

                    if repair_result.get("is_repairable", False):
                        corrected_content = repair_result.get("repaired_content")
                        repairs = repair_result.get("repairs", [])
                        corrections_applied = [
                            repair.get("description", "") for repair in repairs
                        ]

                        # Extract quality after if available
                        quality_after_data = repair_result.get("quality_after")
                        if quality_after_data:
                            with contextlib.suppress(Exception):
                                quality_after = RepairQualityScore(**quality_after_data)

                        logger.info(
                            "proactive_correction_applied",
                            corrections_count=len(corrections_applied),
                        )
                    else:
                        logger.info("proactive_correction_not_repairable")
                except Exception as e:
                    logger.warning("proactive_repair_failed", error=str(e))

            correction_time = time.time() - start_time

            result = NoteCorrectionResult(
                needs_correction=needs_correction,
                corrected_content=corrected_content,
                quality_score=quality_score,
                issues_found=issues_found,
                corrections_applied=corrections_applied,
                confidence=confidence,
                correction_time=correction_time,
                quality_after=quality_after,
            )

            logger.info(
                "proactive_note_analysis_complete",
                needs_correction=needs_correction,
                quality_score=quality_score,
                issues_count=len(issues_found),
                corrections_applied=len(corrections_applied),
                time=correction_time,
            )

            return result

        except Exception as e:
            logger.error("proactive_analysis_failed", error=str(e))
            # Return permissive result on error
            return NoteCorrectionResult(
                needs_correction=False,
                quality_score=0.5,
                issues_found=[f"Analysis failed: {e!s}"],
                corrections_applied=[],
                confidence=0.0,
                correction_time=time.time() - start_time,
            )

    def _validate_repaired_content(
        self, content: str, file_path: Path | None = None
    ) -> list[str]:
        """Validate repaired content against APF and Obsidian requirements.

        Args:
            content: Repaired content to validate
            file_path: Optional file path for context

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check Obsidian markdown conventions
        obsidian_errors = self._validate_obsidian_markdown(content)
        errors.extend(obsidian_errors)

        # Check bilingual consistency if applicable
        bilingual_errors = self._validate_bilingual_consistency(content)
        errors.extend(bilingual_errors)

        # Check frontmatter structure
        frontmatter_errors = self._validate_frontmatter_structure(content)
        errors.extend(frontmatter_errors)

        return errors

    def _validate_obsidian_markdown(self, content: str) -> list[str]:
        """Validate Obsidian markdown conventions.

        Args:
            content: Content to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Check for proper YAML frontmatter delimiters
        if not content.strip().startswith("---"):
            errors.append("Missing YAML frontmatter start delimiter (---)")

        # Count frontmatter delimiters
        delimiter_count = content.count("---")
        if delimiter_count < 2:
            errors.append("Missing YAML frontmatter end delimiter (---)")

        # Check for proper header levels (should use # for questions, ## for answers)
        lines = content.split("\n")
        has_question_header = False
        has_answer_header = False

        for line in lines:
            if line.strip().startswith("# Question") or line.strip().startswith(
                "# Вопрос"
            ):
                has_question_header = True
            if line.strip().startswith("## Answer") or line.strip().startswith(
                "## Ответ"
            ):
                has_answer_header = True

        if not has_question_header and not has_answer_header:
            # Only warn if content seems to be a note (has frontmatter)
            if "---" in content:
                errors.append("Missing question/answer headers")

        return errors

    def _validate_bilingual_consistency(self, content: str) -> list[str]:
        """Validate bilingual consistency (EN/RU).

        Args:
            content: Content to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Check if both languages are present
        has_en_question = "# Question (EN)" in content or "# Question" in content
        has_ru_question = "# Вопрос (RU)" in content or "# Вопрос" in content
        has_en_answer = "## Answer (EN)" in content or "## Answer" in content
        has_ru_answer = "## Ответ (RU)" in content or "## Ответ" in content

        # Extract language tags from frontmatter
        lang_tags_match = re.search(r"language_tags:\s*\[(.*?)\]", content)
        if lang_tags_match:
            lang_tags_str = lang_tags_match.group(1)
            lang_tags = [
                tag.strip().strip('"').strip("'") for tag in lang_tags_str.split(",")
            ]

            # Check consistency
            if "en" in lang_tags and not (has_en_question and has_en_answer):
                errors.append("Language tag 'en' present but EN content missing")

            if "ru" in lang_tags and not (has_ru_question and has_ru_answer):
                errors.append("Language tag 'ru' present but RU content missing")

        return errors

    def _validate_frontmatter_structure(self, content: str) -> list[str]:
        """Validate frontmatter structure.

        Args:
            content: Content to validate

        Returns:
            List of validation errors
        """
        errors = []

        # Extract frontmatter
        frontmatter_match = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            errors.append("Invalid or missing YAML frontmatter")
            return errors

        frontmatter = frontmatter_match.group(1)

        # Check required fields
        required_fields = [
            "id",
            "title",
            "topic",
            "language_tags",
            "created",
            "updated",
        ]
        for field in required_fields:
            if f"{field}:" not in frontmatter:
                errors.append(f"Missing required frontmatter field: {field}")

        # Validate language_tags format
        lang_tags_match = re.search(r"language_tags:\s*\[(.*?)\]", frontmatter)
        if lang_tags_match:
            lang_tags_str = lang_tags_match.group(1)
            lang_tags = [
                tag.strip().strip('"').strip("'") for tag in lang_tags_str.split(",")
            ]
            valid_langs = {"en", "ru"}
            for tag in lang_tags:
                if tag not in valid_langs:
                    errors.append(f"Invalid language tag: {tag} (must be 'en' or 'ru')")

        return errors

    def attempt_partial_repair(
        self, file_path: Path, original_error: Exception
    ) -> PartialRepairResult | None:
        """Attempt partial repair - fix what can be fixed, flag what cannot.

        Args:
            file_path: Path to the note file
            original_error: Original parsing error

        Returns:
            PartialRepairResult with repair status per section, or None if completely unrepairable
        """
        import time

        start_time = time.time()
        logger.info(
            "parser_partial_repair_attempt",
            file=str(file_path),
            error=str(original_error),
        )

        # Read original content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(
                "parser_partial_repair_read_failed", file=str(file_path), error=str(e)
            )
            return None

        # Identify sections
        sections = self._identify_sections(content)
        repaired_content = content
        sections_fixed = []
        sections_failed = []
        section_confidence: dict[str, float] = {}
        repair_report_parts = []

        # Try to repair each section
        for section_name, section_content in sections.items():
            try:
                # Attempt repair for this section
                section_error = f"{original_error} (section: {section_name})"
                repair_prompt = self._build_repair_prompt(
                    section_content,
                    str(section_error),
                    enable_content_gen=self.enable_content_generation,
                )

                repair_json_schema = get_parser_repair_schema()
                repair_result = self.ollama_client.generate_json(
                    model=self.model,
                    prompt=repair_prompt,
                    system="""<role>
You are a partial repair agent. Fix only the section you're given.
If the section cannot be fixed, mark it as unrepairable.
</role>""",
                    temperature=self.temperature,
                    json_schema=repair_json_schema,
                )

                if repair_result.get("is_repairable", False):
                    repaired_section = repair_result.get("repaired_content")
                    if repaired_section:
                        # Replace section in content
                        repaired_content = repaired_content.replace(
                            section_content, repaired_section, 1
                        )
                        sections_fixed.append(section_name)
                        confidence = repair_result.get("quality_after", {}).get(
                            "overall_score", 0.7
                        )
                        section_confidence[section_name] = confidence
                        repair_report_parts.append(
                            f"Section '{section_name}': Fixed (confidence: {confidence:.2f})"
                        )
                    else:
                        sections_failed.append(section_name)
                        section_confidence[section_name] = 0.0
                        repair_report_parts.append(
                            f"Section '{section_name}': Failed (no repaired content)"
                        )
                else:
                    sections_failed.append(section_name)
                    section_confidence[section_name] = 0.0
                    repair_report_parts.append(
                        f"Section '{section_name}': Unrepairable"
                    )

            except Exception as e:
                logger.warning(
                    "partial_repair_section_failed",
                    section=section_name,
                    error=str(e),
                )
                sections_failed.append(section_name)
                section_confidence[section_name] = 0.0
                repair_report_parts.append(f"Section '{section_name}': Error - {e!s}")

        is_complete = len(sections_failed) == 0
        repair_report = "\n".join(repair_report_parts)

        result = PartialRepairResult(
            repaired_content=repaired_content,
            sections_fixed=sections_fixed,
            sections_failed=sections_failed,
            section_confidence=section_confidence,
            is_complete=is_complete,
            repair_report=repair_report,
        )

        logger.info(
            "partial_repair_complete",
            file=str(file_path),
            sections_fixed=len(sections_fixed),
            sections_failed=len(sections_failed),
            is_complete=is_complete,
            time=time.time() - start_time,
        )

        return result

    def _identify_sections(self, content: str) -> dict[str, str]:
        """Identify sections in note content.

        Args:
            content: Note content

        Returns:
            Dictionary mapping section names to content
        """
        sections: dict[str, str] = {}
        lines = content.split("\n")

        # Extract frontmatter
        frontmatter_start = None
        frontmatter_end = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                if frontmatter_start is None:
                    frontmatter_start = i
                else:
                    frontmatter_end = i
                    break

        if frontmatter_start is not None and frontmatter_end is not None:
            sections["frontmatter"] = "\n".join(
                lines[frontmatter_start : frontmatter_end + 1]
            )

        # Extract question/answer sections
        current_section = None
        current_section_lines = []

        for line in lines:
            if line.strip().startswith("# Question") or line.strip().startswith(
                "# Вопрос"
            ):
                if current_section:
                    sections[current_section] = "\n".join(current_section_lines)
                current_section = "question_" + str(
                    len([k for k in sections if k.startswith("question")])
                )
                current_section_lines = [line]
            elif line.strip().startswith("## Answer") or line.strip().startswith(
                "## Ответ"
            ):
                if current_section:
                    sections[current_section] = "\n".join(current_section_lines)
                current_section = "answer_" + str(
                    len([k for k in sections if k.startswith("answer")])
                )
                current_section_lines = [line]
            elif current_section:
                current_section_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_section_lines)

        # If no sections found, treat entire content as one section
        if not sections:
            sections["content"] = content

        return sections


def attempt_repair(
    file_path: Path,
    original_error: Exception,
    ollama_client: BaseLLMProvider,
    model: str = "qwen3:8b",
    enable_content_generation: bool = True,
    repair_missing_sections: bool = True,
) -> tuple[NoteMetadata, list[QAPair]] | None:
    """Helper function to attempt repair of a failed parse.

    Args:
        file_path: Path to the note file
        original_error: Original parsing error
        ollama_client: LLM provider instance
        model: Model to use for repair
        enable_content_generation: Allow LLM to generate missing content
        repair_missing_sections: Generate missing language sections

    Returns:
        Tuple of (metadata, qa_pairs) if successful, None if unrepairable
    """
    agent = ParserRepairAgent(
        ollama_client,
        model,
        enable_content_generation=enable_content_generation,
        repair_missing_sections=repair_missing_sections,
    )
    return agent.repair_and_parse(file_path, original_error)
