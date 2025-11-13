"""Parser-Repair agent for fixing malformed notes.

This agent activates only when rule-based parsing fails:
- Diagnoses parsing errors
- Suggests fixes for common issues
- Repairs content in-memory (doesn't modify source files)
- Re-attempts parsing with repairs
"""

import json
from pathlib import Path

from ..exceptions import ParserError
from ..models import NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .json_schemas import get_parser_repair_schema

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
    ):
        """Initialize parser-repair agent.

        Args:
            ollama_client: LLM provider instance
            model: Model to use for repair
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        logger.info("parser_repair_agent_initialized", model=model)

    def _build_repair_prompt(self, content: str, error: str) -> str:
        """Build repair prompt for the LLM.

        Args:
            content: Original note content
            error: Parser error message

        Returns:
            Formatted prompt string
        """
        return f"""<task>
Diagnose and repair an Obsidian note that failed parsing. Think step by step to identify the root cause and provide targeted fixes.
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

Step 2: Inspect the frontmatter
- Check for missing required fields: id, title, topic, language_tags, created, updated
- Validate YAML syntax (proper indentation, array notation, no quotes issues)
- Verify language_tags contains valid values: 'en', 'ru', or both

Step 3: Check Q&A structure
- Verify presence of question headers: # Question (EN), # Вопрос (RU)
- Verify presence of answer headers: ## Answer (EN), ## Ответ (RU)
- Confirm questions and answers have actual content (not empty)
- Check section ordering is logical (both RU-first and EN-first are valid)

Step 4: Validate markdown syntax
- Check for malformed headers (missing #, incorrect spacing)
- Verify proper YAML frontmatter delimiters (---)
- Look for special characters that might break parsing

Step 5: Determine repairability
- If fixable: prepare complete repaired content
- If fundamentally broken: explain why it cannot be repaired
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
7. Empty question or answer sections → flag as unrepairable if truly empty
8. Language mismatch → ensure content exists for all languages in language_tags

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
            "issue": "Specific issue found",
            "fix": "How to fix it"
        }}
    ],
    "repaired_content": "FULL repaired content including frontmatter, or null if unrepairable",
    "is_repairable": true/false
}}

is_repairable:
- true: Issues can be fixed programmatically
- false: Note is fundamentally broken (missing all content, corrupted beyond repair, etc.)

repaired_content:
- If is_repairable is true: provide the COMPLETE repaired note content
- If is_repairable is false: set to null
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

DO NOT repair (mark as unrepairable):
- Completely empty notes
- Notes with no Q&A content at all
- Corrupted files with no recognizable structure
- Notes missing both questions AND answers
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
        logger.info(
            "parser_repair_attempt",
            file=str(file_path),
            error=str(original_error),
        )

        # Read original content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("parser_repair_read_failed", file=str(file_path), error=str(e))
            raise ParserError(f"Cannot read file for repair: {e}")

        # Build repair prompt
        prompt = self._build_repair_prompt(content, str(original_error))

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

        except json.JSONDecodeError as e:
            logger.error(
                "parser_repair_invalid_json",
                file=str(file_path),
                error=str(e),
            )
            return None
        except Exception as e:
            logger.error("parser_repair_llm_failed", file=str(file_path), error=str(e))
            return None

        # Check if repairable
        if not repair_result.get("is_repairable", False):
            logger.warning(
                "parser_repair_unrepairable",
                file=str(file_path),
                diagnosis=repair_result.get("diagnosis", "Unknown"),
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
        logger.info(
            "parser_repair_applied",
            file=str(file_path),
            diagnosis=repair_result.get("diagnosis", "N/A"),
            repairs_count=len(repairs),
        )

        for repair in repairs:
            logger.debug(
                "parser_repair_detail",
                issue=repair.get("issue"),
                fix=repair.get("fix"),
            )

        # Try parsing repaired content
        # Import here to avoid circular dependency
        from ..obsidian.parser import parse_frontmatter, parse_qa_pairs

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

            logger.info(
                "parser_repair_success",
                file=str(file_path),
                qa_pairs_count=len(qa_pairs),
            )

            return metadata, qa_pairs

        except ParserError as e:
            logger.error(
                "parser_repair_reparse_failed",
                file=str(file_path),
                error=str(e),
            )
            return None


def attempt_repair(
    file_path: Path,
    original_error: Exception,
    ollama_client: BaseLLMProvider,
    model: str = "qwen3:8b",
) -> tuple[NoteMetadata, list[QAPair]] | None:
    """Helper function to attempt repair of a failed parse.

    Args:
        file_path: Path to the note file
        original_error: Original parsing error
        ollama_client: LLM provider instance
        model: Model to use for repair

    Returns:
        Tuple of (metadata, qa_pairs) if successful, None if unrepairable
    """
    agent = ParserRepairAgent(ollama_client, model)
    return agent.repair_and_parse(file_path, original_error)
