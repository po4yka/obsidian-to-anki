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
        return f"""You are a note repair agent. Analyze this Obsidian note that failed parsing and suggest fixes.

PARSING ERROR:
{error}

NOTE CONTENT:
{content[:3000]}

COMMON ISSUES TO CHECK:
1. Empty or missing language_tags (should be [en, ru] if both languages present)
2. Missing required frontmatter fields: id, title, topic, language_tags, created, updated
3. Invalid YAML syntax in frontmatter
4. Missing section headers: # Question (EN), # �оп�ос (RU), ## Answer (EN), ## Ответ (RU)
5. Incorrect section ordering (both RU-first and EN-first are valid)
6. Missing content in question or answer sections

EXPECTED FORMAT (both orderings supported):
---
id: note-id
title: Note Title
topic: topic-name
language_tags: [en, ru]
created: 2025-01-01
updated: 2025-01-01
---

# �оп�ос (RU) OR # Question (EN)
> Question text

# Question (EN) OR # �оп�ос (RU)
> Question text

## Ответ (RU) OR ## Answer (EN)
Answer text

## Answer (EN) OR ## Ответ (RU)
Answer text

RESPOND IN JSON FORMAT:
{{
    "diagnosis": "Brief description of what's wrong",
    "repairs": [
        {{
            "issue": "Description of issue",
            "fix": "What to fix"
        }}
    ],
    "repaired_content": "Full repaired content (only if fixable, otherwise null)",
    "is_repairable": true/false
}}

If the note is fundamentally broken (missing all content, corrupted, etc.), set is_repairable to false.
If repairable, provide the FULL repaired content including frontmatter.
"""

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
        system_prompt = """You are a note repair agent for Obsidian notes.
Analyze structural and formatting issues in notes and provide fixes when possible.
Always respond in valid JSON format with the exact structure requested."""

        # Call LLM for repair analysis
        try:
            repair_result = self.ollama_client.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
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
            qa_pairs = parse_qa_pairs(temp_content_for_parse, metadata)

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
