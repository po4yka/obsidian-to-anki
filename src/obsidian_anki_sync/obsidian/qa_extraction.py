"""Regex-based Q&A pair extraction from Obsidian notes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from obsidian_anki_sync.agents.autofix.handlers import UnbalancedCodeFenceHandler
from obsidian_anki_sync.exceptions import ParserError, TruncationError
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.utils.logging import get_logger

from .parser_state import _get_qa_extractor

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.qa_extractor import QAExtractorAgent

logger = get_logger(__name__)


def _autofix_unbalanced_code_fences(content: str, file_path: Path) -> str:
    """Auto-fix unbalanced code fences in content.

    Detects unclosed ``` code blocks and adds missing closing fences.
    This prevents parsing errors from notes with formatting issues.

    Args:
        content: Note content that may have unbalanced code fences
        file_path: Path to the file (for logging)

    Returns:
        Fixed content with balanced code fences
    """
    handler = UnbalancedCodeFenceHandler()
    issues = handler.detect(content, metadata=None)

    if not issues:
        return content

    # Apply fixes
    fixed_content, updated_issues = handler.fix(content, issues, metadata=None)

    # Log what was fixed
    fixed_count = sum(1 for issue in updated_issues if issue.auto_fixed)
    if fixed_count > 0:
        logger.info(
            "autofix_unbalanced_code_fences",
            file=str(file_path),
            issues_found=len(issues),
            issues_fixed=fixed_count,
        )

    return fixed_content


def parse_qa_pairs(
    content: str,
    metadata: NoteMetadata,
    file_path: Path | None = None,
    qa_extractor: QAExtractorAgent | None = None,
    tolerant_parsing: bool = False,
) -> list[QAPair]:
    """
    Parse Q/A pairs from note content.

    Uses LLM-based extraction if qa_extractor provided, otherwise falls back to rigid parsing.

    Args:
        content: Full note content
        metadata: Parsed metadata
        file_path: Optional file path for logging
        qa_extractor: Optional QA extractor agent for LLM-based extraction.
                     If None, checks thread-local state.
        tolerant_parsing: If True, allows parsing to proceed even with incomplete Q&A pairs.
                        Missing sections will be handled by repair agent.

    Returns:
        List of Q/A pairs

    Raises:
        ParserError: If structure is invalid (unless tolerant_parsing=True)
    """
    # Determine which extractor to use
    extractor_to_use = _get_qa_extractor(qa_extractor)

    # Try LLM-based extraction first if enabled
    if extractor_to_use is not None:
        logger.info(
            "attempting_llm_extraction",
            note_id=metadata.id,
            title=metadata.title,
            file=str(file_path) if file_path else "unknown",
        )

        try:
            qa_pairs = extractor_to_use.extract_qa_pairs(
                note_content=content,
                metadata=metadata,
                file_path=file_path,
            )

            if qa_pairs:
                logger.info(
                    "llm_extraction_succeeded",
                    note_id=metadata.id,
                    title=metadata.title,
                    file=str(file_path) if file_path else "unknown",
                    pairs_count=len(qa_pairs),
                )
                return qa_pairs
            else:
                logger.warning(
                    "llm_extraction_returned_empty_falling_back",
                    note_id=metadata.id,
                    title=metadata.title,
                    file=str(file_path) if file_path else "unknown",
                )
        except TruncationError:
            # Re-raise TruncationError - caller should handle this explicitly
            raise
        except Exception as e:
            logger.warning(
                "llm_extraction_failed_falling_back",
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                error=str(e),
            )

    # Fall back to rigid pattern-based parsing
    logger.debug(
        "using_rigid_parser",
        note_id=metadata.id,
        title=metadata.title,
        file=str(file_path) if file_path else "unknown",
    )

    # Strip frontmatter
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)

    # Normalize line endings and strip BOM
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.removeprefix("\ufeff")

    qa_pairs = []
    card_index = 1

    # Find positions of question markers that start new blocks
    parts = []
    lines = content.split("\n")
    current_block_lines: list[str] = []
    in_answer_section = False

    for line in lines:
        stripped = line.strip()

        # Check if this is a question marker
        is_question_marker = stripped in ("# Question (EN)", "# Вопрос (RU)")

        # Start new block if: (1) question marker AND (2) we've seen answers or this is the first block
        if is_question_marker and (in_answer_section or not current_block_lines):
            if current_block_lines:
                parts.append("\n".join(current_block_lines))
                current_block_lines = []
            in_answer_section = False

        # Check if we're entering answer section
        answer_section_headers = (
            "## Answer (EN)",
            "## Ответ (RU)",
            "## Follow-ups",
            "## References",
            "## Related Questions",
        )
        if stripped in answer_section_headers:
            in_answer_section = True

        current_block_lines.append(line)

    # Add the last block
    if current_block_lines:
        parts.append("\n".join(current_block_lines))

    blocks = parts

    # Track parsing failures to report to caller
    failed_blocks: list[tuple[int, str]] = []

    for block in blocks:
        if not block.strip():
            continue

        # Try to parse this block as a Q/A pair
        try:
            qa_pair = _parse_single_qa_block(block, card_index, metadata, file_path)
            if qa_pair:
                qa_pairs.append(qa_pair)
                card_index += 1
            else:
                # Block was incomplete - attempt LLM extraction if enabled and tolerant_parsing is True
                if tolerant_parsing and extractor_to_use is not None:
                    logger.debug(
                        "attempting_llm_extraction_for_incomplete_block",
                        file=str(file_path) if file_path else "unknown",
                        card_index=card_index,
                        block_preview=block[:200],
                    )
                    try:
                        block_content = f"---\nid: {metadata.id}\ntitle: {metadata.title}\ntopic: {metadata.topic}\nlanguage_tags: {metadata.language_tags}\ncreated: {metadata.created}\nupdated: {metadata.updated}\n---\n\n{block}"
                        extracted_pairs = extractor_to_use.extract_qa_pairs(
                            note_content=block_content,
                            metadata=metadata,
                            file_path=file_path,
                        )
                        if extracted_pairs:
                            qa_pairs.extend(extracted_pairs)
                            logger.info(
                                "llm_extraction_recovered_incomplete_block",
                                file=str(file_path) if file_path else "unknown",
                                card_index=card_index,
                                recovered_pairs=len(extracted_pairs),
                            )
                            card_index += len(extracted_pairs)
                            continue
                    except Exception as e:
                        logger.debug(
                            "llm_extraction_failed_for_incomplete_block",
                            file=str(file_path) if file_path else "unknown",
                            card_index=card_index,
                            error=str(e),
                        )

                # Block was skipped (incomplete or invalid)
                failed_blocks.append((card_index, "Incomplete or invalid Q/A block"))
        except ParserError as e:
            logger.error(
                "qa_parse_error",
                card_index=card_index,
                file=str(file_path) if file_path else "unknown",
                error=str(e),
            )
            failed_blocks.append((card_index, str(e)))

    # Report parsing summary
    if failed_blocks:
        logger.warning(
            "qa_parsing_incomplete",
            file=str(file_path) if file_path else "unknown",
            note_title=metadata.title,
            total_blocks=len(blocks),
            successful=len(qa_pairs),
            failed=len(failed_blocks),
            failed_indices=[idx for idx, _ in failed_blocks],
        )
        for idx, error in failed_blocks[:3]:
            logger.debug(
                "qa_parse_failure_detail",
                file=str(file_path) if file_path else "unknown",
                card_index=idx,
                error=error,
            )

    if not qa_pairs:
        logger.warning(
            "no_qa_pairs_found",
            file=str(file_path) if file_path else "unknown",
            note_title=metadata.title,
        )

    return qa_pairs


def _parse_single_qa_block(
    block: str, card_index: int, metadata: NoteMetadata, file_path: Path | None = None
) -> QAPair | None:
    """Parse a single Q/A block."""
    lines = block.split("\n")

    # State machine for parsing
    state = "INIT"
    question_en: list[str] = []
    question_ru: list[str] = []
    answer_en: list[str] = []
    answer_ru: list[str] = []
    followups: list[str] = []
    references: list[str] = []
    related: list[str] = []
    context_before: list[str] = []

    current_section: list[str] | None = None

    for line in lines:
        stripped = line.strip()

        # Check for section headers
        if stripped == "# Question (EN)":
            state = "QUESTION_EN"
            current_section = question_en
            continue
        elif stripped == "# Вопрос (RU)":
            state = "QUESTION_RU"
            current_section = question_ru
            continue
        elif stripped == "---":
            if state in ("QUESTION_RU", "QUESTION_EN", "ANSWER_EN"):
                state = "SEPARATOR"
                continue
        elif stripped == "## Answer (EN)":
            state = "ANSWER_EN"
            current_section = answer_en
            continue
        elif stripped == "## Ответ (RU)":
            state = "ANSWER_RU"
            current_section = answer_ru
            continue
        elif stripped == "## Follow-ups":
            state = "FOLLOWUPS"
            current_section = followups
            continue
        elif stripped == "## References":
            state = "REFERENCES"
            current_section = references
            continue
        elif stripped == "## Related Questions":
            state = "RELATED"
            current_section = related
            continue

        # Accumulate content
        if state == "INIT" and stripped:
            context_before.append(line)
        elif (
            state
            in (
                "QUESTION_EN",
                "QUESTION_RU",
                "ANSWER_EN",
                "ANSWER_RU",
                "FOLLOWUPS",
                "REFERENCES",
                "RELATED",
            )
            and current_section is not None
        ):
            current_section.append(line)

    # Check if we have all required sections
    if not question_en or not question_ru:
        if state != "INIT":
            missing_sections = []
            if not question_en:
                missing_sections.append("Question (EN)")
            if not question_ru:
                missing_sections.append("Вопрос (RU)")

            block_preview = block[:200].replace("\n", " ")
            if len(block) > 200:
                block_preview += "..."

            logger.warning(
                "incomplete_qa_block",
                file=str(file_path) if file_path else "unknown",
                note_title=metadata.title,
                card_index=card_index,
                state=state,
                missing_sections=missing_sections,
                block_preview=block_preview,
            )
        return None

    # Check if we have at least one answer section
    if not answer_en and not answer_ru:
        if state != "INIT":
            missing_sections = []
            if not answer_en:
                missing_sections.append("Answer (EN)")
            if not answer_ru:
                missing_sections.append("Ответ (RU)")

            logger.warning(
                "no_answers_found",
                file=str(file_path) if file_path else "unknown",
                note_title=metadata.title,
                card_index=card_index,
                state=state,
                missing_sections=missing_sections,
            )
        return None

    # Validate language tags
    has_en = "en" in metadata.language_tags
    has_ru = "ru" in metadata.language_tags

    if has_en and not (question_en and answer_en):
        logger.error(
            "missing_en_content",
            file=str(file_path) if file_path else "unknown",
            note_title=metadata.title,
            card_index=card_index,
            has_question=bool(question_en),
            has_answer=bool(answer_en),
        )
        return None

    if has_ru and not (question_ru and answer_ru):
        logger.error(
            "missing_ru_content",
            file=str(file_path) if file_path else "unknown",
            note_title=metadata.title,
            card_index=card_index,
            has_question=bool(question_ru),
            has_answer=bool(answer_ru),
        )
        return None

    # Helper to clean blockquote markers from text
    def clean_blockquote(text: str) -> str:
        """Remove leading > markers from each line."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("> "):
                stripped = stripped[2:]
            elif stripped == ">":
                stripped = ""
            else:
                stripped = line.rstrip()
            cleaned.append(stripped)
        return "\n".join(cleaned).strip()

    # Build QAPair
    return QAPair(
        card_index=card_index,
        question_en=clean_blockquote("\n".join(question_en)),
        question_ru=clean_blockquote("\n".join(question_ru)),
        answer_en="\n".join(answer_en).strip(),
        answer_ru="\n".join(answer_ru).strip(),
        followups="\n".join(followups).strip(),
        references="\n".join(references).strip(),
        related="\n".join(related).strip(),
        context="\n".join(context_before).strip(),
    )
