"""Obsidian note parser for YAML frontmatter and Q/A pairs."""

import re
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import frontmatter
from ruamel.yaml import YAML

from ..exceptions import ParserError
from ..models import NoteMetadata, QAPair
from ..obsidian.note_validator import validate_note_structure
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..agents.qa_extractor import QAExtractorAgent

logger = get_logger(__name__)

# Configure ruamel.yaml for preserving comments and order
# This is available for future write operations
ruamel_yaml = YAML()
ruamel_yaml.preserve_quotes = True
ruamel_yaml.default_flow_style = False
ruamel_yaml.width = 4096  # Prevent line wrapping


# Backward compatibility - global state (deprecated)
_USE_LLM_EXTRACTION = False
_QA_EXTRACTOR_AGENT = None
_ENFORCE_LANGUAGE_VALIDATION = False


@contextmanager
def temporarily_disable_llm_extraction() -> Any:
    """Temporarily disable global LLM extraction state."""
    global _USE_LLM_EXTRACTION, _QA_EXTRACTOR_AGENT, _ENFORCE_LANGUAGE_VALIDATION

    previous_flag = _USE_LLM_EXTRACTION
    previous_agent = _QA_EXTRACTOR_AGENT
    previous_lang_validation = _ENFORCE_LANGUAGE_VALIDATION

    _USE_LLM_EXTRACTION = False
    _QA_EXTRACTOR_AGENT = None
    _ENFORCE_LANGUAGE_VALIDATION = False
    try:
        logger.debug("llm_extraction_temporarily_disabled")
        yield
    finally:
        _USE_LLM_EXTRACTION = previous_flag
        _QA_EXTRACTOR_AGENT = previous_agent
        _ENFORCE_LANGUAGE_VALIDATION = previous_lang_validation
        logger.debug("llm_extraction_restored", enabled=_USE_LLM_EXTRACTION)


def configure_llm_extraction(
    llm_provider: BaseLLMProvider | None,
    model: str = "qwen3:8b",
    temperature: float = 0.0,
    reasoning_enabled: bool = False,
    enforce_language_validation: bool = False,
) -> None:
    """Configure LLM-based Q&A extraction using global state (deprecated).

    DEPRECATED: Use create_qa_extractor() and pass the extractor to parse functions instead.
    This function remains for backward compatibility but sets global state which is not thread-safe.

    Args:
        llm_provider: LLM provider instance (None to disable)
        model: Model to use for extraction
        temperature: Sampling temperature
        reasoning_enabled: Enable reasoning mode for models that support it
    """
    global _USE_LLM_EXTRACTION, _QA_EXTRACTOR_AGENT, _ENFORCE_LANGUAGE_VALIDATION

    if llm_provider is None:
        _USE_LLM_EXTRACTION = False
        _QA_EXTRACTOR_AGENT = None
        _ENFORCE_LANGUAGE_VALIDATION = False
        logger.info("llm_extraction_disabled")
        return

    from ..agents.qa_extractor import QAExtractorAgent

    _QA_EXTRACTOR_AGENT = QAExtractorAgent(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled,
    )
    _USE_LLM_EXTRACTION = True
    _ENFORCE_LANGUAGE_VALIDATION = enforce_language_validation
    logger.info(
        "llm_extraction_enabled",
        model=model,
        temperature=temperature,
        enforce_language_validation=enforce_language_validation,
    )


def create_qa_extractor(
    llm_provider: BaseLLMProvider,
    model: str = "qwen3:8b",
    temperature: float = 0.0,
    reasoning_enabled: bool = False,
    enable_content_generation: bool = True,
    repair_missing_sections: bool = True,
) -> "QAExtractorAgent":
    """Create a QA extractor agent for LLM-based extraction.

    This is the preferred way to use LLM extraction instead of configure_llm_extraction().

    Args:
        llm_provider: LLM provider instance
        model: Model to use for extraction
        temperature: Sampling temperature
        reasoning_enabled: Enable reasoning mode for models that support it
        enable_content_generation: Allow LLM to generate missing content
        repair_missing_sections: Generate missing language sections

    Returns:
        Configured QAExtractorAgent instance
    """
    from ..agents.qa_extractor import QAExtractorAgent

    return QAExtractorAgent(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
        reasoning_enabled=reasoning_enabled,
        enable_content_generation=enable_content_generation,
        repair_missing_sections=repair_missing_sections,
    )


def parse_note(
    file_path: Path,
    qa_extractor: Optional["QAExtractorAgent"] = None,
    tolerant_parsing: bool = False,
) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note and extract metadata and Q/A pairs.

    Args:
        file_path: Path to the markdown file
        qa_extractor: Optional QA extractor agent for LLM-based extraction.
                     If None, uses global state for backward compatibility.
        tolerant_parsing: If True, validation errors are logged as warnings instead
                        of raising ParserError. Allows parsing to proceed with
                        imperfect notes that can be repaired later.

    Returns:
        Tuple of (metadata, qa_pairs)

    Raises:
        ParserError: If parsing fails (unless tolerant_parsing=True for validation errors)
    """
    # Resolve to absolute path and validate
    try:
        resolved_path = file_path.resolve()
        if not resolved_path.exists():
            raise ParserError(f"File does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise ParserError(f"Path is not a file: {resolved_path}")
    except (OSError, RuntimeError) as e:
        raise ParserError(f"Cannot resolve path {file_path}: {e}")

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ParserError(f"Failed to read file {resolved_path}: {e}")
    except Exception as e:
        logger.exception("unexpected_file_read_error", file=str(resolved_path))
        raise ParserError(f"Unexpected error reading {resolved_path}: {e}")

    # Parse frontmatter
    metadata = parse_frontmatter(content, file_path)

    # Validate note structure
    if _ENFORCE_LANGUAGE_VALIDATION or not tolerant_parsing:
        validation_errors = validate_note_structure(metadata, content)
        if validation_errors:
            if tolerant_parsing:
                # Log as warnings but continue parsing
                for error in validation_errors:
                    logger.warning(
                        "note_validation_warning",
                        file=str(file_path),
                        error=error,
                    )
            else:
                # Raise error (strict mode)
                raise ParserError("; ".join(validation_errors))

    # Parse Q/A pairs
    qa_pairs = parse_qa_pairs(
        content, metadata, file_path, qa_extractor=qa_extractor, tolerant_parsing=tolerant_parsing
    )

    logger.debug(
        "parsed_note",
        file=str(file_path),
        pairs_count=len(qa_pairs),
        languages=metadata.language_tags,
        tolerant_parsing=tolerant_parsing,
    )

    return metadata, qa_pairs


def parse_note_with_repair(
    file_path: Path,
    ollama_client: Optional[BaseLLMProvider] = None,
    repair_model: str = "qwen3:8b",
    enable_repair: bool = True,
    tolerant_parsing: bool = True,
    enable_content_generation: bool = True,
    repair_missing_sections: bool = True,
) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note with automatic repair fallback.

    First attempts rule-based parsing with tolerant mode. If that fails and repair is enabled,
    attempts intelligent repair using LLM agent with content generation.

    Args:
        file_path: Path to the markdown file
        ollama_client: LLM provider for repair (required if enable_repair=True)
        repair_model: Model to use for repair
        enable_repair: Whether to attempt repair on parse failures
        tolerant_parsing: If True, validation errors are warnings, not errors
        enable_content_generation: Allow LLM to generate missing content during repair
        repair_missing_sections: Generate missing language sections during repair

    Returns:
        Tuple of (metadata, qa_pairs)

    Raises:
        ParserError: If both parsing and repair fail
    """
    # First try rule-based parsing with tolerant mode
    try:
        return parse_note(file_path, tolerant_parsing=tolerant_parsing)
    except ParserError as e:
        # If repair is disabled or no client provided, re-raise
        if not enable_repair or ollama_client is None:
            raise

        # Attempt repair with agent
        logger.info(
            "parser_attempting_repair",
            file=str(file_path),
            original_error=str(e),
            enable_content_generation=enable_content_generation,
        )

        from ..agents.parser_repair import attempt_repair

        result = attempt_repair(
            file_path=file_path,
            original_error=e,
            ollama_client=ollama_client,
            model=repair_model,
            enable_content_generation=enable_content_generation,
            repair_missing_sections=repair_missing_sections,
        )

        if result is None:
            # Repair failed, re-raise original error
            logger.error(
                "parser_repair_failed",
                file=str(file_path),
                original_error=str(e),
            )
            raise ParserError(
                f"Parse failed and repair unsuccessful: {e}") from e

        # Repair succeeded
        metadata, qa_pairs = result
        logger.info(
            "parser_repair_succeeded",
            file=str(file_path),
            qa_pairs_count=len(qa_pairs),
        )
        return metadata, qa_pairs


def _preprocess_yaml_frontmatter(content: str) -> str:
    """
    Preprocess YAML frontmatter to fix common syntax errors.

    Fixes issues like:
    - Backticks in YAML arrays/strings (replaces with quotes or removes)
    - Other common YAML syntax problems

    Args:
        content: Full note content with YAML frontmatter

    Returns:
        Preprocessed content with fixed YAML syntax
    """
    # Extract frontmatter section
    frontmatter_match = re.match(
        r"^(---\s*\n)(.*?)(\n---\s*\n)", content, re.DOTALL)
    if not frontmatter_match:
        return content

    frontmatter_start = frontmatter_match.group(1)
    frontmatter_body = frontmatter_match.group(2)
    frontmatter_end = frontmatter_match.group(3)
    rest_content = content[frontmatter_match.end():]

    # Fix backticks in YAML arrays/strings
    # Pattern: matches backticks around words in arrays or after colons
    # Example: aliases: [Completion, `Flow`, onCompletion] -> aliases: [Completion, Flow, onCompletion]
    # Or: aliases: [`Flow`] -> aliases: [Flow]
    def fix_backticks(text: str) -> str:
        # Replace backticks in array values: `word` -> word
        # This handles cases like: [item1, `item2`, item3]
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text

    fixed_frontmatter = fix_backticks(frontmatter_body)

    return frontmatter_start + fixed_frontmatter + frontmatter_end + rest_content


def parse_frontmatter(content: str, file_path: Path) -> NoteMetadata:
    """
    Extract and parse YAML frontmatter from note content.

    Uses python-frontmatter with ruamel.yaml to preserve comments and order.
    Applies preprocessing to fix common YAML syntax errors before parsing.

    Args:
        content: Full note content
        file_path: Path to the file (for context)

    Returns:
        Parsed metadata

    Raises:
        ParserError: If frontmatter is missing or invalid
    """
    # Preprocess content to fix common YAML syntax errors
    try:
        preprocessed_content = _preprocess_yaml_frontmatter(content)
    except Exception as e:
        logger.warning(
            "yaml_preprocessing_failed",
            file=str(file_path),
            error=str(e),
        )
        # Continue with original content if preprocessing fails
        preprocessed_content = content

    # Parse frontmatter using python-frontmatter
    try:
        post = frontmatter.loads(preprocessed_content)
    except Exception as e:
        # Provide more helpful error message
        error_msg = str(e)
        if "backtick" in error_msg.lower() or "`" in error_msg:
            raise ParserError(
                f"Invalid YAML in {file_path}: {e}. "
                "Note: Backticks (`) are not valid YAML syntax. Use quotes or remove them."
            ) from e
        raise ParserError(f"Invalid YAML in {file_path}: {e}") from e

    # Extract metadata dictionary
    data = post.metadata

    if not data:
        raise ParserError(f"No frontmatter found in {file_path}")

    # Validate required fields
    required_fields = ["id", "title", "topic",
                       "language_tags", "created", "updated"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ParserError(f"Missing required fields in {file_path}: {missing}")

    # Parse dates
    try:
        created = _parse_date(data["created"])
        updated = _parse_date(data["updated"])
    except (ValueError, TypeError) as e:
        raise ParserError(f"Invalid date format in {file_path}: {e}")

    # Topic mismatch check is now handled by callers for aggregation

    # Build metadata object
    moc = _normalize_wikilink(data.get("moc"))
    related = _normalize_link_list(data.get("related"))
    tags = _normalize_string_list(data.get("tags"))
    aliases = _normalize_string_list(data.get("aliases"))
    subtopics = _normalize_string_list(data.get("subtopics"))
    sources = _normalize_sources(data.get("sources"))

    metadata = NoteMetadata(
        id=str(data["id"]),
        title=str(data["title"]),
        topic=str(data["topic"]),
        language_tags=_ensure_list(data["language_tags"]),
        created=created,
        updated=updated,
        aliases=aliases,
        subtopics=subtopics,
        question_kind=data.get("question_kind"),
        difficulty=data.get("difficulty"),
        original_language=data.get("original_language"),
        source=data.get("source"),
        source_note=data.get("source_note"),
        status=data.get("status"),
        moc=moc,
        related=related,
        tags=tags,
        sources=sources,
        anki_note_type=data.get("anki_note_type"),
        anki_slugs=_ensure_list(data.get("anki_slugs", [])),
    )

    # Validate original_language is in language_tags
    if (
        metadata.original_language
        and metadata.original_language not in metadata.language_tags
    ):
        logger.warning(
            "original_language_not_in_tags",
            file=str(file_path),
            original=metadata.original_language,
            tags=metadata.language_tags,
        )

    return metadata


def parse_qa_pairs(
    content: str,
    metadata: NoteMetadata,
    file_path: Path | None = None,
    qa_extractor: Optional["QAExtractorAgent"] = None,
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
                     If None, checks global state for backward compatibility.
        tolerant_parsing: If True, allows parsing to proceed even with incomplete Q&A pairs.
                        Missing sections will be handled by repair agent.

    Returns:
        List of Q/A pairs

    Raises:
        ParserError: If structure is invalid (unless tolerant_parsing=True)
    """
    # Determine which extractor to use (parameter takes precedence over global)
    extractor_to_use = qa_extractor
    if extractor_to_use is None and _USE_LLM_EXTRACTION:
        extractor_to_use = _QA_EXTRACTOR_AGENT

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
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "",
                     content, count=1, flags=re.DOTALL)

    # Normalize line endings and strip BOM
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if content.startswith("\ufeff"):
        content = content[1:]

    qa_pairs = []
    card_index = 1

    # Find all Q/A blocks
    # Pattern supports both EN→RU and RU→EN ordering for questions/answers.

    # Find positions of question markers that start new blocks
    # A question marker starts a new block if it comes after a separator or at the start
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
        if stripped.startswith("## "):
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
            qa_pair = _parse_single_qa_block(
                block, card_index, metadata, file_path)
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
                        # Try extracting from just this block
                        # Create a minimal note content with just this block
                        block_content = f"---\nid: {metadata.id}\ntitle: {metadata.title}\ntopic: {metadata.topic}\nlanguage_tags: {metadata.language_tags}\ncreated: {metadata.created}\nupdated: {metadata.updated}\n---\n\n{block}"
                        extracted_pairs = extractor_to_use.extract_qa_pairs(
                            note_content=block_content,
                            metadata=metadata,
                            file_path=file_path,
                        )
                        if extracted_pairs:
                            # Use the first extracted pair and update card_index
                            qa_pairs.extend(extracted_pairs)
                            logger.info(
                                "llm_extraction_recovered_incomplete_block",
                                file=str(
                                    file_path) if file_path else "unknown",
                                card_index=card_index,
                                recovered_pairs=len(extracted_pairs),
                            )
                            card_index += len(extracted_pairs)
                            continue  # Successfully recovered, skip adding to failed_blocks
                    except Exception as e:
                        logger.debug(
                            "llm_extraction_failed_for_incomplete_block",
                            file=str(file_path) if file_path else "unknown",
                            card_index=card_index,
                            error=str(e),
                        )
                        # Fall through to add to failed_blocks

                # Block was skipped (incomplete or invalid)
                failed_blocks.append(
                    (card_index, "Incomplete or invalid Q/A block"))
        except ParserError as e:
            logger.error(
                "qa_parse_error",
                card_index=card_index,
                file=str(file_path) if file_path else "unknown",
                error=str(e),
            )
            failed_blocks.append((card_index, str(e)))
            # Continue processing other blocks

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
        # Log details of first few failures
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
            # Separator can appear:
            # 1. Between RU and EN questions (RU-first format)
            # 2. After both questions before answers (standard format)
            # 3. After EN answer in some cases
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
        if state != "INIT":  # Only warn if we started parsing
            missing_sections = []
            if not question_en:
                missing_sections.append("Question (EN)")
            if not question_ru:
                missing_sections.append("Вопрос (RU)")

            # Show a preview of the block (first 200 chars)
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
    # Note: Separator (---) between questions and answers is optional
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
                stripped = line.rstrip()  # Keep indentation but strip trailing spaces
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


def discover_notes(
    vault_path: Path, source_dir: Path | list[Path] | None = None
) -> list[tuple[Path, str]]:
    """
    Discover Q&A notes in the vault.

    Args:
        vault_path: Root vault path
        source_dir: Single relative source directory, or list of relative directories.
                   If None, searches the entire vault.

    Returns:
        List of (absolute_path, relative_path) tuples
    """
    # Convert single path to list for uniform handling
    if source_dir is None:
        source_dirs = [Path(".")]
    elif isinstance(source_dir, list):
        source_dirs = source_dir
    else:
        source_dirs = [source_dir]

    all_notes = []
    total_searched = 0

    for src_dir in source_dirs:
        full_source = vault_path / src_dir

        if not full_source.exists():
            logger.warning(
                "source_dir_not_found",
                path=str(full_source),
                relative_path=str(src_dir),
            )
            continue

        # Find all q-*.md files recursively in this directory
        dir_notes = []
        for md_file in full_source.rglob("q-*.md"):
            # Ignore certain patterns
            if any(
                part.startswith(("c-", "moc-", "template")) for part in md_file.parts
            ):
                continue

            # Calculate relative path from the source_dir root
            relative = md_file.relative_to(full_source)
            # Prepend source dir to relative path for clarity
            full_relative = src_dir / relative
            dir_notes.append((md_file, str(full_relative)))

        logger.info(
            "discovered_notes_in_dir",
            count=len(dir_notes),
            path=str(full_source),
            relative_path=str(src_dir),
        )
        all_notes.extend(dir_notes)
        total_searched += 1

    logger.info(
        "discovered_notes_total",
        count=len(all_notes),
        directories_searched=total_searched,
    )
    return all_notes


def _parse_date(value: Any) -> datetime:
    """Parse date from various formats."""
    if isinstance(value, datetime):
        return value

    # Handle datetime.date objects from YAML parsing
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    if isinstance(value, str):
        # Try ISO format first
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass

        # Try common formats
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    raise ValueError(f"Cannot parse date: {value}")


def _ensure_list(value: Any) -> list[Any]:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _normalize_wikilink(value: Any) -> str | None:
    """Strip Obsidian wikilink syntax from a single value."""
    if value is None:
        return None

    if isinstance(value, list):
        # Prefer first non-empty entry
        for item in value:
            cleaned = _normalize_wikilink(item)
            if cleaned:
                return cleaned
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].strip()
    return text or None


def _normalize_link_list(value: Any) -> list[str]:
    """Normalize related/moc entries that may include wikilinks or bullet formatting."""
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        # Support multi-line scalar formats or comma-separated strings
        if isinstance(value, str):
            items = re.split(r"[\n,]+", value)
        else:
            items = [value]

    normalized: list[str] = []
    for item in items:
        if item is None:
            continue

        # Handle nested lists (YAML interprets [[link]] as nested list structure)
        if isinstance(item, list):
            # Flatten nested lists - extract the innermost string value
            while isinstance(item, list) and len(item) > 0:
                item = item[0]
            if item is None:
                continue

        text = str(item).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        # Only strip [[ ]] if it's a string representation, not a nested list
        if text.startswith("[[") and text.endswith("]]"):
            text = text[2:-2].strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_sources(value: Any) -> list[dict[str, str]]:
    """Normalize sources into a list of dictionaries with string values."""
    if value is None:
        return []

    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[dict[str, str]] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            normalized.append({k: str(v)
                              for k, v in item.items() if v is not None})
        else:
            text = str(item).strip()
            if text:
                normalized.append({"url": text})
    return normalized
