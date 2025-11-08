"""Obsidian note parser for YAML frontmatter and Q/A pairs."""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import frontmatter
import yaml
from ruamel.yaml import YAML

from ..exceptions import ParserError
from ..models import NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Configure ruamel.yaml for preserving comments and order
# This is available for future write operations
ruamel_yaml = YAML()
ruamel_yaml.preserve_quotes = True
ruamel_yaml.default_flow_style = False
ruamel_yaml.width = 4096  # Prevent line wrapping


def parse_note(file_path: Path) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note and extract metadata and Q/A pairs.

    Args:
        file_path: Path to the markdown file

    Returns:
        Tuple of (metadata, qa_pairs)

    Raises:
        ParserError: If parsing fails
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

    # Parse Q/A pairs
    qa_pairs = parse_qa_pairs(content, metadata)

    logger.debug(
        "parsed_note",
        file=str(file_path),
        pairs_count=len(qa_pairs),
        languages=metadata.language_tags,
    )

    return metadata, qa_pairs


def parse_note_with_repair(
    file_path: Path,
    ollama_client: Optional[BaseLLMProvider] = None,
    repair_model: str = "qwen3:8b",
    enable_repair: bool = True,
) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note with automatic repair fallback.

    First attempts rule-based parsing. If that fails and repair is enabled,
    attempts intelligent repair using LLM agent.

    Args:
        file_path: Path to the markdown file
        ollama_client: LLM provider for repair (required if enable_repair=True)
        repair_model: Model to use for repair
        enable_repair: Whether to attempt repair on parse failures

    Returns:
        Tuple of (metadata, qa_pairs)

    Raises:
        ParserError: If both parsing and repair fail
    """
    # First try rule-based parsing
    try:
        return parse_note(file_path)
    except ParserError as e:
        # If repair is disabled or no client provided, re-raise
        if not enable_repair or ollama_client is None:
            raise

        # Attempt repair with agent
        logger.info(
            "parser_attempting_repair",
            file=str(file_path),
            original_error=str(e),
        )

        from ..agents.parser_repair import attempt_repair

        result = attempt_repair(
            file_path=file_path,
            original_error=e,
            ollama_client=ollama_client,
            model=repair_model,
        )

        if result is None:
            # Repair failed, re-raise original error
            logger.error(
                "parser_repair_failed",
                file=str(file_path),
                original_error=str(e),
            )
            raise ParserError(f"Parse failed and repair unsuccessful: {e}") from e

        # Repair succeeded
        metadata, qa_pairs = result
        logger.info(
            "parser_repair_succeeded",
            file=str(file_path),
            qa_pairs_count=len(qa_pairs),
        )
        return metadata, qa_pairs


def parse_frontmatter(content: str, file_path: Path) -> NoteMetadata:
    """
    Extract and parse YAML frontmatter from note content.

    Uses python-frontmatter with ruamel.yaml to preserve comments and order.

    Args:
        content: Full note content
        file_path: Path to the file (for context)

    Returns:
        Parsed metadata

    Raises:
        ParserError: If frontmatter is missing or invalid
    """
    # Parse frontmatter using python-frontmatter
    try:
        post = frontmatter.loads(content)
    except Exception as e:
        raise ParserError(f"Invalid YAML in {file_path}: {e}")

    # Extract metadata dictionary
    data = post.metadata

    if not data:
        raise ParserError(f"No frontmatter found in {file_path}")

    # Validate required fields
    required_fields = ["id", "title", "topic", "language_tags", "created", "updated"]
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


def parse_qa_pairs(content: str, metadata: NoteMetadata) -> list[QAPair]:
    """
    Parse Q/A pairs from note content.

    Args:
        content: Full note content
        metadata: Parsed metadata

    Returns:
        List of Q/A pairs

    Raises:
        ParserError: If structure is invalid
    """
    # Strip frontmatter
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)

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
            qa_pair = _parse_single_qa_block(block, card_index, metadata)
            if qa_pair:
                qa_pairs.append(qa_pair)
                card_index += 1
            else:
                # Block was skipped (incomplete or invalid)
                failed_blocks.append((card_index, "Incomplete or invalid Q/A block"))
        except ParserError as e:
            logger.error("qa_parse_error", card_index=card_index, error=str(e))
            failed_blocks.append((card_index, str(e)))
            # Continue processing other blocks

    # Report parsing summary
    if failed_blocks:
        logger.warning(
            "qa_parsing_incomplete",
            total_blocks=len(blocks),
            successful=len(qa_pairs),
            failed=len(failed_blocks),
            failed_indices=[idx for idx, _ in failed_blocks],
        )
        # Log details of first few failures
        for idx, error in failed_blocks[:3]:
            logger.debug("qa_parse_failure_detail", card_index=idx, error=error)

    if not qa_pairs:
        logger.warning("no_qa_pairs_found")

    return qa_pairs


def _parse_single_qa_block(
    block: str, card_index: int, metadata: NoteMetadata
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
            logger.warning("incomplete_qa_block", card_index=card_index, state=state)
        return None

    # Check if we have at least one answer section
    # Note: Separator (---) between questions and answers is optional
    if not answer_en and not answer_ru:
        if state != "INIT":
            logger.warning("no_answers_found", card_index=card_index, state=state)
        return None

    # Validate language tags
    has_en = "en" in metadata.language_tags
    has_ru = "ru" in metadata.language_tags

    if has_en and not (question_en and answer_en):
        logger.error(
            "missing_en_content",
            card_index=card_index,
            has_question=bool(question_en),
            has_answer=bool(answer_en),
        )
        return None

    if has_ru and not (question_ru and answer_ru):
        logger.error(
            "missing_ru_content",
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


def discover_notes(vault_path: Path, source_dir: Path) -> list[tuple[Path, str]]:
    """
    Discover Q&A notes in the vault.

    Args:
        vault_path: Root vault path
        source_dir: Relative source directory

    Returns:
        List of (absolute_path, relative_path) tuples
    """
    full_source = vault_path / source_dir

    if not full_source.exists():
        logger.error("source_dir_not_found", path=str(full_source))
        return []

    # Find all q-*.md files recursively
    notes = []
    for md_file in full_source.rglob("q-*.md"):
        # Ignore certain patterns
        if any(part.startswith(("c-", "moc-", "template")) for part in md_file.parts):
            continue

        # Calculate relative path from source_dir
        relative = md_file.relative_to(full_source)
        notes.append((md_file, str(relative)))

    logger.info("discovered_notes", count=len(notes), path=str(full_source))
    return notes


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
            normalized.append({k: str(v) for k, v in item.items() if v is not None})
        else:
            text = str(item).strip()
            if text:
                normalized.append({"url": text})
    return normalized
