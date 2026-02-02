"""Obsidian note parser for YAML frontmatter and Q/A pairs.

Public API module. Implementation is split across:
- frontmatter.py: YAML frontmatter parsing
- qa_extraction.py: Regex-based Q&A pair extraction
- parser_state.py: Thread-local state management
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.obsidian.note_validator import validate_note_structure
from obsidian_anki_sync.utils.logging import get_logger

# Re-export submodule symbols for backward compatibility
from .frontmatter import (
    _detect_content_corruption,
    _ensure_list,
    _normalize_link_list,
    _normalize_sources,
    _normalize_string_list,
    _normalize_wikilink,
    _parse_date,
    _parse_inline_array,
    _preprocess_yaml_frontmatter,
    parse_frontmatter,
    ruamel_yaml,
)
from .parser_state import (
    _get_enforce_language_validation,
    _get_qa_extractor,
    _ParserState,
    _thread_local_state,
    create_qa_extractor,
    set_thread_qa_extractor,
    temporarily_disable_llm_extraction,
)
from .qa_extraction import (
    _autofix_unbalanced_code_fences,
    _parse_single_qa_block,
    parse_qa_pairs,
)

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.qa_extractor import QAExtractorAgent
    from obsidian_anki_sync.models import NoteMetadata, QAPair
    from obsidian_anki_sync.providers.base import BaseLLMProvider

logger = get_logger(__name__)

# Re-export ParserError for modules that import it from here
__all__ = [
    "ParserError",
    "_ParserState",
    "_autofix_unbalanced_code_fences",
    "_detect_content_corruption",
    "_ensure_list",
    "_get_enforce_language_validation",
    "_get_qa_extractor",
    "_normalize_link_list",
    "_normalize_sources",
    "_normalize_string_list",
    "_normalize_wikilink",
    "_parse_date",
    "_parse_inline_array",
    "_parse_single_qa_block",
    "_preprocess_yaml_frontmatter",
    "_thread_local_state",
    "create_qa_extractor",
    "discover_notes",
    "parse_frontmatter",
    "parse_note",
    "parse_note_with_repair",
    "parse_qa_pairs",
    "ruamel_yaml",
    "set_thread_qa_extractor",
    "temporarily_disable_llm_extraction",
]


def parse_note(
    file_path: Path,
    qa_extractor: QAExtractorAgent | None = None,
    tolerant_parsing: bool = False,
    content: str | None = None,
) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note and extract metadata and Q/A pairs.

    Args:
        file_path: Path to the markdown file
        qa_extractor: Optional QA extractor agent for LLM-based extraction.
                     If None, uses thread-local state.
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
            msg = f"File does not exist: {resolved_path}"
            raise ParserError(msg)
        if not resolved_path.is_file():
            msg = f"Path is not a file: {resolved_path}"
            raise ParserError(msg)

        # Security: Check file size to prevent DoS attacks
        file_size = resolved_path.stat().st_size
        max_file_size = 10 * 1024 * 1024  # 10MB limit
        if file_size > max_file_size:
            msg = (
                f"File too large: {resolved_path} ({file_size} bytes). "
                f"Maximum allowed size is {max_file_size} bytes."
            )
            raise ParserError(msg)

    except (OSError, RuntimeError) as e:
        msg = f"Cannot resolve path {file_path}: {e}"
        raise ParserError(msg)

    if content is None:
        try:
            content = resolved_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            msg = f"Failed to read file {resolved_path}: {e}"
            raise ParserError(msg)
        except Exception as e:
            logger.exception("unexpected_file_read_error", file=str(resolved_path))
            msg = f"Unexpected error reading {resolved_path}: {e}"
            raise ParserError(msg)

    # Auto-fix unbalanced code fences before parsing
    content = _autofix_unbalanced_code_fences(content, file_path)

    # Parse frontmatter
    metadata = parse_frontmatter(content, file_path)

    # Validate note structure
    enforce_validation = _get_enforce_language_validation()
    if enforce_validation or not tolerant_parsing:
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
        content,
        metadata,
        file_path,
        qa_extractor=qa_extractor,
        tolerant_parsing=tolerant_parsing,
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
    ollama_client: BaseLLMProvider | None = None,
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

        from obsidian_anki_sync.agents.parser_repair import attempt_repair

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
            msg = f"Parse failed and repair unsuccessful: {e}"
            raise ParserError(msg) from e

        # Repair succeeded
        metadata, qa_pairs = result
        logger.info(
            "parser_repair_succeeded",
            file=str(file_path),
            qa_pairs_count=len(qa_pairs),
        )
        return metadata, qa_pairs


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
        source_dirs = [Path()]
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
