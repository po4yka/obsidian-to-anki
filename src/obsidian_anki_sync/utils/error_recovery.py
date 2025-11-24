"""Error recovery and graceful degradation utilities.

Provides robust error handling, auto-repair capabilities, and fallback strategies
to ensure processing continues even with problematic content.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import NoteMetadata, QAPair
from ..obsidian.parser import parse_note
from ..exceptions import ParserError
from ..utils.logging import get_logger
from ..utils.types import RecoveryResult
from ..agents.specialized_agents import diagnose_and_solve_problems
from .content_preprocessor import ContentPreprocessor, PreprocessingConfig

logger = get_logger(__name__)


class ErrorRecoveryManager:
    """Manages error recovery and graceful degradation for note processing."""

    def __init__(self):
        self.preprocessor = ContentPreprocessor(PreprocessingConfig(
            sanitize_code_fences=True,
            add_missing_languages=True,
            normalize_whitespace=True,
            fix_malformed_frontmatter=True
        ))

    def parse_with_recovery(self, file_path: Path) -> RecoveryResult:
        """
        Parse a note with comprehensive error recovery.

        Args:
            file_path: Path to the markdown file

        Returns:
            RecoveryResult with parsing outcome
        """
        # Strategy 1: Try normal parsing first
        try:
            metadata, qa_pairs = parse_note(file_path, tolerant_parsing=True)
            return RecoveryResult(
                success=True,
                metadata=metadata,
                qa_pairs=qa_pairs,
                method_used="normal_parsing"
            )
        except ParserError as e:
            logger.warning("initial_parsing_failed",
                           file=str(file_path), error=str(e))
            original_error = str(e)

        # Strategy 2: Preprocess and retry
        try:
            processed_content, warnings = self._preprocess_file(file_path)
            if processed_content:
                # Write processed content to temp file and parse
                temp_path = self._create_temp_file(
                    processed_content, file_path)
                try:
                    metadata, qa_pairs = parse_note(
                        temp_path, tolerant_parsing=True)
                    return RecoveryResult(
                        success=True,
                        metadata=metadata,
                        qa_pairs=qa_pairs,
                        method_used="preprocessing_recovery",
                        warnings=warnings,
                        original_error=original_error
                    )
                finally:
                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("preprocessing_recovery_failed", error=str(e))

        # Strategy 3: Specialized agent repair
        try:
            result = self._try_specialized_agents(file_path, original_error)
            if result:
                metadata, qa_pairs = result
                return RecoveryResult(
                    success=True,
                    metadata=metadata,
                    qa_pairs=qa_pairs,
                    method_used="specialized_agent_recovery",
                    warnings=["Content was repaired by specialized agent"],
                    original_error=original_error
                )
        except Exception as e:
            logger.warning("specialized_agent_recovery_failed", error=str(e))

        # Strategy 4: Auto-repair specific issues
        try:
            repaired_content = self._auto_repair_file(file_path)
            if repaired_content:
                temp_path = self._create_temp_file(repaired_content, file_path)
                try:
                    metadata, qa_pairs = parse_note(
                        temp_path, tolerant_parsing=True)
                    return RecoveryResult(
                        success=True,
                        metadata=metadata,
                        qa_pairs=qa_pairs,
                        method_used="auto_repair_recovery",
                        warnings=["Content was automatically repaired"],
                        original_error=original_error
                    )
                finally:
                    temp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("auto_repair_recovery_failed", error=str(e))

        # Strategy 4: Create minimal valid structure
        try:
            metadata, qa_pairs = self._create_minimal_structure(file_path)
            return RecoveryResult(
                success=True,
                metadata=metadata,
                qa_pairs=qa_pairs,
                method_used="minimal_structure_fallback",
                warnings=[
                    "Created minimal valid structure due to parsing failures"],
                original_error=original_error
            )
        except Exception as e:
            logger.error("minimal_structure_fallback_failed", error=str(e))

        # Final failure - return failure result
        return RecoveryResult(
            success=False,
            method_used="all_recovery_failed",
            warnings=["All recovery strategies failed"],
            original_error=original_error
        )

    def _preprocess_file(self, file_path: Path) -> Tuple[Optional[str], List[str]]:
        """Preprocess file content to fix common issues."""
        try:
            content = file_path.read_text(encoding='utf-8')
            processed_content, warnings = self.preprocessor.preprocess_content(
                content)
            return processed_content, warnings
        except Exception as e:
            logger.error("preprocessing_failed", error=str(e))
            return None, [f"Preprocessing failed: {e}"]

    def _auto_repair_file(self, file_path: Path) -> Optional[str]:
        """Apply targeted auto-repair to file content."""
        try:
            content = file_path.read_text(encoding='utf-8')

            # Apply specific repairs
            content = self._repair_code_fences(content)
            content = self._repair_frontmatter(content)
            content = self._repair_missing_sections(content)

            return content
        except Exception as e:
            logger.error("auto_repair_failed", error=str(e))
            return None

    def _repair_code_fences(self, content: str) -> str:
        """Repair code fence issues."""
        lines = content.splitlines()
        repaired_lines = []
        fence_stack = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if fence_stack:
                    # Close the current fence
                    fence_stack.pop()
                    repaired_lines.append(line)
                else:
                    # Start a new fence
                    fence_stack.append(stripped)
                    repaired_lines.append(line)
            else:
                repaired_lines.append(line)

        # Close any remaining open fences
        while fence_stack:
            repaired_lines.append("```")
            fence_stack.pop()

        return '\n'.join(repaired_lines)

    def _repair_frontmatter(self, content: str) -> str:
        """Repair frontmatter issues."""
        lines = content.splitlines()

        # Ensure opening marker
        if not lines or lines[0].strip() != '---':
            lines.insert(0, '---')

        # Find or add closing marker
        found_closing = False
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                found_closing = True
                break

        if not found_closing:
            # Add closing marker before first heading or at end
            insert_pos = len(lines)
            for i, line in enumerate(lines):
                if line.startswith('#') and i > 0:
                    insert_pos = i
                    break
            lines.insert(insert_pos, '---')

        return '\n'.join(lines)

    def _repair_missing_sections(self, content: str) -> str:
        """Add missing required sections based on detected languages."""
        # Extract languages from frontmatter
        languages = self._extract_languages_from_frontmatter(content)

        lines = content.splitlines()
        content_str = '\n'.join(lines)

        # Add missing sections for each language
        for lang in languages:
            question_marker = f"# Question ({lang.upper()})"
            answer_marker = f"## Answer ({lang.upper()})"

            if question_marker not in content_str:
                # Add after frontmatter
                frontmatter_end = self._find_frontmatter_end(lines)
                lines.insert(frontmatter_end + 1, "")
                lines.insert(frontmatter_end + 2, question_marker)
                lines.insert(frontmatter_end + 3, "")
                lines.insert(frontmatter_end + 4,
                             "[Question content to be added]")

            if answer_marker not in content_str:
                # Add after question section
                question_idx = next((i for i, line in enumerate(
                    lines) if question_marker in line), -1)
                if question_idx >= 0:
                    # Find end of question section
                    insert_pos = question_idx + 1
                    while insert_pos < len(lines) and not lines[insert_pos].startswith('##'):
                        insert_pos += 1

                    lines.insert(insert_pos, "")
                    lines.insert(insert_pos + 1, answer_marker)
                    lines.insert(insert_pos + 2, "")
                    lines.insert(insert_pos + 3,
                                 "[Answer content to be added]")

        return '\n'.join(lines)

    def _create_minimal_structure(self, file_path: Path) -> Tuple[NoteMetadata, List[QAPair]]:
        """Create a minimal valid note structure when all else fails."""
        from ..models import NoteMetadata, QAPair

        # Extract basic info from filename
        filename = file_path.stem
        parts = filename.split('--')

        # Create minimal metadata
        metadata = NoteMetadata(
            id=filename,
            title=filename.replace('-', ' ').title(),
            topic=parts[1] if len(parts) > 1 else 'unknown',
            subtopics=[],
            question_kind='theory',
            difficulty='medium',
            language_tags=['en'],
            original_language='en',
            aliases=[],
            source='unknown',
            source_note='Auto-generated minimal structure',
            status='draft',
            related=[],
            created='2024-01-01',
            updated='2024-01-01',
            tags=[]
        )

        # Create minimal QA pair
        qa_pair = QAPair(
            question_en="Question content could not be parsed",
            answer_en="Answer content could not be parsed",
            card_index=1,
            confidence=0.0
        )

        return metadata, [qa_pair]

    def _create_temp_file(self, content: str, original_path: Path) -> Path:
        """Create a temporary file with processed content."""
        import tempfile

        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"obsidian_recovery_{original_path.name}"

        temp_file.write_text(content, encoding='utf-8')
        return temp_file

    def _extract_languages_from_frontmatter(self, content: str) -> List[str]:
        """Extract language tags from frontmatter."""
        languages = []
        lines = content.splitlines()

        in_frontmatter = False
        for line in lines:
            if line.strip() == '---':
                in_frontmatter = not in_frontmatter
                if not in_frontmatter:
                    break
                continue

            if in_frontmatter and line.startswith('language_tags:'):
                # Extract from YAML list format
                match = re.search(r'language_tags:\s*\[(.*?)\]', line)
                if match:
                    languages = [lang.strip().strip('"\'')
                                 for lang in match.group(1).split(', ')]
                    break

        return languages or ['en']  # Default to English

    def _try_specialized_agents(self, file_path: Path, original_error: str) -> Optional[Tuple[NoteMetadata, List[QAPair]]]:
        """Try specialized agents to repair the content."""
        try:
            # Read the original content
            content = file_path.read_text(encoding='utf-8')

            # Prepare error context
            error_context = {
                'error_message': original_error,
                'processing_stage': 'parsing',
                'file_path': str(file_path)
            }

            # Use specialized agents
            agent_results = diagnose_and_solve_problems(content, error_context)

            # Find the first successful result
            for result in agent_results:
                if result.success and result.content:
                    logger.info(
                        "specialized_agent_repair_attempt",
                        confidence=result.confidence,
                        reasoning=result.reasoning[:100] + "..." if len(
                            result.reasoning) > 100 else result.reasoning
                    )

                    # Try to parse the repaired content
                    try:
                        temp_path = self._create_temp_file(
                            result.content, file_path)
                        metadata, qa_pairs = parse_note(temp_path)
                        temp_path.unlink(missing_ok=True)

                        return metadata, qa_pairs

                    except Exception as parse_error:
                        logger.warning(
                            "specialized_agent_repair_parse_failed",
                            parse_error=str(parse_error)
                        )
                        continue

            logger.info("no_specialized_agent_succeeded",
                        attempts=len(agent_results))
            return None

        except Exception as e:
            logger.error("specialized_agents_failed", error=str(e))
            return None

    def _find_frontmatter_end(self, lines: List[str]) -> int:
        """Find the end of frontmatter."""
        for i, line in enumerate(lines):
            if line.strip() == '---' and i > 0:
                return i
        return 0


def parse_note_with_recovery(file_path: Path) -> RecoveryResult:
    """
    Convenience function to parse a note with comprehensive error recovery.

    Args:
        file_path: Path to the markdown file

    Returns:
        RecoveryResult with parsing outcome
    """
    recovery_manager = ErrorRecoveryManager()
    return recovery_manager.parse_with_recovery(file_path)


def parse_note_with_error_recovery(file_path: Path) -> tuple[NoteMetadata, list[QAPair]]:
    """
    Parse an Obsidian note with comprehensive error recovery.

    Uses multiple fallback strategies:
    1. Normal tolerant parsing
    2. Content preprocessing and retry
    3. Auto-repair of common issues
    4. Minimal valid structure creation

    Args:
        file_path: Path to the markdown file

    Returns:
        Tuple of (metadata, qa_pairs)

    Raises:
        ParserError: If all recovery strategies fail
    """
    result = parse_note_with_recovery(file_path)

    if result.success:
        logger.info(
            "parse_with_recovery_successful",
            file=str(file_path),
            method=result.method_used,
            qa_pairs_count=len(result.qa_pairs) if result.qa_pairs else 0,
            warnings=result.warnings
        )

        if result.warnings:
            for warning in result.warnings:
                logger.warning("recovery_warning",
                               warning=warning, file=str(file_path))

        return result.metadata, result.qa_pairs
    else:
        error_msg = f"Failed to parse {file_path} with all recovery strategies"
        if result.original_error:
            error_msg += f": {result.original_error}"

        logger.error(
            "parse_recovery_failed",
            file=str(file_path),
            original_error=result.original_error,
            warnings=result.warnings
        )

        raise ParserError(error_msg)
