"""Registry for auto-fix handlers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import yaml

from obsidian_anki_sync.agents.autofix.handlers import (
    AutoFixHandler,
    BrokenRelatedEntryHandler,
    BrokenWikilinkHandler,
    EmptyReferencesHandler,
    MissingRelatedQuestionsHandler,
    MocMismatchHandler,
    SectionOrderHandler,
    TitleFormatHandler,
    TrailingWhitespaceHandler,
    UnbalancedCodeFenceHandler,
)
from obsidian_anki_sync.agents.models import AutoFixIssue, AutoFixResult
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from obsidian_anki_sync.validation.ai_fixer import AIFixer

logger = get_logger(__name__)


class AutoFixRegistry:
    """Registry that manages all auto-fix handlers.

    Provides a unified interface for detecting and fixing issues in notes.
    """

    def __init__(
        self,
        note_index: set[str] | None = None,
        enabled_handlers: list[str] | None = None,
        write_back: bool = False,
        ai_fixer: AIFixer | None = None,
    ):
        """Initialize the registry.

        Args:
            note_index: Set of existing note IDs for link validation
            enabled_handlers: List of handler types to enable (None = all)
            write_back: Whether to write fixes back to source files
            ai_fixer: Optional AIFixer instance for AI-powered fixes
        """
        self.note_index = note_index or set()
        self.write_back = write_back
        self.ai_fixer = ai_fixer

        # Initialize all handlers
        self._handlers: dict[str, AutoFixHandler] = {
            "trailing_whitespace": TrailingWhitespaceHandler(),
            "empty_references": EmptyReferencesHandler(),
            "title_format": TitleFormatHandler(),
            "moc_mismatch": MocMismatchHandler(),
            "section_order": SectionOrderHandler(ai_fixer=ai_fixer),
            "missing_related_questions": MissingRelatedQuestionsHandler(),
            "broken_wikilink": BrokenWikilinkHandler(note_index=self.note_index),
            "broken_related_entry": BrokenRelatedEntryHandler(
                note_index=self.note_index
            ),
            "unbalanced_code_fence": UnbalancedCodeFenceHandler(),
        }

        # Filter to enabled handlers
        if enabled_handlers:
            self._handlers = {
                k: v for k, v in self._handlers.items() if k in enabled_handlers
            }

        logger.debug(
            "autofix_registry_initialized",
            enabled_handlers=list(self._handlers.keys()),
            note_index_size=len(self.note_index),
            write_back=write_back,
        )

    def update_note_index(self, note_index: set[str]) -> None:
        """Update the note index for link validation.

        Args:
            note_index: Updated set of existing note IDs
        """
        self.note_index = note_index

        # Update handlers that use note index
        if "broken_wikilink" in self._handlers:
            self._handlers["broken_wikilink"] = BrokenWikilinkHandler(
                note_index=self.note_index
            )
        if "broken_related_entry" in self._handlers:
            self._handlers["broken_related_entry"] = BrokenRelatedEntryHandler(
                note_index=self.note_index
            )

    def _parse_yaml_frontmatter(self, content: str) -> dict[str, Any] | None:
        """Parse YAML frontmatter from note content.

        Args:
            content: Note content

        Returns:
            Parsed frontmatter dict or None
        """
        if not content.startswith("---"):
            return None

        try:
            end_match = content.find("\n---", 3)
            if end_match == -1:
                return None

            yaml_content = content[4:end_match]
            result = yaml.safe_load(yaml_content)
            if isinstance(result, dict):
                return result
            return None
        except yaml.YAMLError as e:
            logger.warning("yaml_parse_error", error=str(e))
            return None

    def detect_all(
        self, content: str, file_path: Path | None = None
    ) -> list[AutoFixIssue]:
        """Detect all issues using all enabled handlers.

        Args:
            content: Note content to analyze
            file_path: Optional file path for logging

        Returns:
            List of all detected issues
        """
        all_issues: list[AutoFixIssue] = []
        metadata = self._parse_yaml_frontmatter(content)

        for handler_name, handler in self._handlers.items():
            try:
                issues = handler.detect(content, metadata)
                all_issues.extend(issues)

                if issues:
                    logger.debug(
                        "issues_detected",
                        handler=handler_name,
                        issue_count=len(issues),
                        file_path=str(file_path) if file_path else None,
                    )
            except Exception as e:
                logger.warning(
                    "handler_detect_error",
                    handler=handler_name,
                    error=str(e),
                    file_path=str(file_path) if file_path else None,
                )

        return all_issues

    def fix_all(
        self,
        content: str,
        file_path: Path | None = None,
    ) -> AutoFixResult:
        """Detect and fix all issues in content.

        Args:
            content: Note content to analyze and fix
            file_path: Optional file path for write-back

        Returns:
            AutoFixResult with all detected issues and fixes applied
        """
        start_time = time.time()
        original_content = content
        current_content = content
        all_issues: list[AutoFixIssue] = []
        metadata = self._parse_yaml_frontmatter(content)

        # Run each handler in sequence
        for handler_name, handler in self._handlers.items():
            try:
                # Detect issues
                issues = handler.detect(current_content, metadata)

                if issues:
                    # Apply fixes
                    current_content, updated_issues = handler.fix(
                        current_content, issues, metadata
                    )
                    all_issues.extend(updated_issues)

                    # Re-parse metadata if content changed (for subsequent handlers)
                    if current_content != content:
                        metadata = self._parse_yaml_frontmatter(current_content)

                    logger.debug(
                        "handler_executed",
                        handler=handler_name,
                        issues_found=len(issues),
                        issues_fixed=sum(1 for i in updated_issues if i.auto_fixed),
                        file_path=str(file_path) if file_path else None,
                    )
            except Exception as e:
                logger.warning(
                    "handler_fix_error",
                    handler=handler_name,
                    error=str(e),
                    file_path=str(file_path) if file_path else None,
                )

        # Calculate statistics
        issues_fixed = sum(1 for i in all_issues if i.auto_fixed)
        issues_skipped = len(all_issues) - issues_fixed
        file_modified = current_content != original_content

        # Write back to file if enabled and content changed
        if self.write_back and file_modified and file_path:
            try:
                file_path.write_text(current_content, encoding="utf-8")
                logger.info(
                    "autofix_written",
                    file_path=str(file_path),
                    issues_fixed=issues_fixed,
                )
            except Exception as e:
                logger.error(
                    "autofix_write_error",
                    file_path=str(file_path),
                    error=str(e),
                )
                file_modified = False

        result = AutoFixResult(
            file_modified=file_modified,
            issues_found=all_issues,
            issues_fixed=issues_fixed,
            issues_skipped=issues_skipped,
            original_content=original_content if file_modified else None,
            fixed_content=current_content if file_modified else None,
            fix_time=time.time() - start_time,
            write_back_enabled=self.write_back,
        )

        if all_issues:
            logger.info(
                "autofix_completed",
                file_path=str(file_path) if file_path else None,
                issues_found=len(all_issues),
                issues_fixed=issues_fixed,
                issues_skipped=issues_skipped,
                file_modified=file_modified,
                fix_time_ms=round(result.fix_time * 1000, 2),
            )

        return result

    def get_handler(self, handler_type: str) -> AutoFixHandler | None:
        """Get a specific handler by type.

        Args:
            handler_type: Handler type name

        Returns:
            Handler instance or None
        """
        return self._handlers.get(handler_type)

    def fix_until_valid(
        self,
        content: str,
        max_retries: int = 3,
        file_path: Path | None = None,
    ) -> AutoFixResult:
        """Loop fixes until no more issues are found or max retries reached.

        Args:
            content: Note content
            max_retries: Maximum number of fix iterations
            file_path: Optional file path for logging/writing

        Returns:
            Final AutoFixResult
        """
        original_content = content
        current_content = content
        all_issues = []
        total_fixed = 0

        for i in range(max_retries):
            logger.debug(
                "fix_loop_iteration",
                iteration=i + 1,
                max_retries=max_retries,
            )

            # Run detection first to see if we're done
            current_issues = self.detect_all(current_content, file_path)
            if not current_issues:
                logger.debug("fix_loop_complete_no_issues")
                break

            # Run fix pass
            result = self.fix_all(current_content, file_path)
            current_content = result.fixed_content or current_content
            total_fixed += result.issues_fixed
            all_issues.extend(result.issues_found)

            if not result.file_modified:
                logger.debug("fix_loop_stopped_no_changes")
                break

        # Final check
        final_issues = self.detect_all(current_content, file_path)
        file_modified = current_content != original_content

        return AutoFixResult(
            file_modified=file_modified,
            issues_found=final_issues,
            issues_fixed=total_fixed,
            issues_skipped=len(final_issues),
            original_content=original_content if file_modified else None,
            fixed_content=current_content if file_modified else None,
            fix_time=0.0,  # Accumulation not tracked precisely
            write_back_enabled=self.write_back,
        )

    def list_handlers(self) -> list[dict]:
        """List all available handlers.

        Returns:
            List of handler info dicts
        """
        return [
            {
                "type": handler_type,
                "description": handler.description,
            }
            for handler_type, handler in self._handlers.items()
        ]
