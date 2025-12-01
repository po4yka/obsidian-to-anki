"""Individual auto-fix handlers for specific note issues."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from obsidian_anki_sync.agents.models import AutoFixIssue
from obsidian_anki_sync.utils.logging import get_logger

if TYPE_CHECKING:
    from obsidian_anki_sync.validation.ai_fixer import AIFixer

logger = get_logger(__name__)


class AutoFixHandler(ABC):
    """Base class for auto-fix handlers."""

    issue_type: str = "unknown"
    description: str = "Unknown fix handler"

    @abstractmethod
    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        """Detect issues in the content.

        Args:
            content: Note content to analyze
            metadata: Optional parsed YAML frontmatter

        Returns:
            List of detected issues
        """

    @abstractmethod
    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        """Apply fixes to the content.

        Args:
            content: Note content to fix
            issues: Issues detected by detect()
            metadata: Optional parsed YAML frontmatter

        Returns:
            Tuple of (fixed_content, updated_issues with fix_description)
        """


class TrailingWhitespaceHandler(AutoFixHandler):
    """Fix trailing whitespace on lines.

    Preserves markdown line breaks (2+ trailing spaces) which are intentional.
    Only removes single trailing spaces/tabs that are likely unintentional.
    """

    issue_type = "trailing_whitespace"
    description = (
        "Remove trailing spaces and tabs from lines (preserves markdown line breaks)"
    )

    def _is_markdown_line_break(self, line: str) -> bool:
        """Check if trailing spaces are a markdown line break (2+ spaces).

        In Markdown, two or more trailing spaces indicate a <br> line break.
        We should preserve these intentional line breaks.
        """
        stripped = line.rstrip()
        trailing = line[len(stripped) :]
        # Markdown line break requires 2+ trailing spaces
        return len(trailing) >= 2 and trailing.strip() == ""

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues = []
        lines_with_trailing = []

        for i, line in enumerate(content.split("\n"), 1):
            # Skip lines with markdown line breaks (2+ trailing spaces)
            if self._is_markdown_line_break(line):
                continue
            # Check for single trailing space/tab (unintentional)
            if line != line.rstrip():
                lines_with_trailing.append(i)

        if lines_with_trailing:
            issues.append(
                AutoFixIssue(
                    issue_type="trailing_whitespace",
                    severity="warning",
                    description=f"Trailing whitespace on {len(lines_with_trailing)} lines",
                    location=f"lines {lines_with_trailing[:5]}{'...' if len(lines_with_trailing) > 5 else ''}",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues:
            return content, issues

        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            # Preserve markdown line breaks (2+ trailing spaces)
            if self._is_markdown_line_break(line):
                fixed_lines.append(line)
            else:
                fixed_lines.append(line.rstrip())

        fixed_content = "\n".join(fixed_lines)

        for issue in issues:
            issue.auto_fixed = True
            issue.fix_description = (
                "Removed trailing whitespace (preserved markdown line breaks)"
            )

        return fixed_content, issues


class EmptyReferencesHandler(AutoFixHandler):
    """Remove empty References sections."""

    issue_type = "empty_references"
    description = "Remove empty ## References sections"

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues = []

        # Check for empty References section (English)
        pattern_en = r"^## References\s*\n(?=\n|$|#)"
        if re.search(pattern_en, content, re.MULTILINE):
            issues.append(
                AutoFixIssue(
                    issue_type="empty_references",
                    severity="warning",
                    description="Empty '## References' section",
                    location="## References",
                )
            )

        # Check for empty References section (Russian)
        pattern_ru = r"^##\s*[Сс]сылки\s*\(RU\)\s*\n(?=\n|$|#)"
        if re.search(pattern_ru, content, re.MULTILINE):
            issues.append(
                AutoFixIssue(
                    issue_type="empty_references",
                    severity="warning",
                    description="Empty Russian references section",
                    location="## Ссылки (RU)",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues:
            return content, issues

        fixed_content = content

        # Remove empty English References section
        fixed_content = re.sub(
            r"^## References\s*\n+(?=\n|$|#|\Z)",
            "",
            fixed_content,
            flags=re.MULTILINE,
        )

        # Remove empty Russian References section
        fixed_content = re.sub(
            r"^##\s*[Сс]сылки\s*\(RU\)\s*\n+(?=\n|$|#|\Z)",
            "",
            fixed_content,
            flags=re.MULTILINE,
        )

        for issue in issues:
            issue.auto_fixed = True
            issue.fix_description = "Removed empty references section"

        return fixed_content, issues


class TitleFormatHandler(AutoFixHandler):
    """Fix title to have bilingual format 'EN / RU'."""

    issue_type = "title_format"
    description = "Ensure title has bilingual format 'English / Russian'"

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues: list[AutoFixIssue] = []

        if not metadata:
            return issues

        title = metadata.get("title", "")
        if not title:
            return issues

        # Check if title has bilingual separator
        if " / " not in title:
            issues.append(
                AutoFixIssue(
                    issue_type="title_format",
                    severity="warning",
                    description=f"Title missing bilingual format: '{title[:50]}...'",
                    location="title",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues or not metadata:
            return content, issues

        title = metadata.get("title", "")
        aliases = metadata.get("aliases", [])

        if not title or " / " in title:
            return content, issues

        # Try to find Russian translation in aliases
        russian_title = None
        for alias in aliases:
            # Check if alias contains Cyrillic characters
            if re.search(r"[\u0400-\u04FF]", alias):
                russian_title = alias
                break

        if russian_title:
            new_title = f"{title} / {russian_title}"

            # Escape special YAML characters in title
            if ":" in new_title or '"' in new_title:
                new_title_escaped = f'"{new_title}"'
            else:
                new_title_escaped = new_title

            # Replace title in YAML frontmatter
            old_title_escaped = re.escape(title)
            fixed_content = re.sub(
                rf'^(title:\s*)["\']?{old_title_escaped}["\']?\s*$',
                f"\\1{new_title_escaped}",
                content,
                count=1,
                flags=re.MULTILINE,
            )

            for issue in issues:
                issue.auto_fixed = True
                issue.fix_description = "Updated title to bilingual format"
        else:
            # Cannot fix without Russian translation
            for issue in issues:
                issue.auto_fixed = False
                issue.fix_description = "No Russian translation found in aliases"
            fixed_content = content

        return fixed_content, issues


class MocMismatchHandler(AutoFixHandler):
    """Fix MOC field to match topic."""

    issue_type = "moc_mismatch"
    description = "Update MOC field to match topic (moc-{topic})"

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues: list[AutoFixIssue] = []

        if not metadata:
            return issues

        topic = metadata.get("topic", "")
        moc = metadata.get("moc", "")

        if not topic or not moc:
            return issues

        expected_moc = f"moc-{topic}"
        if moc != expected_moc:
            issues.append(
                AutoFixIssue(
                    issue_type="moc_mismatch",
                    severity="warning",
                    description=f"MOC '{moc}' doesn't match topic '{topic}'",
                    location="moc",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues or not metadata:
            return content, issues

        topic = metadata.get("topic", "")
        old_moc = metadata.get("moc", "")

        if not topic:
            return content, issues

        new_moc = f"moc-{topic}"

        # Replace moc in YAML frontmatter
        fixed_content = re.sub(
            rf"^(moc:\s*){re.escape(old_moc)}\s*$",
            f"\\1{new_moc}",
            content,
            count=1,
            flags=re.MULTILINE,
        )

        for issue in issues:
            issue.auto_fixed = True
            issue.fix_description = f"Changed MOC from '{old_moc}' to '{new_moc}'"

        return fixed_content, issues


class SectionOrderHandler(AutoFixHandler):
    """Detect section order issues (RU Q, EN Q, RU A, EN A).

    This handler only DETECTS section order issues.
    If AIFixer is available, it can attempt to fix it.
    """

    issue_type = "section_order"
    description = "Detect sections not in expected order"

    EXPECTED_ORDER = [
        r"#\s*[Вв]опрос\s*\(RU\)",  # Russian Question
        r"#\s*Question\s*\(EN\)",  # English Question
        r"##\s*[Оо]твет\s*\(RU\)",  # Russian Answer
        r"##\s*Answer\s*\(EN\)",  # English Answer
    ]

    def __init__(self, ai_fixer: AIFixer | None = None):
        self.ai_fixer = ai_fixer

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues: list[AutoFixIssue] = []

        # Find positions of each section
        positions = []
        for pattern in self.EXPECTED_ORDER:
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                positions.append(match.start())
            else:
                positions.append(-1)

        # Check if sections are in correct order
        valid_positions = [p for p in positions if p >= 0]
        if valid_positions != sorted(valid_positions):
            issues.append(
                AutoFixIssue(
                    issue_type="section_order",
                    severity="warning",
                    description="Sections not in expected order (RU Q, EN Q, RU A, EN A)",
                    location="content structure",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues:
            return content, issues

        if self.ai_fixer and self.ai_fixer.available:
            try:
                fixed_content, changes = self.ai_fixer.fix_note_structure(content)
                if fixed_content != content:
                    for issue in issues:
                        issue.auto_fixed = True
                        issue.fix_description = (
                            f"AI reorganized structure: {', '.join(changes)}"
                        )
                    return fixed_content, issues
            except Exception as e:
                logger.warning("ai_structure_fix_failed", error=str(e))

        # Fallback if no AI or AI failed
        for issue in issues:
            issue.auto_fixed = False
            issue.fix_description = "Manual fix required (AI fixer disabled or failed)"

        return content, issues


class MissingRelatedQuestionsHandler(AutoFixHandler):
    """Add Related Questions section from YAML related field."""

    issue_type = "missing_related_questions"
    description = "Add ## Related Questions section from YAML related field"

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues: list[AutoFixIssue] = []

        if not metadata:
            return issues

        related = metadata.get("related", [])
        if not related:
            return issues

        # Check if Related Questions section exists
        has_related_section = bool(
            re.search(
                r"^##\s*Related\s*Questions", content, re.MULTILINE | re.IGNORECASE
            )
        )

        if not has_related_section:
            issues.append(
                AutoFixIssue(
                    issue_type="missing_related_questions",
                    severity="warning",
                    description=f"Missing '## Related Questions' section ({len(related)} related notes in YAML)",
                    location="end of file",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues or not metadata:
            return content, issues

        related = metadata.get("related", [])
        if not related:
            return content, issues

        # Build Related Questions section
        related_links = [f"- [[{note}]]" for note in related if note]
        related_section = "\n## Related Questions\n\n" + "\n".join(related_links) + "\n"

        # Append to end of content
        fixed_content = content.rstrip() + "\n" + related_section

        for issue in issues:
            issue.auto_fixed = True
            issue.fix_description = (
                f"Added Related Questions section with {len(related)} links"
            )

        return fixed_content, issues


class BrokenWikilinkHandler(AutoFixHandler):
    """Remove broken wikilinks from content."""

    issue_type = "broken_wikilink"
    description = "Remove wikilinks to non-existent notes"

    # Common broken patterns that should be removed
    BROKEN_PATTERNS = [
        r"c--kotlin--medium",
        r"c-concepts--kotlin--medium",
        r"c-await--kotlin--medium",
        r"c-variable--programming-languages--easy",
        r"c--android--hard",
    ]

    def __init__(self, note_index: set[str] | None = None):
        """Initialize with optional note index.

        Args:
            note_index: Set of existing note IDs in the vault
        """
        self.note_index = note_index or set()

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues = []

        # Find all wikilinks
        wikilinks = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content)

        broken_links = []
        for link in wikilinks:
            # Check against known broken patterns
            for pattern in self.BROKEN_PATTERNS:
                if re.match(pattern, link):
                    broken_links.append(link)
                    break
            else:
                # Check against note index if available
                if self.note_index and link not in self.note_index:
                    # Only flag if it looks like a note reference (not a concept)
                    if link.startswith("q-") and "--" in link:
                        broken_links.append(link)

        if broken_links:
            issues.append(
                AutoFixIssue(
                    issue_type="broken_wikilink",
                    severity="error",
                    description=f"Found {len(broken_links)} broken wikilinks",
                    location=f"links: {broken_links[:3]}{'...' if len(broken_links) > 3 else ''}",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues:
            return content, issues

        fixed_content = content
        removed_count = 0

        # Remove known broken patterns
        for pattern in self.BROKEN_PATTERNS:
            # Remove wikilinks matching the pattern
            before = fixed_content
            fixed_content = re.sub(
                rf"\[\[{pattern}(?:\|[^\]]+)?\]\]",
                "",
                fixed_content,
            )
            if before != fixed_content:
                removed_count += 1

        # Clean up empty list items
        fixed_content = re.sub(r"^-\s*$\n", "", fixed_content, flags=re.MULTILINE)

        # Clean up multiple blank lines
        fixed_content = re.sub(r"\n{3,}", "\n\n", fixed_content)

        for issue in issues:
            if removed_count > 0:
                issue.auto_fixed = True
                issue.fix_description = f"Removed {removed_count} broken wikilinks"
            else:
                issue.auto_fixed = False
                issue.fix_description = "No known broken patterns found to remove"

        return fixed_content, issues


class BrokenRelatedEntryHandler(AutoFixHandler):
    """Remove non-existent entries from YAML related field."""

    issue_type = "broken_related_entry"
    description = "Remove non-existent notes from YAML related field"

    def __init__(self, note_index: set[str] | None = None):
        """Initialize with optional note index.

        Args:
            note_index: Set of existing note IDs in the vault
        """
        self.note_index = note_index or set()

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        issues: list[AutoFixIssue] = []

        if not metadata or not self.note_index:
            return issues

        related = metadata.get("related", [])
        if not related:
            return issues

        broken_entries = []
        for entry in related:
            if entry and entry not in self.note_index:
                # Skip concept references (c-*) - they may be valid
                if not entry.startswith("c-") and not entry.startswith("moc-"):
                    broken_entries.append(entry)

        if broken_entries:
            issues.append(
                AutoFixIssue(
                    issue_type="broken_related_entry",
                    severity="error",
                    description=f"YAML related contains {len(broken_entries)} non-existent notes",
                    location=f"related: {broken_entries[:3]}{'...' if len(broken_entries) > 3 else ''}",
                )
            )

        return issues

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues or not metadata or not self.note_index:
            return content, issues

        related = metadata.get("related", [])
        if not related:
            return content, issues

        # Filter to only existing entries
        valid_entries = []
        for entry in related:
            if entry in self.note_index or entry.startswith(("c-", "moc-")):
                valid_entries.append(entry)

        removed_count = len(related) - len(valid_entries)

        if removed_count > 0:
            # Build new related field
            if valid_entries:
                new_related = "related: [" + ", ".join(valid_entries) + "]"
            else:
                new_related = "related: []"

            # Replace in content using regex
            # Handle both inline and multi-line formats
            fixed_content = re.sub(
                r"^related:\s*\[.*?\]$",
                new_related,
                content,
                count=1,
                flags=re.MULTILINE,
            )

            # Also handle multi-line format
            if fixed_content == content:
                fixed_content = re.sub(
                    r"^related:\s*\n(?:\s*-\s*.*\n)+",
                    new_related + "\n",
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )

            for issue in issues:
                issue.auto_fixed = True
                issue.fix_description = (
                    f"Removed {removed_count} broken entries from related field"
                )
        else:
            fixed_content = content
            for issue in issues:
                issue.auto_fixed = False
                issue.fix_description = "No broken entries found to remove"

        return fixed_content, issues


class UnknownErrorHandler(AutoFixHandler):
    """Handler for logging unknown errors for future analysis.

    This handler doesn't fix anything but logs the error details
    to a file so we can implement new handlers later.
    """

    issue_type = "unknown_error"
    description = "Log unknown errors for future automation"

    def __init__(self, log_file: str = "unknown_errors.log"):
        self.log_file = log_file

    def detect(self, content: str, metadata: dict | None = None) -> list[AutoFixIssue]:
        # This handler is special - it doesn't detect issues in content
        # It's called explicitly when other handlers fail
        return []

    def fix(
        self, content: str, issues: list[AutoFixIssue], metadata: dict | None = None
    ) -> tuple[str, list[AutoFixIssue]]:
        if not issues:
            return content, issues

        # Log issues to file
        try:
            with open(self.log_file, "a") as f:
                for issue in issues:
                    f.write("--- Unknown Error ---\n")
                    f.write(f"Type: {issue.issue_type}\n")
                    f.write(f"Description: {issue.description}\n")
                    f.write(f"Location: {issue.location}\n")
                    f.write(f"Content snippet: {content[:200]}...\n\n")

            for issue in issues:
                issue.auto_fixed = False
                issue.fix_description = f"Logged to {self.log_file} for analysis"

        except Exception as e:
            logger.error("failed_to_log_unknown_error", error=str(e))

        return content, issues
