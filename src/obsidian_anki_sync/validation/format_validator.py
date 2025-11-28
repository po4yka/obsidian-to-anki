"""Formatting and style validator for Q&A notes."""

import re
from pathlib import Path
from typing import Any

from .base import BaseValidator, Severity


class FormatValidator(BaseValidator):
    """Validates formatting and style rules for Q&A notes."""

    # Topic to folder mapping
    TOPIC_TO_FOLDER = {
        "algorithms": "20-Algorithms",
        "data-structures": "20-Algorithms",
        "system-design": "30-System-Design",
        "android": "40-Android",
        "kotlin": "70-Kotlin",
        "programming-languages": "70-Kotlin",
        "databases": "50-Backend",
        "networking": "50-Backend",
        "operating-systems": "60-CompSci",
        "concurrency": "60-CompSci",
        "distributed-systems": "30-System-Design",
        "architecture-patterns": "60-CompSci",
        "testing": "60-CompSci",
        "security": "60-CompSci",
        "performance": "60-CompSci",
        "cloud": "60-CompSci",
        "devops-ci-cd": "60-CompSci",
        "tools": "80-Tools",
        "debugging": "80-Tools",
        "ui-ux-accessibility": "60-CompSci",
        "behavioral": "60-CompSci",
        "cs": "60-CompSci",
    }

    def __init__(
        self, content: str, frontmatter: dict[str, Any], filepath: str
    ) -> None:
        """Initialize format validator.

        Args:
            content: Full note content
            frontmatter: Parsed YAML frontmatter
            filepath: Path to the note file
        """
        super().__init__(content, frontmatter, filepath)

    def validate(self) -> list:
        """Perform format validation."""
        self._check_filename_format()
        self._check_filename_language()
        self._check_folder_topic_match()
        self._check_trailing_whitespace()
        self._check_references_section()

        return self.issues

    def _check_filename_format(self) -> None:
        """Check filename matches pattern q-[slug]--[topic]--[difficulty].md."""
        filename = Path(self.filepath).name

        # Pattern: q-[slug]--[topic]--[difficulty].md
        pattern = r"^q-[a-z0-9-]+--[a-z-]+--(?:easy|medium|hard)\.md$"

        if not re.match(pattern, filename):
            self.add_issue(
                Severity.CRITICAL,
                f"Filename '{filename}' doesn't match pattern: "
                "q-[slug]--[topic]--[difficulty].md",
            )
        else:
            self.add_passed("Filename follows naming pattern")

    def _check_filename_language(self) -> None:
        """Check filename is English-only (no Cyrillic)."""
        filename = Path(self.filepath).name

        cyrillic_pattern = re.compile(r"[а-яА-ЯёЁ]")

        if cyrillic_pattern.search(filename):
            self.add_issue(
                Severity.CRITICAL,
                f"Filename '{filename}' contains Cyrillic characters. "
                "Use English-only.",
            )
        else:
            self.add_passed("Filename is English-only")

        # Check for uppercase
        if any(c.isupper() for c in filename.replace(".md", "")):
            self.add_issue(
                Severity.CRITICAL,
                f"Filename '{filename}' contains uppercase letters. "
                "Use lowercase only.",
            )
        else:
            self.add_passed("Filename is lowercase")

    def _check_folder_topic_match(self) -> None:
        """Check file folder matches topic field."""
        if "topic" not in self.frontmatter:
            return

        topic = self.frontmatter["topic"]
        filepath = Path(self.filepath)

        # Extract folder name
        folder = filepath.parent.name

        expected_folder = self.TOPIC_TO_FOLDER.get(topic)

        if expected_folder and folder != expected_folder:
            self.add_issue(
                Severity.CRITICAL,
                f"File in wrong folder. Topic '{topic}' should be in "
                f"'{expected_folder}/', found in '{folder}/'",
            )
        else:
            self.add_passed(f"File in correct folder for topic '{topic}'")

        # Also check filename topic matches YAML topic
        filename = filepath.name
        # Extract topic from filename (between first -- and second --)
        match = re.search(r"--([a-z-]+)--(?:easy|medium|hard)\.md$", filename)
        if match:
            filename_topic = match.group(1)
            if filename_topic != topic:
                self.add_issue(
                    Severity.CRITICAL,
                    f"Filename topic '{filename_topic}' doesn't match "
                    f"YAML topic '{topic}'",
                )
            else:
                self.add_passed("Filename topic matches YAML topic")

    def _check_trailing_whitespace(self) -> None:
        """Check for trailing whitespace."""
        lines = self.content.split("\n")
        lines_with_trailing: list[int] = []

        for i, line in enumerate(lines, start=1):
            # Has trailing space and is not empty
            if line.rstrip() != line and line.strip():
                lines_with_trailing.append(i)

        if lines_with_trailing:
            if len(lines_with_trailing) > 5:
                self.add_issue(
                    Severity.WARNING,
                    f"Trailing whitespace found on {len(lines_with_trailing)} lines "
                    f"(first few: {lines_with_trailing[:3]})",
                )
            else:
                self.add_issue(
                    Severity.WARNING,
                    f"Trailing whitespace found on lines: {lines_with_trailing}",
                )
            # Add auto-fix for trailing whitespace
            self.add_fix(
                f"Remove trailing whitespace from {len(lines_with_trailing)} lines",
                lambda: self._fix_trailing_whitespace(),
                Severity.WARNING,
                safe=True,
            )
        else:
            self.add_passed("No trailing whitespace")

    def _fix_trailing_whitespace(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Remove trailing whitespace."""
        lines = self.content.split("\n")
        fixed_lines = [line.rstrip() for line in lines]
        new_content = "\n".join(fixed_lines)

        return new_content, self.frontmatter

    def _check_references_section(self) -> None:
        """Check References section - should be omitted if empty."""
        # Find References section
        references_match = re.search(r"^## References\s*$", self.content, re.MULTILINE)

        if not references_match:
            # No References section - that's fine
            self.add_passed("No empty References section")
            return

        # Extract content after References until next section
        lines = self.content.split("\n")
        in_references = False
        references_content: list[str] = []

        for line in lines:
            if re.match(r"^## References\s*$", line):
                in_references = True
                continue
            elif in_references and re.match(r"^##? ", line):
                # Next section started
                break
            elif in_references:
                references_content.append(line)

        # Check if empty
        content_text = "\n".join(references_content).strip()
        if not content_text:
            self.add_issue(
                Severity.WARNING,
                "Empty '## References' section found. "
                "Omit this section if there are no references.",
                section="References",
            )
        else:
            self.add_passed("References section has content")
