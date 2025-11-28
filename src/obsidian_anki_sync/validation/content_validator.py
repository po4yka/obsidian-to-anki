"""Content structure validator for Q&A notes."""

import re
from typing import Any

from langdetect import DetectorFactory, detect

from .base import BaseValidator, Severity

# Set seed for consistent language detection results
DetectorFactory.seed = 0


class ContentValidator(BaseValidator):
    """Validates content structure and sections of Q&A notes."""

    # Required sections in order (RU first)
    REQUIRED_SECTIONS = [
        (r"^# Вопрос \(RU\)", "# Вопрос (RU)"),
        (r"^# Question \(EN\)", "# Question (EN)"),
        (r"^## Ответ \(RU\)", "## Ответ (RU)"),
        (r"^## Answer \(EN\)", "## Answer (EN)"),
    ]

    # Optional but recommended sections
    RECOMMENDED_SECTIONS = [
        (r"^## Follow-ups", "## Follow-ups"),
        (r"^## Related Questions", "## Related Questions"),
    ]

    def __init__(
        self, content: str, frontmatter: dict[str, Any], filepath: str
    ) -> None:
        """Initialize content validator.

        Args:
            content: Full note content
            frontmatter: Parsed YAML frontmatter
            filepath: Path to the note file
        """
        super().__init__(content, frontmatter, filepath)

    def validate(self) -> list:
        """Perform content validation."""
        self._check_yaml_separator()
        self._check_required_sections()
        self._check_section_order()
        self._check_empty_sections()
        self._check_language_sections()
        self._check_code_blocks()
        self._check_list_formatting()
        self._check_emoji()
        self._check_related_questions_section()

        return self.issues

    def _check_yaml_separator(self) -> None:
        """Check no extra blank line after YAML closing ---."""
        lines = self.content.split("\n")

        # Find closing --- of YAML
        yaml_end_idx = None
        in_yaml = False

        for i, line in enumerate(lines):
            if line.strip() == "---":
                if not in_yaml:
                    in_yaml = True
                else:
                    yaml_end_idx = i
                    break

        if yaml_end_idx is not None:
            # Check if next non-empty line is immediately after
            if yaml_end_idx + 1 < len(lines):
                next_line = lines[yaml_end_idx + 1]
                if next_line.strip() == "" and yaml_end_idx + 2 < len(lines):
                    following_line = lines[yaml_end_idx + 2]
                    if following_line.strip().startswith("#"):
                        self.add_issue(
                            Severity.WARNING,
                            f"Extra blank line after YAML closing (line {yaml_end_idx + 2}). "
                            "Remove it.",
                            line=yaml_end_idx + 2,
                        )
                        self.add_fix(
                            "Remove extra blank line after YAML",
                            lambda idx=yaml_end_idx: self._fix_extra_blank_after_yaml(
                                idx
                            ),
                            Severity.WARNING,
                            safe=True,
                        )
                        return

        self.add_passed("No extra blank line after YAML")

    def _fix_extra_blank_after_yaml(
        self, yaml_end_idx: int
    ) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Remove extra blank line after YAML closing."""
        lines = self.content.split("\n")
        # Remove the blank line after YAML closing (yaml_end_idx + 1)
        fixed_lines = lines[: yaml_end_idx + 1] + lines[yaml_end_idx + 2 :]
        new_content = "\n".join(fixed_lines)
        return new_content, self.frontmatter

    def _check_required_sections(self) -> None:
        """Check all required sections exist."""
        missing = []

        for pattern, section_name in self.REQUIRED_SECTIONS:
            if not re.search(pattern, self.content, re.MULTILINE):
                missing.append(section_name)

        if missing:
            self.add_issue(
                Severity.CRITICAL,
                f"Missing required sections: {', '.join(missing)}",
            )
        else:
            self.add_passed("All required sections present")

    def _check_section_order(self) -> None:
        """Check sections appear in correct order."""
        lines = self.content.split("\n")
        section_positions: dict[str, int] = {}

        for pattern, section_name in self.REQUIRED_SECTIONS:
            for i, line in enumerate(lines):
                if re.match(pattern, line):
                    section_positions[section_name] = i
                    break

        # Check order
        if len(section_positions) >= 2:
            positions = list(section_positions.values())
            if positions != sorted(positions):
                self.add_issue(
                    Severity.WARNING,
                    "Sections are not in the expected order "
                    "(should be: RU Question, EN Question, RU Answer, EN Answer)",
                )
            else:
                self.add_passed("Sections are in correct order")

    def _check_empty_sections(self) -> None:
        """Check for empty required sections."""
        lines = self.content.split("\n")
        current_section: str | None = None
        section_content: dict[str, list[str]] = {}

        for line in lines:
            # Check if this is a section header
            for pattern, section_name in self.REQUIRED_SECTIONS:
                if re.match(pattern, line):
                    if current_section:
                        section_content[current_section] = section_content.get(
                            current_section, []
                        )
                    current_section = section_name
                    section_content[current_section] = []
                    break
            else:
                # Not a section header, add to current section content
                if current_section and line.strip():
                    section_content[current_section].append(line)

        # Check for empty sections
        for section_name, content in section_content.items():
            if not content or all(not line.strip() for line in content):
                self.add_issue(
                    Severity.CRITICAL,
                    f"Section '{section_name}' is empty",
                    section=section_name,
                )

    def _check_language_sections(self) -> None:
        """Verify EN/RU sections are in correct language using language detection."""
        # Extract sections
        en_question = re.search(
            r"# Question \(EN\)(.*?)(?=^#|\Z)", self.content, re.DOTALL | re.MULTILINE
        )
        ru_question = re.search(
            r"# Вопрос \(RU\)(.*?)(?=^#|\Z)", self.content, re.DOTALL | re.MULTILINE
        )
        en_answer = re.search(
            r"## Answer \(EN\)(.*?)(?=^#|\Z)", self.content, re.DOTALL | re.MULTILINE
        )
        ru_answer = re.search(
            r"## Ответ \(RU\)(.*?)(?=^#|\Z)", self.content, re.DOTALL | re.MULTILINE
        )

        sections_to_check = [
            (en_question, "en", "# Question (EN)"),
            (ru_question, "ru", "# Вопрос (RU)"),
            (en_answer, "en", "## Answer (EN)"),
            (ru_answer, "ru", "## Ответ (RU)"),
        ]

        reversed_sections: list[str] = []

        for match, expected_lang, section_name in sections_to_check:
            if match:
                text = match.group(1).strip()

                # Remove code blocks and links for better detection
                text_no_code = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
                text_no_links = re.sub(r"\[\[.*?\]\]", "", text_no_code)
                text_clean = re.sub(r"http[s]?://\S+", "", text_no_links)

                # Need at least 50 characters for reliable detection
                if len(text_clean.strip()) > 50:
                    try:
                        detected_lang = detect(text_clean)

                        # Check if detected language matches expected
                        if detected_lang != expected_lang:
                            # Map en/ru to common detected variants
                            lang_variants: dict[str, list[str]] = {
                                "en": ["en"],
                                # Ukrainian/Belarusian can be detected as similar
                                "ru": ["ru", "uk", "be"],
                            }

                            if detected_lang not in lang_variants.get(
                                expected_lang, [expected_lang]
                            ):
                                self.add_issue(
                                    Severity.WARNING,
                                    f"Section '{section_name}' appears to be in "
                                    f"'{detected_lang}' not '{expected_lang}'",
                                    section=section_name,
                                )
                                reversed_sections.append(section_name)
                    except Exception:
                        # Not enough text or detection failed
                        pass

        # If both EN and RU sections appear reversed, suggest swap
        if len(reversed_sections) >= 2:
            has_en_reversed = any("EN" in s for s in reversed_sections)
            has_ru_reversed = any("RU" in s for s in reversed_sections)

            if has_en_reversed and has_ru_reversed:
                self.add_issue(
                    Severity.ERROR,
                    "EN and RU sections appear to be swapped!",
                )
                self.add_fix(
                    "Swap EN/RU sections (they appear reversed)",
                    lambda: self._fix_swap_language_sections(),
                    Severity.ERROR,
                    safe=False,  # User should review
                )

        if not reversed_sections:
            self.add_passed("Language sections verified correct")

    def _fix_swap_language_sections(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Swap EN and RU sections (for when they're reversed)."""
        # Extract all sections
        en_question = re.search(
            r"(# Question \(EN\))(.*?)(?=^#|\Z)",
            self.content,
            re.DOTALL | re.MULTILINE,
        )
        ru_question = re.search(
            r"(# Вопрос \(RU\))(.*?)(?=^#|\Z)",
            self.content,
            re.DOTALL | re.MULTILINE,
        )
        en_answer = re.search(
            r"(## Answer \(EN\))(.*?)(?=^#|\Z)",
            self.content,
            re.DOTALL | re.MULTILINE,
        )
        ru_answer = re.search(
            r"(## Ответ \(RU\))(.*?)(?=^#|\Z)",
            self.content,
            re.DOTALL | re.MULTILINE,
        )

        if not all([en_question, ru_question, en_answer, ru_answer]):
            return self.content, self.frontmatter

        new_content = self.content

        # Swap question sections (swap content, keep headers)
        if en_question and ru_question:
            en_q_content = en_question.group(2)
            ru_q_content = ru_question.group(2)

            new_content = new_content.replace(
                f"# Question (EN){en_q_content}",
                f"# Question (EN){ru_q_content}",
            )
            new_content = new_content.replace(
                f"# Вопрос (RU){ru_q_content}",
                f"# Вопрос (RU){en_q_content}",
            )

        # Swap answer sections
        if en_answer and ru_answer:
            en_a_content = en_answer.group(2)
            ru_a_content = ru_answer.group(2)

            new_content = new_content.replace(
                f"## Answer (EN){en_a_content}",
                f"## Answer (EN){ru_a_content}",
            )
            new_content = new_content.replace(
                f"## Ответ (RU){ru_a_content}",
                f"## Ответ (RU){en_a_content}",
            )

        return new_content, self.frontmatter

    def _check_code_blocks(self) -> None:
        """Check code blocks specify language."""
        lines = self.content.split("\n")
        in_code_block = False

        for i, line in enumerate(lines, start=1):
            if line.strip().startswith("```"):
                if not in_code_block:
                    # Opening code block
                    in_code_block = True

                    # Check if language is specified
                    lang_spec = line.strip()[3:].strip()
                    if not lang_spec:
                        self.add_issue(
                            Severity.WARNING,
                            "Code block should specify language "
                            "(e.g., ```kotlin, ```python)",
                            line=i,
                        )
                else:
                    # Closing code block
                    in_code_block = False

        if not in_code_block:
            self.add_passed("Code blocks properly formatted")

    def _check_list_formatting(self) -> None:
        """Check list formatting (exactly one space after dash)."""
        lines = self.content.split("\n")
        issues_found = False

        for i, line in enumerate(lines, start=1):
            # Check for list items
            if re.match(r"^(\s*)-\s\s+\S", line):
                # More than one space after dash
                self.add_issue(
                    Severity.WARNING,
                    "List item has multiple spaces after dash. "
                    "Use exactly one space: '- Item'",
                    line=i,
                )
                issues_found = True

        if not issues_found:
            self.add_passed("List formatting correct")

    def _check_emoji(self) -> None:
        """Check for emoji in content."""
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        )

        if emoji_pattern.search(self.content):
            # Find line with emoji
            lines = self.content.split("\n")
            for i, line in enumerate(lines, start=1):
                if emoji_pattern.search(line):
                    self.add_issue(
                        Severity.CRITICAL,
                        "Emoji are forbidden. Use text equivalents "
                        "(REQUIRED, FORBIDDEN, WARNING, NOTE)",
                        line=i,
                    )
                    return

        self.add_passed("No emoji in content")

    def _check_related_questions_section(self) -> None:
        """Check Related Questions section."""
        if not re.search(r"^## Related Questions", self.content, re.MULTILINE):
            self.add_issue(
                Severity.WARNING,
                "Missing '## Related Questions' section",
                section="Related Questions",
            )
            return

        # Extract section content
        lines = self.content.split("\n")
        in_section = False
        section_content: list[str] = []

        for line in lines:
            if re.match(r"^## Related Questions", line):
                in_section = True
                continue
            elif in_section and re.match(r"^##? ", line):
                # Next section started
                break
            elif in_section:
                section_content.append(line)

        # Check if section is empty
        content_text = "\n".join(section_content).strip()
        if not content_text:
            self.add_issue(
                Severity.WARNING,
                "Related Questions section is empty. "
                "Add 3-8 items or use descriptive bullets.",
                section="Related Questions",
            )
        else:
            self.add_passed("Related Questions section present")
