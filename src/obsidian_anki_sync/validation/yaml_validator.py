"""YAML frontmatter validator for Q&A notes."""

import re
from datetime import datetime
from io import StringIO
from typing import Any

from ruamel.yaml import YAML

from .base import BaseValidator, Severity


class YAMLValidator(BaseValidator):
    """Validates YAML frontmatter fields for Q&A notes."""

    # Required fields for Q&A notes
    REQUIRED_FIELDS = [
        "id",
        "title",
        "aliases",
        "topic",
        "subtopics",
        "question_kind",
        "difficulty",
        "original_language",
        "language_tags",
        "status",
        "moc",
        "related",
        "created",
        "updated",
        "tags",
    ]

    # Valid enum values
    VALID_QUESTION_KINDS = ["coding", "theory", "system-design", "android"]
    VALID_DIFFICULTIES = ["easy", "medium", "hard"]
    VALID_LANGUAGES = ["en", "ru"]
    VALID_STATUSES = ["draft", "reviewed", "ready", "retired"]

    # MOC mapping by topic
    TOPIC_TO_MOC = {
        "algorithms": "moc-algorithms",
        "data-structures": "moc-algorithms",
        "system-design": "moc-system-design",
        "android": "moc-android",
        "kotlin": "moc-kotlin",
        "programming-languages": "moc-kotlin",
        "databases": "moc-backend",
        "networking": "moc-backend",
        "operating-systems": "moc-cs",
        "concurrency": "moc-cs",
        "distributed-systems": "moc-system-design",
        "architecture-patterns": "moc-cs",
        "testing": "moc-cs",
        "security": "moc-cs",
        "performance": "moc-cs",
        "cloud": "moc-cs",
        "devops-ci-cd": "moc-cs",
        "tools": "moc-tools",
        "debugging": "moc-tools",
        "ui-ux-accessibility": "moc-cs",
        "behavioral": "moc-cs",
        "cs": "moc-cs",
    }

    def __init__(
        self,
        content: str,
        frontmatter: dict[str, Any],
        filepath: str,
        valid_topics: list[str],
    ) -> None:
        """Initialize YAML validator.

        Args:
            content: Full note content
            frontmatter: Parsed YAML frontmatter
            filepath: Path to the note file
            valid_topics: List of valid topic values from taxonomy
        """
        super().__init__(content, frontmatter, filepath)
        self.valid_topics = valid_topics

    def validate(self) -> list:
        """Perform YAML validation."""
        self._check_required_fields()
        self._check_id_format()
        self._check_title_format()
        self._check_topic()
        self._check_subtopics()
        self._check_question_kind()
        self._check_difficulty()
        self._check_language_fields()
        self._check_status()
        self._check_moc()
        self._check_related()
        self._check_dates()
        self._check_tags()
        self._check_forbidden_patterns()

        return self.issues

    def _check_required_fields(self) -> None:
        """Check all required fields are present."""
        missing = [
            field for field in self.REQUIRED_FIELDS if field not in self.frontmatter
        ]
        if missing:
            self.add_issue(
                Severity.CRITICAL,
                f"Missing required YAML fields: {', '.join(missing)}",
            )
        else:
            self.add_passed("All required YAML fields present")

    def _check_id_format(self) -> None:
        """Check id field format (Topic + Sequential Number: topic-NNN)."""
        if "id" not in self.frontmatter:
            return

        id_value = str(self.frontmatter["id"])

        # New format: topic-NNN (e.g., android-001, kotlin-042)
        new_pattern = r"^(\w+)-(\d{3})$"

        # Old format for backward compatibility: YYYYMMDD-HHmmss
        old_pattern = r"^\d{8}-\d{6}$"

        new_match = re.match(new_pattern, id_value)
        old_match = re.match(old_pattern, id_value)

        if new_match:
            # Validate new format
            _, number = new_match.groups()
            num = int(number)
            if num < 1 or num > 999:
                self.add_issue(
                    Severity.CRITICAL,
                    f"Invalid id format: '{id_value}' (number must be 001-999)",
                )
            else:
                self.add_passed("ID format valid")
        elif old_match:
            # Old timestamp format (deprecated but still valid)
            try:
                datetime.strptime(id_value.split("-")[0], "%Y%m%d")
                self.add_passed("ID format valid (legacy timestamp)")
            except ValueError:
                self.add_issue(
                    Severity.WARNING,
                    f"ID contains invalid date: '{id_value}'",
                )
        else:
            # Neither format matches
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid id format: '{id_value}' "
                "(expected topic-NNN like android-001 or kotlin-042)",
            )

    def _check_title_format(self) -> None:
        """Check title contains bilingual format."""
        if "title" not in self.frontmatter:
            return

        title = self.frontmatter["title"]
        if " / " not in title:
            self.add_issue(
                Severity.WARNING,
                "Title should contain bilingual format: 'EN / RU'",
            )
        else:
            self.add_passed("Title has bilingual format")

    def _check_topic(self) -> None:
        """Check topic is valid and singular."""
        if "topic" not in self.frontmatter:
            return

        topic = self.frontmatter["topic"]

        # Check if it's a list (multiple topics - FORBIDDEN)
        if isinstance(topic, list):
            self.add_issue(
                Severity.CRITICAL,
                f"Multiple topics forbidden: {topic}. Use exactly ONE topic.",
            )
            return

        # Check if topic is valid
        if topic not in self.valid_topics:
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid topic: '{topic}'. Must be one of: "
                f"{', '.join(self.valid_topics[:5])}... (see TAXONOMY.md)",
            )
        else:
            self.add_passed(f"Topic '{topic}' is valid")

    def _check_subtopics(self) -> None:
        """Check subtopics count."""
        if "subtopics" not in self.frontmatter:
            return

        subtopics = self.frontmatter["subtopics"]

        if not isinstance(subtopics, list):
            self.add_issue(
                Severity.CRITICAL,
                "Subtopics must be an array",
            )
            return

        if len(subtopics) < 1 or len(subtopics) > 3:
            self.add_issue(
                Severity.WARNING,
                f"Subtopics should contain 1-3 values (found {len(subtopics)})",
            )
        else:
            self.add_passed(f"Subtopics count valid ({len(subtopics)} values)")

    def _check_question_kind(self) -> None:
        """Check question_kind is valid."""
        if "question_kind" not in self.frontmatter:
            return

        kind = self.frontmatter["question_kind"]

        if kind not in self.VALID_QUESTION_KINDS:
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid question_kind: '{kind}'. "
                f"Must be one of: {', '.join(self.VALID_QUESTION_KINDS)}",
            )
        else:
            self.add_passed(f"Question kind '{kind}' is valid")

    def _check_difficulty(self) -> None:
        """Check difficulty is valid."""
        if "difficulty" not in self.frontmatter:
            return

        difficulty = self.frontmatter["difficulty"]

        if difficulty not in self.VALID_DIFFICULTIES:
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid difficulty: '{difficulty}'. "
                f"Must be one of: {', '.join(self.VALID_DIFFICULTIES)}",
            )
        else:
            self.add_passed(f"Difficulty '{difficulty}' is valid")

    def _check_language_fields(self) -> None:
        """Check language-related fields."""
        if "original_language" in self.frontmatter:
            orig_lang = self.frontmatter["original_language"]
            if orig_lang not in self.VALID_LANGUAGES:
                self.add_issue(
                    Severity.WARNING,
                    f"Invalid original_language: '{orig_lang}'. Should be 'en' or 'ru'",
                )
            else:
                self.add_passed("Original language is valid")

        if "language_tags" in self.frontmatter:
            lang_tags = self.frontmatter["language_tags"]
            if not isinstance(lang_tags, list):
                self.add_issue(
                    Severity.CRITICAL,
                    "language_tags must be an array",
                )
            elif not all(tag in self.VALID_LANGUAGES for tag in lang_tags):
                self.add_issue(
                    Severity.WARNING,
                    "language_tags contains invalid values. "
                    f"Should only contain: {self.VALID_LANGUAGES}",
                )
            else:
                self.add_passed("Language tags are valid")

    def _check_status(self) -> None:
        """Check status is valid."""
        if "status" not in self.frontmatter:
            return

        status = self.frontmatter["status"]

        if status not in self.VALID_STATUSES:
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid status: '{status}'. "
                f"Must be one of: {', '.join(self.VALID_STATUSES)}",
            )
        else:
            self.add_passed(f"Status '{status}' is valid")

    def _check_moc(self) -> None:
        """Check MOC field format and correctness."""
        if "moc" not in self.frontmatter:
            return

        moc = self.frontmatter["moc"]

        # Check for brackets (FORBIDDEN)
        if isinstance(moc, str) and ("[[" in moc or "]]" in moc):
            self.add_issue(
                Severity.CRITICAL,
                f"MOC field contains brackets: '{moc}'. "
                "Use plain text without brackets.",
            )
            return

        # Check if it's a list (should be single value)
        if isinstance(moc, list):
            self.add_issue(
                Severity.CRITICAL,
                f"MOC must be a single value, not an array: {moc}",
            )
            return

        # Check if MOC matches topic
        if "topic" in self.frontmatter:
            topic = self.frontmatter["topic"]
            expected_moc = self.TOPIC_TO_MOC.get(topic)

            if expected_moc and moc != expected_moc:
                self.add_issue(
                    Severity.WARNING,
                    f"MOC '{moc}' may not match topic '{topic}' "
                    f"(expected: '{expected_moc}')",
                )
            else:
                self.add_passed("MOC field valid and matches topic")
        else:
            self.add_passed("MOC field format valid")

    def _check_related(self) -> None:
        """Check related field format and count."""
        if "related" not in self.frontmatter:
            return

        related = self.frontmatter["related"]

        # Check it's an array
        if not isinstance(related, list):
            self.add_issue(
                Severity.CRITICAL,
                "Related field must be an array",
            )
            return

        # Check for double brackets (FORBIDDEN)
        for item in related:
            if isinstance(item, str) and ("[[" in item or "]]" in item):
                self.add_issue(
                    Severity.CRITICAL,
                    f"Related field contains double brackets: '{item}'. "
                    "Use plain text without brackets.",
                )
                return

        # Check count
        if len(related) < 2:
            self.add_issue(
                Severity.WARNING,
                f"Related field should contain 2-5 items (found {len(related)})",
            )
        elif len(related) > 5:
            self.add_issue(
                Severity.INFO,
                f"Related field has many items ({len(related)}). "
                "Consider limiting to 2-5 most relevant.",
            )
        else:
            self.add_passed(
                f"Related field has appropriate count ({len(related)} items)"
            )

    def _check_dates(self) -> None:
        """Check date formats."""
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"

        for field in ["created", "updated"]:
            if field not in self.frontmatter:
                continue

            date_value = str(self.frontmatter[field])

            if not re.match(date_pattern, date_value):
                self.add_issue(
                    Severity.WARNING,
                    f"Invalid {field} date format: '{date_value}' (expected YYYY-MM-DD)",
                )
            else:
                try:
                    datetime.strptime(date_value, "%Y-%m-%d")
                    self.add_passed(f"Date field '{field}' is valid")
                except ValueError:
                    self.add_issue(
                        Severity.WARNING,
                        f"Invalid {field} date value: '{date_value}'",
                    )

    def _check_tags(self) -> None:
        """Check tags array."""
        if "tags" not in self.frontmatter:
            return

        tags = self.frontmatter["tags"]

        if not isinstance(tags, list):
            self.add_issue(
                Severity.CRITICAL,
                "Tags must be an array",
            )
            return

        # Check for Russian characters (FORBIDDEN)
        cyrillic_pattern = re.compile(r"[а-яА-ЯёЁ]")
        has_russian = False

        for tag in tags:
            if isinstance(tag, str) and cyrillic_pattern.search(tag):
                self.add_issue(
                    Severity.CRITICAL,
                    f"Tags must be English-only. Found Russian in tag: '{tag}'",
                )
                has_russian = True

        if not has_russian:
            self.add_passed("Tags are English-only")

        # Check for difficulty tag
        if "difficulty" in self.frontmatter:
            difficulty = self.frontmatter["difficulty"]
            expected_tag = f"difficulty/{difficulty}"

            if expected_tag not in tags:
                self.add_issue(
                    Severity.WARNING,
                    f"Missing required tag: '{expected_tag}'",
                )
                # Add auto-fix for missing difficulty tag
                self.add_fix(
                    f"Add missing difficulty tag: '{expected_tag}'",
                    lambda: self._fix_add_difficulty_tag(),
                    Severity.WARNING,
                    safe=True,
                )
            else:
                self.add_passed(f"Difficulty tag '{expected_tag}' present")

    def _fix_add_difficulty_tag(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Add missing difficulty tag (preserves YAML formatting)."""
        if "difficulty" not in self.frontmatter or "tags" not in self.frontmatter:
            return self.content, self.frontmatter

        difficulty = self.frontmatter["difficulty"]
        expected_tag = f"difficulty/{difficulty}"

        # Parse with ruamel.yaml to preserve formatting
        parts = self.content.split("---", 2)
        if len(parts) < 3:
            return self.content, self.frontmatter

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 4096  # Prevent line wrapping

        try:
            frontmatter = yaml.load(parts[1])
        except Exception:
            return self.content, self.frontmatter

        # Add tag if missing
        if expected_tag not in frontmatter["tags"]:
            frontmatter["tags"].append(expected_tag)

        # Dump preserving formatting
        stream = StringIO()
        yaml.dump(frontmatter, stream)

        new_content = f"---\n{stream.getvalue()}---{parts[2]}"
        return new_content, dict(frontmatter)

    def _check_forbidden_patterns(self) -> None:
        """Check for forbidden patterns."""
        # Check for emoji in YAML values
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        )

        for key, value in self.frontmatter.items():
            if isinstance(value, str) and emoji_pattern.search(value):
                self.add_issue(
                    Severity.CRITICAL,
                    f"Emoji forbidden in YAML. Found in field '{key}': {value}",
                )
                return

        self.add_passed("No forbidden patterns in YAML")
