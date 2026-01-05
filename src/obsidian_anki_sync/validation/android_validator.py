"""Android-specific validator for Q&A notes."""

from io import StringIO
from typing import Any

from ruamel.yaml import YAML

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseValidator, Severity

logger = get_logger(__name__)


class AndroidValidator(BaseValidator):
    """Validates Android-specific requirements for Q&A notes.

    Only runs validation for notes with topic='android'.
    Checks:
    - Android subtopics are from controlled list
    - Subtopics are mirrored to tags as android/<subtopic>
    - MOC field links to moc-android
    """

    # Android-specific subtopics from TAXONOMY.md
    ANDROID_SUBTOPICS = [
        # UI & Compose
        "ui-compose",
        "ui-views",
        "ui-navigation",
        "ui-state",
        "ui-animation",
        "ui-theming",
        "ui-accessibility",
        "ui-graphics",
        "ui-widgets",
        # Architecture
        "architecture-mvvm",
        "architecture-mvi",
        "architecture-clean",
        "architecture-modularization",
        "di-hilt",
        "di-koin",
        "feature-flags-remote-config",
        # Lifecycle & Components
        "lifecycle",
        "activity",
        "fragment",
        "service",
        "broadcast-receiver",
        "content-provider",
        "app-startup",
        "processes",
        # Concurrency
        "coroutines",
        "flow",
        "threads-sync",
        "background-execution",
        # Data & Storage
        "room",
        "sqldelight",
        "datastore",
        "files-media",
        "serialization",
        "cache-offline",
        # Networking
        "networking-http",
        "websockets",
        "grpc",
        "graphql",
        "connectivity-caching",
        # Performance
        "performance-startup",
        "performance-rendering",
        "performance-memory",
        "performance-battery",
        "strictmode-anr",
        "profiling",
        # Testing
        "testing-unit",
        "testing-instrumented",
        "testing-ui",
        "testing-screenshot",
        "testing-benchmark",
        "testing-mocks",
        # Build & Tooling
        "gradle",
        "build-variants",
        "dependency-management",
        "static-analysis",
        "ci-cd",
        "versioning",
        # Distribution
        "app-bundle",
        "play-console",
        "in-app-updates",
        "in-app-review",
        "billing",
        "instant-apps",
        # Security
        "permissions",
        "keystore-crypto",
        "obfuscation",
        "network-security-config",
        "privacy-sdks",
        # Device Features
        "camera",
        "media",
        "location",
        "bluetooth",
        "nfc",
        "sensors",
        "notifications",
        "intents-deeplinks",
        "shortcuts-widgets",
        # Localization
        "i18n-l10n",
        "a11y",
        # Multiplatform
        "kmp",
        "compose-multiplatform",
        # Form Factors
        "wear",
        "tv",
        "auto",
        "foldables-chromeos",
        # Monitoring
        "analytics",
        "logging-tracing",
        "crash-reporting",
        "monitoring-slo",
        # Engagement
        "ads",
        "engagement-retention",
        "ab-testing",
    ]

    def __init__(
        self, content: str, frontmatter: dict[str, Any], filepath: str
    ) -> None:
        """Initialize Android validator.

        Args:
            content: Full note content
            frontmatter: Parsed YAML frontmatter
            filepath: Path to the note file
        """
        super().__init__(content, frontmatter, filepath)
        self.is_android = frontmatter.get("topic") == "android"

    def validate(self) -> list:
        """Perform Android-specific validation."""
        if not self.is_android:
            # Not an Android note, skip Android-specific checks
            return self.issues

        self._check_android_subtopics()
        self._check_android_tag_mirroring()
        self._check_android_moc()

        return self.issues

    def _check_android_subtopics(self) -> None:
        """Check Android subtopics are from controlled list."""
        if "subtopics" not in self.frontmatter:
            return

        subtopics = self.frontmatter["subtopics"]

        if not isinstance(subtopics, list):
            return

        invalid_subtopics = [st for st in subtopics if st not in self.ANDROID_SUBTOPICS]

        if invalid_subtopics:
            self.add_issue(
                Severity.CRITICAL,
                f"Invalid Android subtopics: {', '.join(invalid_subtopics)}. "
                "Must use values from TAXONOMY.md Android subtopics list.",
            )
        else:
            self.add_passed(f"Android subtopics valid ({len(subtopics)} checked)")

    def _check_android_tag_mirroring(self) -> None:
        """Check Android subtopics are mirrored to tags as android/<subtopic>."""
        if "subtopics" not in self.frontmatter or "tags" not in self.frontmatter:
            return

        subtopics = self.frontmatter["subtopics"]
        tags = self.frontmatter["tags"]

        if not isinstance(subtopics, list) or not isinstance(tags, list):
            return

        # Check each subtopic has corresponding android/<subtopic> tag
        missing_mirrors: list[str] = []

        for subtopic in subtopics:
            expected_tag = f"android/{subtopic}"
            if expected_tag not in tags:
                missing_mirrors.append(expected_tag)

        if missing_mirrors:
            self.add_issue(
                Severity.CRITICAL,
                f"Android subtopics must be mirrored to tags. "
                f"Missing: {', '.join(missing_mirrors)}",
            )
            # Add auto-fix for missing mirrors
            self.add_fix(
                f"Add missing Android tag mirrors: {', '.join(missing_mirrors)}",
                lambda: self._fix_android_tag_mirroring(),
                Severity.CRITICAL,
                safe=True,
            )
        else:
            self.add_passed(
                f"Android subtopic mirroring correct ({len(subtopics)} mirrored)"
            )

    def _fix_android_tag_mirroring(self) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Add missing Android subtopic tags (preserves YAML formatting)."""
        if "subtopics" not in self.frontmatter or "tags" not in self.frontmatter:
            return self.content, self.frontmatter

        subtopics = self.frontmatter["subtopics"]

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
        except (ValueError, TypeError) as e:
            logger.debug("yaml_load_failed", error=str(e))
            return self.content, self.frontmatter

        # Add missing android/* tags
        for subtopic in subtopics:
            expected_tag = f"android/{subtopic}"
            if expected_tag not in frontmatter["tags"]:
                frontmatter["tags"].append(expected_tag)

        # Dump preserving formatting
        stream = StringIO()
        yaml.dump(frontmatter, stream)

        new_content = f"---\n{stream.getvalue()}---{parts[2]}"
        return new_content, dict(frontmatter)

    def _check_android_moc(self) -> None:
        """Check Android notes link to moc-android."""
        if "moc" not in self.frontmatter:
            return

        moc = self.frontmatter["moc"]

        if moc != "moc-android":
            self.add_issue(
                Severity.WARNING,
                f"Android notes should link to 'moc-android' (found: '{moc}')",
            )
        else:
            self.add_passed("Android MOC correct")
