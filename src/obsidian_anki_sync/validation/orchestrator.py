"""Note validation orchestrator.

Coordinates running multiple validators on Q&A notes.
"""

import inspect
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from obsidian_anki_sync.utils.logging import get_logger

from .android_validator import AndroidValidator
from .base import AutoFix, Severity
from .content_validator import ContentValidator
from .format_validator import FormatValidator
from .hash_tracker import HashTracker
from .link_validator import LinkValidator
from .parallel_validator import ParallelConfig, ParallelValidator
from .report_generator import ReportGenerator
from .taxonomy_loader import TaxonomyLoader
from .yaml_validator import YAMLValidator

logger = get_logger(__name__)


class NoteValidator:
    """Main validator orchestrator for Q&A notes.

    Coordinates running all validators on notes and collecting results.
    Supports incremental validation using file hash tracking.
    """

    def __init__(
        self,
        vault_root: Path,
        incremental: bool = False,
        enable_ai: bool = False,
        auto_translate: bool = False,
        enable_ai_fix: bool = False,
        ai_model: str | None = None,
        ai_provider: str = "ollama",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the note validator.

        Args:
            vault_root: Root path of the Obsidian vault
            incremental: Only validate files changed since last run
            enable_ai: Enable AI-powered validation/enhancement
            auto_translate: Enable AI-powered translation
            enable_ai_fix: Enable AI-powered fixes
            ai_model: AI model to use
            ai_provider: AI provider (ollama, openrouter, etc.)
            cache_dir: Directory to store validation cache (defaults to vault_root if None)
        """
        self.vault_root = vault_root
        self.incremental = incremental
        self.enable_ai = enable_ai
        self.auto_translate = auto_translate
        self.enable_ai_fix = enable_ai_fix
        self.ai_model = ai_model
        self.ai_provider = ai_provider
        self.cache_dir = cache_dir

        # Load taxonomy
        taxonomy_file = TaxonomyLoader.find_taxonomy_file(vault_root)
        if taxonomy_file:
            self.taxonomy = TaxonomyLoader(taxonomy_file)
        else:
            self.taxonomy = TaxonomyLoader(vault_root / "TAXONOMY.md")

        self.valid_topics = self.taxonomy.get_valid_topics()

        # Initialize hash tracker for incremental validation
        self.hash_tracker = HashTracker(
            vault_root, cache_dir) if incremental else None

        # AI components (initialized lazily)
        self._ai_fixer = None

    def parse_note(self, filepath: Path) -> tuple[str, dict[str, Any]]:
        """Parse note and extract frontmatter and content.

        Args:
            filepath: Path to the note file

        Returns:
            Tuple of (content, frontmatter dict)

        Raises:
            ValueError: If note has invalid format
        """
        content = filepath.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        if not content.startswith("---"):
            msg = f"No YAML frontmatter found in {filepath}"
            raise ValueError(msg)

        # Find closing ---
        parts = content.split("---", 2)
        if len(parts) < 3:
            msg = f"Invalid YAML frontmatter in {filepath}"
            raise ValueError(msg)

        frontmatter_str = parts[1]
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in {filepath}: {e}"
            raise ValueError(msg)

        return content, frontmatter or {}

    def validate_file(
        self, filepath: Path, collect_fixes: bool = False
    ) -> dict[str, Any]:
        """Validate a single file.

        Args:
            filepath: Path to the note file
            collect_fixes: Whether to collect auto-fix functions

        Returns:
            Validation result dict with keys:
            - file: relative path string
            - filepath: Path object
            - success: bool
            - error: error message if failed
            - issues: dict of severity -> list of issues
            - passed: list of passed check names
            - fixes: list of AutoFix objects (if collect_fixes=True)
        """
        try:
            content, frontmatter = self.parse_note(filepath)
        except Exception as e:
            logger.exception(
                "validation_failed_unexpectedly",
                filepath=str(filepath),
                error=str(e),
            )
            return {
                "file": str(filepath),
                "success": False,
                "error": f"Unexpected error: {e}",
                "issues": {},
                "passed": [],
                "fixes": [],
            }

        # Run all validators
        all_issues: dict[Severity, list] = {}
        all_passed: list[str] = []
        all_fixes: list[AutoFix] = []

        validators = [
            YAMLValidator(content, frontmatter, str(
                filepath), self.valid_topics),
            ContentValidator(content, frontmatter, str(filepath)),
            LinkValidator(content, frontmatter, str(
                filepath), self.vault_root),
            FormatValidator(content, frontmatter, str(filepath)),
            AndroidValidator(content, frontmatter, str(filepath)),
        ]

        # TODO: Add AI validators when Phase 4 is complete
        # if self.enable_ai_fix and self.ai_fixer:
        #     validators.append(AIFixerValidator(...))
        # if self.enable_ai or self.auto_translate:
        #     validators.append(AIValidator(...))

        for validator in validators:
            issues = validator.validate()

            # Group issues by severity
            for issue in issues:
                if issue.severity not in all_issues:
                    all_issues[issue.severity] = []
                all_issues[issue.severity].append(issue)

            # Collect passed checks
            all_passed.extend(validator.passed_checks)

            # Collect fixes if requested
            if collect_fixes:
                all_fixes.extend(validator.fixes)

        return {
            "file": str(filepath.relative_to(self.vault_root)),
            "filepath": filepath,
            "success": True,
            "issues": all_issues,
            "passed": all_passed,
            "fixes": all_fixes,
        }

    def apply_fixes(
        self, filepath: Path, fixes: list[AutoFix]
    ) -> tuple[int, list[str]]:
        """Apply auto-fixes to a file.

        Args:
            filepath: Path to the file
            fixes: List of AutoFix objects to apply

        Returns:
            Tuple of (count of fixes applied, list of fix descriptions)
        """
        if not fixes:
            return 0, []

        applied_fixes: list[str] = []
        original_content = filepath.read_text(encoding="utf-8")
        current_content = original_content

        # Parse current frontmatter
        parts = current_content.split("---", 2)
        if len(parts) < 3:
            return 0, []

        current_frontmatter = yaml.safe_load(parts[1])

        # Apply each fix
        for fix in fixes:
            try:
                # Try to pass current content/frontmatter for cumulative fixes
                sig = inspect.signature(fix.fix_function)
                if len(sig.parameters) >= 2:
                    # New style: fix function accepts (content, frontmatter)
                    new_content, new_frontmatter = fix.fix_function(
                        current_content, current_frontmatter
                    )
                else:
                    # Fix function uses captured self.content
                    new_content, new_frontmatter = fix.fix_function()
                current_content = new_content
                current_frontmatter = new_frontmatter
                applied_fixes.append(fix.description)
            except Exception as e:
                # Log the failure but continue with other fixes
                logger.warning(
                    "fix_application_failed",
                    filepath=str(filepath),
                    fix_description=fix.description,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                continue

        # Only write if content changed
        if current_content != original_content:
            try:
                filepath.write_text(current_content, encoding="utf-8")
            except Exception as e:
                logger.error(
                    "file_write_failed",
                    filepath=str(filepath),
                    fixes_to_apply=len(applied_fixes),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return 0, []

        return len(applied_fixes), applied_fixes

    def validate_directory(
        self,
        directory: Path,
        show_progress: bool = True,
        collect_fixes: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Validate all Q&A notes in a directory with progress bar.

        Args:
            directory: Directory to validate
            show_progress: Whether to show tqdm progress bar
            collect_fixes: Whether to collect auto-fix functions

        Returns:
            Tuple of (list of result dicts, skipped file count)
        """
        results: list[dict[str, Any]] = []
        skipped_count = 0

        # Find all q-*.md files
        md_files = [
            md_file
            for md_file in sorted(directory.rglob("q-*.md"))
            if not any(part.startswith(".") for part in md_file.parts)
        ]

        # Filter to changed files only if incremental mode
        if self.hash_tracker:
            md_files, skipped_count = self.hash_tracker.get_changed_files(
                md_files)

        # Validate with progress bar
        if show_progress and len(md_files) > 1:
            for md_file in tqdm(md_files, desc="Validating", unit="file"):
                result = self.validate_file(
                    md_file, collect_fixes=collect_fixes)
                results.append(result)
                # Update hash tracker cache
                if self.hash_tracker and result["success"]:
                    issues_count = sum(len(v)
                                       for v in result["issues"].values())
                    passed = issues_count == 0
                    self.hash_tracker.update(md_file, passed, issues_count)
        else:
            for md_file in md_files:
                result = self.validate_file(
                    md_file, collect_fixes=collect_fixes)
                results.append(result)
                # Update hash tracker cache
                if self.hash_tracker and result["success"]:
                    issues_count = sum(len(v)
                                       for v in result["issues"].values())
                    passed = issues_count == 0
                    self.hash_tracker.update(md_file, passed, issues_count)

        # Save cache after validation
        if self.hash_tracker:
            self.hash_tracker.save_cache()

        return results, skipped_count

    def validate_directory_parallel(
        self,
        directory: Path,
        show_progress: bool = True,
        collect_fixes: bool = False,
        max_workers: int | None = None,
        batch_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """Validate all Q&A notes in a directory using parallel processing.

        Uses asyncio and ThreadPoolExecutor for parallel file validation,
        which can significantly speed up validation on large vaults.

        Args:
            directory: Directory to validate
            show_progress: Whether to show tqdm progress bar
            collect_fixes: Whether to collect auto-fix functions
            max_workers: Maximum parallel workers (default: CPU count, max 8)
            batch_size: Number of files per batch

        Returns:
            Tuple of (list of result dicts, skipped file count)
        """
        skipped_count = 0

        # Find all q-*.md files
        md_files = [
            md_file
            for md_file in sorted(directory.rglob("q-*.md"))
            if not any(part.startswith(".") for part in md_file.parts)
        ]

        # Filter to changed files only if incremental mode
        if self.hash_tracker:
            md_files, skipped_count = self.hash_tracker.get_changed_files(
                md_files)

        if not md_files:
            return [], skipped_count

        # Create validation function that captures collect_fixes
        def validate_file_wrapper(filepath: Path) -> dict[str, Any]:
            return self.validate_file(filepath, collect_fixes=collect_fixes)

        # Configure parallel validation
        config = ParallelConfig(
            max_workers=max_workers,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Run parallel validation
        validator = ParallelValidator(config)
        results = validator.validate_files_sync(
            md_files, validate_file_wrapper, show_progress
        )

        # Update hash tracker cache for all results
        if self.hash_tracker:
            for result in results:
                if result["success"] and result.get("filepath"):
                    issues_count = sum(len(v)
                                       for v in result["issues"].values())
                    passed = issues_count == 0
                    self.hash_tracker.update(
                        result["filepath"], passed, issues_count)
            self.hash_tracker.save_cache()

        return results, skipped_count

    def generate_report(
        self, results: list[dict[str, Any]], use_colors: bool = True
    ) -> str:
        """Generate a summary report.

        Args:
            results: List of validation result dicts
            use_colors: Whether to use ANSI colors

        Returns:
            Formatted summary string
        """
        return ReportGenerator.format_summary(results, use_colors=use_colors)

    def generate_markdown_report(self, results: list[dict[str, Any]]) -> str:
        """Generate a markdown report.

        Args:
            results: List of validation result dicts

        Returns:
            Markdown formatted report
        """
        return ReportGenerator.generate_markdown_report(results)

    def write_log_file(
        self, results: list[dict[str, Any]], log_dir: Path, skipped_count: int = 0
    ) -> Path:
        """Write detailed log to file.

        Args:
            results: List of validation result dicts
            log_dir: Directory to write log file
            skipped_count: Number of files skipped

        Returns:
            Path to written log file
        """
        return ReportGenerator.write_log_file(results, log_dir, skipped_count)

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        if self.hash_tracker:
            self.hash_tracker.clear_cache()

    def get_cache_stats(self) -> dict[str, int]:
        """Get validation cache statistics.

        Returns:
            Dict with total_cached, passed, and failed counts
        """
        if self.hash_tracker:
            return self.hash_tracker.get_stats()
        return {"total_cached": 0, "passed": 0, "failed": 0}
