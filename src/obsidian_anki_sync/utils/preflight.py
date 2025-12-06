"""Pre-flight checks for validating environment before sync operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from obsidian_anki_sync.utils.logging import get_logger

from .preflight.checks import (
    check_anki_connectivity,
    check_db_path,
    check_deck_name,
    check_disk_space,
    check_git_repo,
    check_llm_provider,
    check_memory,
    check_network_latency,
    check_note_type,
    check_source_dir,
    check_vault_path,
    check_vault_structure,
)

if TYPE_CHECKING:
    from obsidian_anki_sync.config import Config

    from .preflight.models import CheckResult

logger = get_logger(__name__)


class PreflightChecker:
    """Performs pre-flight checks before running sync operations."""

    def __init__(self, config: Config):
        """Initialize pre-flight checker."""
        self.config = config
        self.results: list[CheckResult] = []

    def run_all_checks(
        self, check_anki: bool = True, check_llm: bool = True
    ) -> tuple[bool, list[CheckResult]]:
        """Run all pre-flight checks."""
        logger.info(
            "preflight_checks_started", check_anki=check_anki, check_llm=check_llm
        )

        self.results = []

        # Critical checks (must pass)
        self.results.append(check_vault_path(self.config))
        self.results.append(check_source_dir(self.config))
        self.results.append(check_db_path(self.config))

        # Provider checks
        if check_llm:
            self.results.append(check_llm_provider(self.config))

        # Anki checks (skip for dry-run and export)
        if check_anki:
            self.results.append(check_anki_connectivity(self.config))

        # Additional checks
        note_type_result = check_note_type(self.config, self.results)
        if note_type_result:
            self.results.append(note_type_result)

        deck_result = check_deck_name(self.config, self.results)
        if deck_result:
            self.results.append(deck_result)

        # Extended checks
        self.results.append(check_git_repo(self.config))
        self.results.append(check_vault_structure(self.config))
        self.results.append(check_disk_space(self.config))
        self.results.append(check_memory())
        if check_anki or check_llm:
            self.results.append(check_network_latency(check_anki, check_llm))

        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        blocking_warnings = [
            r for r in self.results if not r.passed and r.severity == "blocking_warning"
        ]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]

        strict_mode = getattr(self.config, "strict_mode", True)
        all_passed = (
            len(errors) == 0 and len(blocking_warnings) == 0
            if strict_mode
            else len(errors) == 0
        )

        logger.info(
            "preflight_checks_completed",
            passed=all_passed,
            total_checks=len(self.results),
            errors=len(errors),
            blocking_warnings=len(blocking_warnings),
            warnings=len(warnings),
            strict_mode=strict_mode,
        )

        return all_passed, self.results
