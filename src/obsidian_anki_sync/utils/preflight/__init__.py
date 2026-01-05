"""Pre-flight checks for validating environment before sync operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from obsidian_anki_sync.utils.logging import get_logger

from .checks import (
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

    from .models import CheckResult

logger = get_logger(__name__)


class PreflightChecker:
    """Performs pre-flight checks before running sync operations."""

    def __init__(self, config: Config):
        """Initialize pre-flight checker."""
        self.config = config
        self.results: list[CheckResult] = []

    def _record_result(self, result: CheckResult | None) -> None:
        """Append a check result if present."""
        if result:
            self.results.append(result)

    # Legacy granular checks retained for test compatibility
    def _check_git_repo(self) -> None:
        """Run git repository check."""
        self._record_result(check_git_repo(self.config))

    def _check_vault_structure(self) -> None:
        """Run vault structure check."""
        self._record_result(check_vault_structure(self.config))

    def _check_disk_space(self) -> None:
        """Run disk space check."""
        self._record_result(check_disk_space(self.config))

    def _check_memory(self) -> None:
        """Run memory availability check."""
        self._record_result(check_memory())

    def _check_network_latency(
        self, check_anki: bool = True, check_llm: bool = True
    ) -> None:
        """Run network latency checks for Anki and LLM providers."""
        self._record_result(
            check_network_latency(check_anki=check_anki, check_llm=check_llm)
        )

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
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]

        logger.info(
            "preflight_checks_completed",
            passed=len(errors) == 0,
            errors=len(errors),
            warnings=len(warnings),
        )

        return len(errors) == 0, self.results


def run_preflight_checks(
    config: Config,
    check_anki: bool = True,
    check_llm: bool = True,
) -> tuple[bool, list[CheckResult]]:
    """Convenience function to run pre-flight checks."""
    checker = PreflightChecker(config)
    return checker.run_all_checks(check_anki=check_anki, check_llm=check_llm)
