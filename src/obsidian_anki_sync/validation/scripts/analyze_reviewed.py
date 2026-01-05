#!/usr/bin/env python3
"""Analyze all reviewed notes in 40-Android directory."""

from collections import defaultdict
from pathlib import Path

from obsidian_anki_sync.validation.base import Severity
from obsidian_anki_sync.validation.orchestrator import NoteValidator


def main() -> None:
    """Analyze reviewed notes and generate statistics report."""
    vault_root = Path.cwd()

    # Find all reviewed notes
    android_dir = vault_root / "40-Android"
    reviewed_notes: list[Path] = []

    for md_file in android_dir.glob("q-*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if "status: reviewed" in content:
                reviewed_notes.append(md_file)
        except (OSError, UnicodeDecodeError):
            continue

    # Validate all reviewed notes
    validator = NoteValidator(vault_root)
    results = []

    for i, note in enumerate(reviewed_notes, 1):
        if i % 10 == 0:
            pass
        result = validator.validate_file(note)
        results.append(result)

    # Analyze results
    stats: dict[str, int] = {
        "total": len(results),
        "passed": 0,
        "with_issues": 0,
        "critical": 0,
        "errors": 0,
        "warnings": 0,
        "info": 0,
    }

    files_by_severity: dict[Severity, list[tuple[str, int]]] = {
        Severity.CRITICAL: [],
        Severity.ERROR: [],
        Severity.WARNING: [],
    }

    issue_types: dict[str, int] = defaultdict(int)

    for result in results:
        if not result["success"]:
            continue

        issues = result["issues"]
        has_issues = any(issues.values())

        if has_issues:
            stats["with_issues"] += 1
        else:
            stats["passed"] += 1

        # Count by severity
        for severity, issue_list in issues.items():
            count = len(issue_list)
            if severity == Severity.CRITICAL:
                stats["critical"] += count
                if count > 0:
                    files_by_severity[Severity.CRITICAL].append((result["file"], count))
            elif severity == Severity.ERROR:
                stats["errors"] += count
                if count > 0:
                    files_by_severity[Severity.ERROR].append((result["file"], count))
            elif severity == Severity.WARNING:
                stats["warnings"] += count
                if count > 0:
                    files_by_severity[Severity.WARNING].append((result["file"], count))
            elif severity == Severity.INFO:
                stats["info"] += count

            # Collect issue types
            for issue in issue_list:
                issue_msg = str(issue).split("[")[0].strip()
                issue_types[issue_msg] += 1

    # Print report

    # Files with critical issues
    if files_by_severity[Severity.CRITICAL]:
        for filename, count in sorted(
            files_by_severity[Severity.CRITICAL], key=lambda x: -x[1]
        )[:10]:
            pass
        if len(files_by_severity[Severity.CRITICAL]) > 10:
            pass

    # Files with errors
    if files_by_severity[Severity.ERROR]:
        for filename, count in sorted(
            files_by_severity[Severity.ERROR], key=lambda x: -x[1]
        )[:10]:
            pass
        if len(files_by_severity[Severity.ERROR]) > 10:
            pass

    # Top issue types
    for issue_msg, count in sorted(issue_types.items(), key=lambda x: -x[1])[:10]:
        pass

    # Pass rate assessment
    pass_rate = stats["passed"] / stats["total"] * 100
    if pass_rate >= 80:
        status = "GOOD"
    elif pass_rate >= 50:
        status = "MODERATE"
    else:
        status = "NEEDS ATTENTION"


if __name__ == "__main__":
    main()
