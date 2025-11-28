#!/usr/bin/env python3
"""Analyze all reviewed notes in 40-Android directory."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ..base import Severity
from ..orchestrator import NoteValidator


def main() -> None:
    """Analyze reviewed notes and generate statistics report."""
    vault_root = Path.cwd()

    # Find all reviewed notes
    android_dir = vault_root / "40-Android"
    reviewed_notes: List[Path] = []

    for md_file in android_dir.glob("q-*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if "status: reviewed" in content:
                reviewed_notes.append(md_file)
        except Exception:
            continue

    print(f"Found {len(reviewed_notes)} notes with 'reviewed' status")
    print("Validating...\n")

    # Validate all reviewed notes
    validator = NoteValidator(vault_root)
    results = []

    for i, note in enumerate(reviewed_notes, 1):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(reviewed_notes)}")
        result = validator.validate_file(note)
        results.append(result)

    # Analyze results
    stats: Dict[str, int] = {
        "total": len(results),
        "passed": 0,
        "with_issues": 0,
        "critical": 0,
        "errors": 0,
        "warnings": 0,
        "info": 0,
    }

    files_by_severity: Dict[Severity, List[tuple[str, int]]] = {
        Severity.CRITICAL: [],
        Severity.ERROR: [],
        Severity.WARNING: [],
    }

    issue_types: Dict[str, int] = defaultdict(int)

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
    print("\n" + "=" * 80)
    print("REVIEWED NOTES ANALYSIS - 40-Android")
    print("=" * 80)
    print()

    print("SUMMARY")
    print("-" * 80)
    print(f"Total reviewed notes: {stats['total']}")
    print(
        f"Passed validation:    {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)"
    )
    print(
        f"With issues:          {stats['with_issues']} ({stats['with_issues']/stats['total']*100:.1f}%)"
    )
    print()
    print(f"Critical issues:      {stats['critical']}")
    print(f"Errors:               {stats['errors']}")
    print(f"Warnings:             {stats['warnings']}")
    print(f"Info:                 {stats['info']}")
    print()

    # Files with critical issues
    if files_by_severity[Severity.CRITICAL]:
        print("FILES WITH CRITICAL ISSUES")
        print("-" * 80)
        for filename, count in sorted(
            files_by_severity[Severity.CRITICAL], key=lambda x: -x[1]
        )[:10]:
            print(f"  {filename}: {count} critical")
        if len(files_by_severity[Severity.CRITICAL]) > 10:
            print(f"  ... and {len(files_by_severity[Severity.CRITICAL]) - 10} more")
        print()

    # Files with errors
    if files_by_severity[Severity.ERROR]:
        print("FILES WITH ERRORS")
        print("-" * 80)
        for filename, count in sorted(
            files_by_severity[Severity.ERROR], key=lambda x: -x[1]
        )[:10]:
            print(f"  {filename}: {count} errors")
        if len(files_by_severity[Severity.ERROR]) > 10:
            print(f"  ... and {len(files_by_severity[Severity.ERROR]) - 10} more")
        print()

    # Top issue types
    print("TOP 10 ISSUE TYPES")
    print("-" * 80)
    for issue_msg, count in sorted(issue_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}x - {issue_msg[:70]}")
    print()

    print("=" * 80)

    # Pass rate assessment
    pass_rate = stats["passed"] / stats["total"] * 100
    if pass_rate >= 80:
        status = "GOOD"
    elif pass_rate >= 50:
        status = "MODERATE"
    else:
        status = "NEEDS ATTENTION"

    print(f"Overall status: {status} ({pass_rate:.1f}% pass rate)")
    print("=" * 80)


if __name__ == "__main__":
    main()
