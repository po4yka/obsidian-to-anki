"""Generate validation reports."""

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .base import Severity, ValidationIssue


class ReportGenerator:
    """Generates formatted validation reports.

    Supports console output with colors, markdown reports,
    and detailed log files.
    """

    @staticmethod
    def generate_console_report(
        filepath: str,
        all_issues: dict[Severity, list[ValidationIssue]],
        passed_checks: list[str],
        use_colors: bool = True,
    ) -> str:
        """Generate a console-friendly report with colors.

        Args:
            filepath: Path to the validated file
            all_issues: Dict mapping severity to list of issues
            passed_checks: List of checks that passed
            use_colors: Whether to use ANSI colors

        Returns:
            Formatted report string
        """
        if not use_colors:
            # Fallback to old plain text format
            return ReportGenerator._generate_plain_report(
                filepath, all_issues, passed_checks
            )

        # Count issues by severity
        critical = all_issues.get(Severity.CRITICAL, [])
        warnings = all_issues.get(Severity.WARNING, [])
        errors = all_issues.get(Severity.ERROR, [])
        info = all_issues.get(Severity.INFO, [])

        total_issues = len(critical) + len(warnings) + len(errors) + len(info)

        # Overall status with color
        if critical:
            status = Text("CRITICAL ISSUES", style="bold red")
        elif errors:
            status = Text("ERRORS", style="bold yellow")
        elif warnings:
            status = Text("WARNINGS", style="yellow")
        else:
            status = Text("PASSED", style="bold green")

        # Create header panel
        header = Panel(
            f"[bold]{Path(filepath).name}[/bold]\n"
            f"Status: {status.plain} ({total_issues} total issues)",
            title="Validation Report",
            border_style="cyan",
        )

        # Build output
        output = StringIO()
        temp_console = Console(file=output, force_terminal=True)

        temp_console.print(header)
        temp_console.print()

        # Critical issues
        if critical:
            temp_console.print("[bold red]CRITICAL (Must Fix)[/bold red]")
            for issue in critical:
                temp_console.print(f"  [red]x[/red] {issue}")
            temp_console.print()

        # Errors
        if errors:
            temp_console.print("[bold yellow]ERRORS[/bold yellow]")
            for issue in errors:
                temp_console.print(f"  [yellow]![/yellow] {issue}")
            temp_console.print()

        # Warnings
        if warnings:
            temp_console.print("[yellow]WARNINGS[/yellow]")
            for issue in warnings:
                temp_console.print(f"  [yellow]![/yellow] {issue}")
            temp_console.print()

        # Info
        if info:
            temp_console.print("[blue]INFO[/blue]")
            for issue in info:
                temp_console.print(f"  [blue]i[/blue] {issue}")
            temp_console.print()

        # Passed checks
        if not total_issues or passed_checks:
            temp_console.print("[green]PASSED CHECKS[/green]")
            for check in passed_checks[:10]:
                temp_console.print(f"  [green]+[/green] {check}")
            if len(passed_checks) > 10:
                temp_console.print(
                    f"  [dim]... and {len(passed_checks) - 10} more[/dim]"
                )
            temp_console.print()

        return output.getvalue()

    @staticmethod
    def _generate_plain_report(
        filepath: str,
        all_issues: dict[Severity, list[ValidationIssue]],
        passed_checks: list[str],
    ) -> str:
        """Generate plain text report without colors (fallback)."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"Validation Report: {Path(filepath).name}")
        lines.append("=" * 80)
        lines.append("")

        # Count issues by severity
        critical = all_issues.get(Severity.CRITICAL, [])
        warnings = all_issues.get(Severity.WARNING, [])
        errors = all_issues.get(Severity.ERROR, [])
        info = all_issues.get(Severity.INFO, [])

        total_issues = len(critical) + len(warnings) + len(errors) + len(info)

        # Overall status
        if critical:
            status = "CRITICAL ISSUES"
        elif errors:
            status = "ERRORS"
        elif warnings:
            status = "WARNINGS"
        else:
            status = "PASSED"

        lines.append(f"Overall Status: {status} ({total_issues} total issues)")
        lines.append("")

        # Critical issues
        if critical:
            lines.append("CRITICAL (Must Fix)")
            lines.append("-" * 80)
            for issue in critical:
                lines.append(f"  - {issue}")
            lines.append("")

        # Errors
        if errors:
            lines.append("ERRORS")
            lines.append("-" * 80)
            for issue in errors:
                lines.append(f"  - {issue}")
            lines.append("")

        # Warnings
        if warnings:
            lines.append("WARNINGS")
            lines.append("-" * 80)
            for issue in warnings:
                lines.append(f"  - {issue}")
            lines.append("")

        # Info
        if info:
            lines.append("INFO")
            lines.append("-" * 80)
            for issue in info:
                lines.append(f"  - {issue}")
            lines.append("")

        # Passed checks
        if not total_issues or passed_checks:
            lines.append("PASSED CHECKS")
            lines.append("-" * 80)
            for check in passed_checks[:10]:
                lines.append(f"  + {check}")
            if len(passed_checks) > 10:
                lines.append(f"  ... and {len(passed_checks) - 10} more")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def generate_markdown_report(results: list[dict[str, Any]]) -> str:
        """Generate a markdown report for multiple files.

        Args:
            results: List of validation result dicts

        Returns:
            Markdown formatted report
        """
        lines = []
        lines.append("# Validation Report")
        lines.append("")
        lines.append(f"**Total Files Validated:** {len(results)}")
        lines.append("")

        # Summary statistics
        total_critical = sum(
            len(r["issues"].get(Severity.CRITICAL, [])) for r in results
        )
        total_errors = sum(len(r["issues"].get(Severity.ERROR, [])) for r in results)
        total_warnings = sum(
            len(r["issues"].get(Severity.WARNING, [])) for r in results
        )

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Critical Issues: {total_critical}")
        lines.append(f"- Errors: {total_errors}")
        lines.append(f"- Warnings: {total_warnings}")
        lines.append("")

        # Files with issues
        files_with_critical = [r for r in results if r["issues"].get(Severity.CRITICAL)]
        files_with_errors = [r for r in results if r["issues"].get(Severity.ERROR)]
        files_with_warnings = [r for r in results if r["issues"].get(Severity.WARNING)]

        if files_with_critical:
            lines.append("## Files with Critical Issues")
            lines.append("")
            for result in files_with_critical:
                lines.append(f"### {result['file']}")
                lines.append("")
                for issue in result["issues"][Severity.CRITICAL]:
                    lines.append(f"- {issue}")
                lines.append("")

        if files_with_errors:
            lines.append("## Files with Errors")
            lines.append("")
            for result in files_with_errors:
                lines.append(f"### {result['file']}")
                lines.append("")
                for issue in result["issues"][Severity.ERROR]:
                    lines.append(f"- {issue}")
                lines.append("")

        if files_with_warnings:
            lines.append("## Files with Warnings")
            lines.append("")
            for result in files_with_warnings:
                lines.append(f"### {result['file']}")
                lines.append("")
                for issue in result["issues"][Severity.WARNING]:
                    lines.append(f"- {issue}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_summary(results: list[dict[str, Any]], use_colors: bool = True) -> str:
        """Generate a brief summary with optional colors.

        Args:
            results: List of validation result dicts
            use_colors: Whether to use ANSI colors

        Returns:
            Summary string
        """
        total_files = len(results)
        files_with_issues = len([r for r in results if any(r["issues"].values())])
        files_passed = total_files - files_with_issues

        total_critical = sum(
            len(r["issues"].get(Severity.CRITICAL, [])) for r in results
        )
        total_errors = sum(len(r["issues"].get(Severity.ERROR, [])) for r in results)
        total_warnings = sum(
            len(r["issues"].get(Severity.WARNING, [])) for r in results
        )

        if not use_colors:
            return (
                f"Validated {total_files} files: "
                f"{files_passed} passed, {files_with_issues} with issues "
                f"({total_critical} critical, {total_errors} errors, "
                f"{total_warnings} warnings)"
            )

        # Colored summary
        output = StringIO()
        temp_console = Console(file=output, force_terminal=True)

        temp_console.print(f"[bold]Validated {total_files} files:[/bold] ", end="")
        temp_console.print(f"[green]{files_passed} passed[/green], ", end="")
        temp_console.print(f"[yellow]{files_with_issues} with issues[/yellow] ", end="")
        temp_console.print(f"([red]{total_critical} critical[/red], ", end="")
        temp_console.print(f"[yellow]{total_errors} errors[/yellow], ", end="")
        temp_console.print(f"[yellow]{total_warnings} warnings[/yellow])")

        return output.getvalue().strip()

    @staticmethod
    def generate_detailed_log(
        results: list[dict[str, Any]], skipped_count: int = 0
    ) -> str:
        """Generate a detailed plain text log of all validation results.

        Args:
            results: List of validation result dicts
            skipped_count: Number of files skipped (unchanged)

        Returns:
            Detailed log content
        """
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("=" * 80)
        lines.append(f"Validation Report - {timestamp}")
        lines.append("=" * 80)
        lines.append("")

        # Summary first
        total_files = len(results)
        files_with_issues = len([r for r in results if any(r["issues"].values())])
        files_passed = total_files - files_with_issues

        total_critical = sum(
            len(r["issues"].get(Severity.CRITICAL, [])) for r in results
        )
        total_errors = sum(len(r["issues"].get(Severity.ERROR, [])) for r in results)
        total_warnings = sum(
            len(r["issues"].get(Severity.WARNING, [])) for r in results
        )
        total_info = sum(len(r["issues"].get(Severity.INFO, [])) for r in results)

        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total files validated: {total_files}")
        lines.append(f"Files passed: {files_passed}")
        lines.append(f"Files with issues: {files_with_issues}")
        if skipped_count > 0:
            lines.append(f"Files skipped (unchanged): {skipped_count}")
        lines.append("")
        lines.append(f"Critical issues: {total_critical}")
        lines.append(f"Errors: {total_errors}")
        lines.append(f"Warnings: {total_warnings}")
        lines.append(f"Info: {total_info}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")

        # Only show files with issues
        files_with_any_issues = [r for r in results if any(r["issues"].values())]

        if not files_with_any_issues:
            lines.append("All files passed validation!")
            lines.append("")
        else:
            lines.append(f"FILES WITH ISSUES ({len(files_with_any_issues)} files)")
            lines.append("=" * 80)
            lines.append("")

            for result in files_with_any_issues:
                if not result.get("success", True):
                    lines.append(f"ERROR: {result['file']}")
                    lines.append(f"  {result.get('error', 'Unknown error')}")
                    lines.append("")
                    continue

                filepath = result["file"]
                issues = result["issues"]

                lines.append(f"File: {filepath}")
                lines.append("-" * 80)

                # Critical
                critical = issues.get(Severity.CRITICAL, [])
                if critical:
                    lines.append("  CRITICAL:")
                    for issue in critical:
                        lines.append(f"    - {issue}")

                # Errors
                errors = issues.get(Severity.ERROR, [])
                if errors:
                    lines.append("  ERRORS:")
                    for issue in errors:
                        lines.append(f"    - {issue}")

                # Warnings
                warnings = issues.get(Severity.WARNING, [])
                if warnings:
                    lines.append("  WARNINGS:")
                    for issue in warnings:
                        lines.append(f"    - {issue}")

                # Info
                info = issues.get(Severity.INFO, [])
                if info:
                    lines.append("  INFO:")
                    for issue in info:
                        lines.append(f"    - {issue}")

                lines.append("")

        lines.append("=" * 80)
        lines.append(f"End of report - {timestamp}")
        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def write_log_file(
        results: list[dict[str, Any]], log_dir: Path, skipped_count: int = 0
    ) -> Path:
        """Write detailed log to file and return the path.

        Args:
            results: List of validation result dicts
            log_dir: Directory to write log file to
            skipped_count: Number of files skipped (unchanged)

        Returns:
            Path to the written log file
        """
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"validation-{timestamp}.log"

        log_content = ReportGenerator.generate_detailed_log(results, skipped_count)
        log_path.write_text(log_content, encoding="utf-8")

        return log_path
