"""Log analysis command implementations."""

from typing import Any

from rich.table import Table

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.log_analyzer import LogAnalyzer
from obsidian_anki_sync.utils.problematic_notes import ProblematicNotesArchiver

from .shared import console


def run_analyze_logs(config: Config, logger: Any, days: int) -> None:
    """Execute the analyze-logs operation.

    Args:
        config: Configuration object
        logger: Logger instance
        days: Number of days to analyze

    Raises:
        typer.Exit: On analyze-logs failure
    """
    logger.info("analyze_logs_started", days=days)

    analyzer = LogAnalyzer(log_dir=config.project_log_dir)
    summary = analyzer.generate_summary_report(days=days)
    error_analysis = analyzer.analyze_errors(days=days)

    console.print(f"\n[bold cyan]Log Analysis Summary (Last {days} days)[/bold cyan]\n")

    # Overall statistics
    table = Table(title="Overall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Log Entries", str(summary["total_log_entries"]))
    table.add_row("Total Errors", str(error_analysis["total_errors"]))

    for level, count in summary["levels"].items():
        table.add_row(f"Level: {level}", str(count))

    console.print(table)
    console.print()

    # Error breakdown
    if error_analysis["total_errors"] > 0:
        error_table = Table(title="Error Breakdown")
        error_table.add_column("Category", style="cyan")
        error_table.add_column("Count", style="red")

        error_table.add_row("Total Errors", str(error_analysis["total_errors"]))

        if error_analysis["errors_by_type"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]By Error Type[/bold]", "")
            for error_type, count in list(error_analysis["errors_by_type"].items())[
                :10
            ]:
                error_table.add_row(f"  {error_type}", str(count))

        if error_analysis["errors_by_stage"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]By Processing Stage[/bold]", "")
            for stage, count in error_analysis["errors_by_stage"].items():
                error_table.add_row(f"  {stage}", str(count))

        if error_analysis["errors_by_file"]:
            error_table.add_row("", "")
            error_table.add_row("[bold]Most Problematic Files[/bold]", "")
            for file_path, count in list(error_analysis["errors_by_file"].items())[:5]:
                error_table.add_row(f"  {file_path}", str(count))

        console.print(error_table)
        console.print()

        # Recent errors
        if error_analysis["recent_errors"]:
            recent_table = Table(title="Recent Errors (Last 10)")
            recent_table.add_column("Timestamp", style="dim")
            recent_table.add_column("Level", style="red")
            recent_table.add_column("Error Type", style="yellow")
            recent_table.add_column("File", style="cyan")
            recent_table.add_column("Message", style="white", max_width=50)

            for error in error_analysis["recent_errors"][:10]:
                recent_table.add_row(
                    error.get("timestamp", "")[:19],
                    error.get("level", ""),
                    error.get("error_type", "Unknown")[:30],
                    (error.get("file") or "")[:40],
                    error.get("message", "")[:50],
                )

            console.print(recent_table)
    else:
        console.print("[green]No errors found in the specified period![/green]\n")

    logger.info("analyze_logs_completed")


def run_list_problematic_notes(
    config: Config,
    logger: Any,
    error_type: str | None,
    category: str | None,
    date: str | None,
    limit: int | None,
) -> None:
    """Execute the list-problematic-notes operation.

    Args:
        config: Configuration object
        logger: Logger instance
        error_type: Filter by error type
        category: Filter by category
        date: Filter by date (YYYY-MM-DD)
        limit: Limit number of results

    Raises:
        typer.Exit: On list-problematic-notes failure
    """
    logger.info("list_problematic_notes_started")

    archiver = ProblematicNotesArchiver(
        archive_dir=config.problematic_notes_dir,
        enabled=True,
    )

    notes = archiver.get_archived_notes(
        error_type=error_type,
        category=category,
        date=date,
        limit=limit,
    )

    if not notes:
        console.print(
            "[yellow]No problematic notes found matching the criteria.[/yellow]\n"
        )
        return

    console.print(f"\n[bold cyan]Problematic Notes ({len(notes)} found)[/bold cyan]\n")

    table = Table()
    table.add_column("Original Path", style="cyan", max_width=50)
    table.add_column("Error Type", style="red")
    table.add_column("Category", style="yellow")
    table.add_column("Stage", style="dim")
    table.add_column("Timestamp", style="dim")

    for note in notes:
        table.add_row(
            note.get("original_path", "")[:50],
            note.get("error_type", "Unknown"),
            note.get("category", "unknown"),
            note.get("processing_stage", "unknown"),
            note.get("timestamp", "")[:19] if note.get("timestamp") else "",
        )

    console.print(table)
    console.print()

    console.print(
        f"[dim]Archived notes are stored in: {config.problematic_notes_dir}[/dim]\n"
    )

    logger.info("list_problematic_notes_completed", count=len(notes))
