"""Logging-related CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from .log_handler import LogAnalyzer, ProblematicNotesArchiver
from .shared import console, get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register log and archival commands on the given Typer app."""

    @app.command(name="analyze-logs")
    def analyze_logs(
        days: Annotated[
            int,
            typer.Option(
                "--days",
                help="Number of days to analyze (default: 7)",
            ),
        ] = 7,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Analyze log files and show error summary."""
        config, logger = get_config_and_logger(config_path, log_level)
        logger.info("analyze_logs_started", days=days)

        analyzer = LogAnalyzer(log_dir=config.project_log_dir)
        summary = analyzer.generate_summary_report(days=days)
        error_analysis = analyzer.analyze_errors(days=days)

        console.print(
            f"\n[bold cyan]Log Analysis Summary (Last {days} days)[/bold cyan]\n"
        )

        table = Table(title="Overall Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Log Entries", str(summary["total_log_entries"]))
        table.add_row("Total Errors", str(error_analysis["total_errors"]))

        for level, count in summary["levels"].items():
            table.add_row(f"Level: {level}", str(count))

        console.print(table)
        console.print()

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
                for file_path, count in list(error_analysis["errors_by_file"].items())[
                    :5
                ]:
                    error_table.add_row(f"  {file_path}", str(count))

            console.print(error_table)
            console.print()

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

    @app.command(name="list-problematic-notes")
    def list_problematic_notes(
        error_type: Annotated[
            str | None,
            typer.Option("--error-type", help="Filter by error type"),
        ] = None,
        category: Annotated[
            str | None,
            typer.Option(
                "--category",
                help="Filter by category (parser_errors, validation_errors, llm_errors, generation_errors, other_errors)",
            ),
        ] = None,
        date: Annotated[
            str | None,
            typer.Option("--date", help="Filter by date (YYYY-MM-DD)"),
        ] = None,
        limit: Annotated[
            int | None,
            typer.Option("--limit", "-n", help="Limit number of results"),
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """List archived problematic notes."""
        config, logger = get_config_and_logger(config_path, log_level)
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

        table = Table(title="Problematic Notes")
        table.add_column("Filename", style="cyan")
        table.add_column("Error Type", style="red")
        table.add_column("Category", style="yellow")
        table.add_column("Date", style="green")
        table.add_column("Path", style="dim")

        for note in notes:
            table.add_row(
                note.get("filename", "unknown"),
                note.get("error_type", "unknown"),
                note.get("category", "unknown"),
                note.get("date", "unknown"),
                note.get("path", "unknown"),
            )

        console.print()
        console.print(table)
        console.print()
        console.print(
            f"[dim]Archived notes are stored in: {config.problematic_notes_dir}[/dim]\n"
        )

        logger.info("list_problematic_notes_completed", count=len(notes))
