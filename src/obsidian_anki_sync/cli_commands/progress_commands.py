"""Progress and index-related CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from .shared import console, get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register progress-related commands on the given Typer app."""

    @app.command(name="index")
    def show_index(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Show vault and Anki card index statistics."""
        from ..sync.state_db import StateDB

        config, logger = get_config_and_logger(config_path, log_level)
        logger.info("show_index_started")

        try:
            with StateDB(config.db_path) as db:
                stats = db.get_index_statistics()

                if stats["total_notes"] == 0 and stats["total_cards"] == 0:
                    console.print(
                        "\n[yellow]Index is empty. Run sync to build the index.[/yellow]"
                    )
                    return

                console.print("\n[bold cyan]Vault & Anki Index:[/bold cyan]\n")

                notes_table = Table(
                    title="Notes Index",
                    show_header=True,
                    header_style="bold magenta",
                )
                notes_table.add_column("Metric", style="cyan")
                notes_table.add_column("Count", style="green")
                notes_table.add_row("Total Notes", str(stats["total_notes"]))
                note_status = stats.get("note_status", {})
                for status, count in sorted(note_status.items()):
                    notes_table.add_row(f"  {status.capitalize()}", str(count))
                console.print(notes_table)
                console.print()

                cards_table = Table(
                    title="Cards Index", show_header=True, header_style="bold magenta"
                )
                cards_table.add_column("Metric", style="cyan")
                cards_table.add_column("Count", style="green")
                cards_table.add_row("Total Cards", str(stats["total_cards"]))
                cards_table.add_row("In Obsidian", str(stats["cards_in_obsidian"]))
                cards_table.add_row("In Anki", str(stats["cards_in_anki"]))
                cards_table.add_row("In Database", str(stats["cards_in_database"]))
                console.print(cards_table)
                console.print()

                card_status = stats.get("card_status", {})
                if card_status:
                    console.print("[bold]Card Status Breakdown:[/bold]")
                    for status, count in sorted(card_status.items()):
                        console.print(f"  [cyan]{status}:[/cyan] {count}")
                    console.print()

            logger.info("show_index_completed")

        except Exception as exc:  # noqa: BLE001
            logger.error("show_index_failed", error=str(exc))
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    @app.command(name="progress")
    def show_progress(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Show recent sync progress and incomplete sessions."""
        from ..sync.state_db import StateDB

        config, logger = get_config_and_logger(config_path, log_level)
        logger.info("show_progress_started")

        try:
            with StateDB(config.db_path) as db:
                incomplete = db.get_incomplete_progress()

                if incomplete:
                    console.print("\n[bold yellow]Incomplete Syncs:[/bold yellow]")
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Phase", style="yellow")
                    table.add_column("Progress", style="green")
                    table.add_column("Updated At", style="blue")

                    for session in incomplete:
                        progress_str = (
                            f"{session['notes_processed']}/{session['total_notes']} notes"
                        )
                        table.add_row(
                            session["session_id"][:8] + "...",
                            session["phase"],
                            progress_str,
                            session["updated_at"],
                        )

                    console.print(table)
                    console.print(
                        "\n[cyan]Resume with: obsidian-anki-sync sync --resume <session-id>[/cyan]"
                    )
                else:
                    console.print("\n[green]No incomplete syncs found.[/green]")

                recent = db.get_all_progress(limit=10)

                if recent:
                    console.print("\n[bold]Recent Syncs:[/bold]")
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Phase", style="yellow")
                    table.add_column("Progress", style="green")
                    table.add_column("Errors", style="red")
                    table.add_column("Started At", style="blue")

                    for session in recent:
                        progress_str = (
                            f"{session['notes_processed']}/{session['total_notes']}"
                        )
                        table.add_row(
                            session["session_id"][:8] + "...",
                            session["phase"],
                            progress_str,
                            str(session.get("errors", 0)),
                            session["started_at"],
                        )

                    console.print(table)
                else:
                    console.print("\n[yellow]No recent syncs found.[/yellow]")

            logger.info("show_progress_completed")

        except Exception as exc:  # noqa: BLE001
            logger.error("show_progress_failed", error=str(exc))
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    @app.command(name="clean-progress")
    def clean_progress(
        session_id: Annotated[
            str | None,
            typer.Option("--session", help="Session ID to clean up"),
        ] = None,
        all_completed: Annotated[
            bool,
            typer.Option("--all-completed", help="Delete all completed sessions"),
        ] = False,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Clean up sync progress records."""
        from ..sync.state_db import StateDB

        config, logger = get_config_and_logger(config_path, log_level)
        logger.info("clean_progress_started")

        try:
            with StateDB(config.db_path) as db:
                if session_id:
                    db.delete_progress(session_id)
                    console.print(
                        f"[green] Deleted progress for session: {session_id}[/green]"
                    )
                    logger.info("progress_deleted", session_id=session_id)

                elif all_completed:
                    all_progress = db.get_all_progress(limit=1000)
                    deleted_count = 0

                    for session in all_progress:
                        if session["phase"] in ("completed", "failed"):
                            db.delete_progress(session["session_id"])
                            deleted_count += 1

                    console.print(
                        f"[green] Deleted {deleted_count} completed sessions[/green]"
                    )
                    logger.info("completed_progress_deleted", count=deleted_count)

                else:
                    console.print(
                        "[yellow]Please specify --session <id> or --all-completed[/yellow]"
                    )

            logger.info("clean_progress_completed")

        except Exception as exc:  # noqa: BLE001
            logger.error("clean_progress_failed", error=str(exc))
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)


