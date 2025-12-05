"""Index and progress command implementations."""

from typing import Any

import typer
from rich.table import Table

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.state_db import StateDB

from .shared import console


def run_show_index(config: Config, logger: Any) -> None:
    """Execute the show-index operation.

    Args:
        config: Configuration object
        logger: Logger instance

    Raises:
        typer.Exit: On show-index failure
    """
    logger.info("show_index_started")

    try:
        with StateDB(config.db_path) as db:
            stats = db.get_index_statistics()

            if stats["total_notes"] == 0 and stats["total_cards"] == 0:
                console.print(
                    "\n[yellow]Index is empty. Run sync to build the index.[/yellow]"
                )
                return

            # Display index statistics
            console.print("\n[bold cyan]Vault & Anki Index:[/bold cyan]\n")

            # Notes table
            notes_table = Table(
                title="Notes Index", show_header=True, header_style="bold magenta"
            )
            notes_table.add_column("Metric", style="cyan")
            notes_table.add_column("Count", style="green")

            notes_table.add_row("Total Notes", str(stats["total_notes"]))

            note_status = stats.get("note_status", {})
            for status, count in sorted(note_status.items()):
                notes_table.add_row(f"  {status.capitalize()}", str(count))

            console.print(notes_table)
            console.print()

            # Cards table
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

            # Card status breakdown
            card_status = stats.get("card_status", {})
            if card_status:
                console.print("[bold]Card Status Breakdown:[/bold]")
                for status, count in sorted(card_status.items()):
                    console.print(f"  [cyan]{status}:[/cyan] {count}")
                console.print()

        logger.info("show_index_completed")

    except Exception as e:
        logger.error("show_index_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_show_progress(config: Config, logger: Any) -> None:
    """Execute the show-progress operation.

    Args:
        config: Configuration object
        logger: Logger instance

    Raises:
        typer.Exit: On show-progress failure
    """
    logger.info("show_progress_started")

    try:
        with StateDB(config.db_path) as db:
            # Get incomplete syncs
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

            # Get recent syncs
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
                        str(session["errors"]),
                        session["started_at"],
                    )

                console.print(table)

        logger.info("show_progress_completed")

    except Exception as e:
        logger.error("show_progress_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def run_clean_progress(
    config: Config,
    logger: Any,
    session_id: str | None,
    all_completed: bool,
) -> None:
    """Execute the clean-progress operation.

    Args:
        config: Configuration object
        logger: Logger instance
        session_id: Specific session ID to delete
        all_completed: Delete all completed sessions

    Raises:
        typer.Exit: On clean-progress failure
    """
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
                # Get all completed sessions and delete them
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

    except Exception as e:
        logger.error("clean_progress_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
