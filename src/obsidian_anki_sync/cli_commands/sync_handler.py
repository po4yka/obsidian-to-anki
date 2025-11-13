"""Sync command implementation logic."""

from typing import Any

import typer
from rich.table import Table

from ..anki.client import AnkiClient
from ..config import Config
from ..sync.engine import SyncEngine
from ..sync.progress import ProgressTracker
from ..sync.state_db import StateDB
from ..utils.preflight import run_preflight_checks
from .shared import console


def run_sync(
    config: Config,
    logger: Any,
    dry_run: bool = False,
    incremental: bool = False,
    no_index: bool = False,
    resume: str | None = None,
    no_resume: bool = False,
) -> None:
    """Execute the sync operation.

    Args:
        config: Configuration object
        logger: Logger instance
        dry_run: Preview changes without applying
        incremental: Only process new notes
        no_index: Skip indexing phase
        resume: Session ID to resume
        no_resume: Disable automatic resume

    Raises:
        typer.Exit: On sync failure
    """
    logger.info(
        "sync_started",
        dry_run=dry_run,
        incremental=incremental,
        vault=str(config.vault_path),
    )

    # Run pre-flight checks
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    # Skip Anki check for dry-run mode
    check_anki = not dry_run
    check_llm = True

    passed, results = run_preflight_checks(config, check_anki=check_anki, check_llm=check_llm)

    # Display results
    for result in results:
        if result.passed:
            icon = "[green]✓[/green]"
            color = "green"
        elif result.severity == "warning":
            icon = "[yellow]⚠[/yellow]"
            color = "yellow"
        else:
            icon = "[red]✗[/red]"
            color = "red"

        console.print(f"{icon} {result.name}: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]{result.fix_suggestion}[/dim]")

    console.print()

    # Count errors and warnings
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]

    if errors:
        console.print(f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]")
        console.print("[yellow]Fix the errors above and try again.[/yellow]\n")
        raise typer.Exit(code=1)

    if warnings:
        console.print(f"\n[bold yellow]Pre-flight checks passed with {len(warnings)} warning(s).[/bold yellow]")
        console.print("[dim]You may proceed, but some features may not work as expected.[/dim]\n")
    else:
        console.print("[bold green]All pre-flight checks passed![/bold green]\n")

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            progress_tracker = _handle_progress_tracking(db, resume, no_resume, logger)

            # Show incremental mode info
            if incremental:
                processed_count = len(db.get_processed_note_paths())
                console.print(
                    f"\n[cyan]Incremental mode: Skipping {processed_count} already processed notes[/cyan]\n"
                )

            engine = SyncEngine(config, db, anki, progress_tracker=progress_tracker)
            stats = engine.sync(
                dry_run=dry_run, incremental=incremental, build_index=not no_index
            )

            _display_sync_results(stats, no_index)
            logger.info("sync_completed", stats=stats)

    except Exception as e:
        logger.error("sync_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def _handle_progress_tracking(
    db: StateDB,
    resume: str | None,
    no_resume: bool,
    logger: Any,
) -> ProgressTracker | None:
    """Handle progress tracking setup for resumable syncs.

    Args:
        db: State database
        resume: Session ID to resume
        no_resume: Disable automatic resume
        logger: Logger instance

    Returns:
        Progress tracker instance or None
    """
    if no_resume:
        return None

    session_id = resume

    if not session_id:
        # Check for incomplete syncs
        incomplete = db.get_incomplete_progress()
        if incomplete:
            latest = incomplete[0]
            console.print(
                f"\n[yellow]Found incomplete sync from {latest['updated_at']}[/yellow]"
            )
            console.print(f"  Session: {latest['session_id']}")
            console.print(
                f"  Progress: {latest['notes_processed']}/{latest['total_notes']} notes"
            )

            # Ask user if they want to resume
            resume_choice = typer.confirm("Resume this sync?", default=True)
            if resume_choice:
                session_id = latest["session_id"]

    # Create or resume progress tracker
    if session_id:
        try:
            progress_tracker = ProgressTracker(db, session_id=session_id)
            console.print(f"\n[green]Resuming sync session: {session_id}[/green]\n")
            return progress_tracker
        except ValueError as e:
            console.print(f"\n[red]Cannot resume: {e}[/red]\n")
            raise typer.Exit(code=1)
    else:
        # Start new sync with progress tracking
        progress_tracker = ProgressTracker(db)
        console.print(
            f"\n[cyan]Starting new sync session: {progress_tracker.progress.session_id}[/cyan]\n"
        )
        return progress_tracker


def _display_sync_results(stats: dict[str, Any], no_index: bool) -> None:
    """Display sync results in a formatted table.

    Args:
        stats: Sync statistics
        no_index: Whether indexing was skipped
    """
    # Show index results if available
    if "index" in stats and not no_index:
        console.print("\n[bold cyan]Index Statistics:[/bold cyan]")
        index_table = Table(show_header=True, header_style="bold magenta")
        index_table.add_column("Category", style="cyan")
        index_table.add_column("Metric", style="yellow")
        index_table.add_column("Value", style="green")

        index_stats = stats["index"]

        # Note statistics
        index_table.add_row("Notes", "Total", str(index_stats.get("total_notes", 0)))
        note_status = index_stats.get("note_status", {})
        for status, count in note_status.items():
            index_table.add_row("Notes", status.capitalize(), str(count))

        # Card statistics
        index_table.add_row("Cards", "Total", str(index_stats.get("total_cards", 0)))
        index_table.add_row(
            "Cards",
            "In Obsidian",
            str(index_stats.get("cards_in_obsidian", 0)),
        )
        index_table.add_row(
            "Cards", "In Anki", str(index_stats.get("cards_in_anki", 0))
        )
        index_table.add_row(
            "Cards",
            "In Database",
            str(index_stats.get("cards_in_database", 0)),
        )

        console.print(index_table)
        console.print()

    # Create a Rich table for sync results
    table = Table(title="Sync Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        if key != "index":  # Skip index stats (already displayed)
            table.add_row(key, str(value))

    console.print()
    console.print(table)
