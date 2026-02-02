"""Test run command implementation logic."""

import time
from typing import Any

import typer
from rich.table import Table

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.sync.engine import SyncEngine
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.preflight import run_preflight_checks
from obsidian_anki_sync.utils.progress_display import ProgressDisplay

from .shared import console


def run_test_run(
    config: Config,
    logger: Any,
    count: int = 10,
    dry_run: bool = True,
    index: bool = False,
) -> None:
    """Execute the test-run operation.

    Args:
        config: Configuration object
        logger: Logger instance
        count: Number of random notes to process
        dry_run: Preview changes without applying
        index: Build full vault index before sampling

    Raises:
        typer.Exit: On test-run failure
    """
    start_time = time.time()

    # Log command entry
    logger.info(
        "cli_command_started",
        command="test-run",
        count=count,
        dry_run=dry_run,
        log_level=config.log_level,
        index=index,
        vault_path=str(config.vault_path) if config.vault_path else None,
    )

    logger.info("test_run_started", sample_count=count, dry_run=dry_run)

    # Run pre-flight checks (skip Anki if dry-run)
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    _passed, results = run_preflight_checks(
        config, check_anki=not dry_run, check_llm=True
    )

    # Display results
    for result in results:
        if result.passed:
            icon = "[green]PASS[/green]"
        elif result.severity == "warning":
            icon = "[yellow]WARN[/yellow]"
        else:
            icon = "[red]FAIL[/red]"

        console.print(f"{icon} {result.name}: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]{result.fix_suggestion}[/dim]")

    console.print()

    # Check for errors
    errors = [r for r in results if not r.passed and r.severity == "error"]

    if errors:
        console.print(
            f"\n[bold red]Pre-flight checks failed with {len(errors)} error(s).[/bold red]"
        )
        console.print("[yellow]Fix the errors above and try again.[/yellow]\n")
        raise typer.Exit(code=1)

    console.print("[bold green]All pre-flight checks passed![/bold green]\n")

    try:
        with StateDB(config.db_path) as db, AnkiClient(config.anki_connect_url) as anki:
            # Create progress display for visual feedback
            progress_display = ProgressDisplay(show_reflections=True)

            engine = SyncEngine(config, db, anki)
            engine.set_progress_display(progress_display)

            stats = engine.sync(
                dry_run=dry_run,
                sample_size=count,
                build_index=index,
            )

            mode_text = "dry-run" if dry_run else "applied"
            console.print(
                f"\n[cyan]Processed sample of {count} notes ({mode_text}).[/cyan]"
            )

            # Create a Rich table for results
            table = Table(
                title="Sample Results", show_header=True, header_style="bold magenta"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                table.add_row(key, str(value))

            console.print(table)

            duration = time.time() - start_time
            logger.info(
                "cli_command_completed",
                command="test-run",
                duration=round(duration, 2),
                success=True,
                stats=stats,
            )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "cli_command_failed",
            command="test-run",
            duration=round(duration, 2),
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise
