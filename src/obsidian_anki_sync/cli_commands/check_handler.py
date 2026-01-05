"""Check command implementation logic."""

from typing import Any

import typer
from rich.table import Table

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.preflight import run_preflight_checks

from .shared import console


def run_check_setup(
    config: Config,
    logger: Any,
    skip_anki: bool,
    skip_llm: bool,
) -> None:
    """Execute the check-setup operation.

    Args:
        config: Configuration object
        logger: Logger instance
        skip_anki: Skip Anki connectivity check
        skip_llm: Skip LLM provider connectivity check

    Raises:
        typer.Exit: On check-setup failure
    """
    logger.info("check_setup_started")

    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    _passed, results = run_preflight_checks(
        config, check_anki=not skip_anki, check_llm=not skip_llm
    )

    # Display results with detailed formatting
    for result in results:
        if result.passed:
            icon = "[green]PASS[/green]"
        elif result.severity == "warning":
            icon = "[yellow]WARN[/yellow]"
        else:
            icon = "[red]FAIL[/red]"

        console.print(f"{icon} [bold]{result.name}[/bold]: {result.message}")

        if not result.passed and result.fix_suggestion:
            console.print(f"  [dim]TIP: {result.fix_suggestion}[/dim]")

    console.print()

    # Count errors and warnings
    errors = [r for r in results if not r.passed and r.severity == "error"]
    warnings = [r for r in results if not r.passed and r.severity == "warning"]
    passed_checks = [r for r in results if r.passed]

    # Display summary table
    summary_table = Table(
        title="Check Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Status", style="cyan")
    summary_table.add_column("Count", style="green")

    summary_table.add_row("PASS", str(len(passed_checks)))
    summary_table.add_row("WARN", str(len(warnings)))
    summary_table.add_row("FAIL", str(len(errors)))

    console.print(summary_table)
    console.print()

    if errors:
        console.print("[bold red]ERROR: Setup validation failed![/bold red]")
        console.print("[yellow]Fix the errors above before running sync.[/yellow]\n")
        logger.error("check_setup_failed", errors=len(errors), warnings=len(warnings))
        raise typer.Exit(code=1)
    elif warnings:
        console.print(
            "[bold yellow]WARNING: Setup validation passed with warnings.[/bold yellow]"
        )
        console.print(
            "[dim]You may proceed, but some features may not work as expected.[/dim]\n"
        )
        logger.info("check_setup_passed_with_warnings", warnings=len(warnings))
    else:
        console.print(
            "[bold green]SUCCESS: All checks passed! Your setup is ready.[/bold green]\n"
        )
        logger.info("check_setup_passed")
