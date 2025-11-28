"""CLI commands for vault validation."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .shared import get_config_and_logger

validate_app = typer.Typer(
    name="validate",
    help="Vault validation commands for Q&A notes.",
    no_args_is_help=True,
)

console = Console()


@validate_app.command(name="all")
def validate_all(
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Apply safe auto-fixes"),
    ] = False,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental", help="Only validate files changed since last run"
        ),
    ] = False,
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel", "-p", help="Use parallel processing for faster validation"
        ),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (default: CPU count, max 8)",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Generate markdown report to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show per-file details"),
    ] = False,
    no_colors: Annotated[
        bool,
        typer.Option("--no-colors", help="Disable colored output"),
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
    """Validate all Q&A notes in the vault."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.validation import NoteValidator, ReportGenerator, Severity

    logger.info(
        "validate_all_started", incremental=incremental, fix=fix, parallel=parallel
    )

    vault_path = config.vault_path / config.source_dir
    if not vault_path.exists():
        console.print(f"[bold red]Error:[/bold red] Vault path not found: {vault_path}")
        raise typer.Exit(code=1)

    mode_str = "parallel" if parallel else "sequential"
    console.print(f"\n[cyan]Validating notes in:[/cyan] {vault_path} ({mode_str})")

    validator = NoteValidator(
        vault_root=vault_path,
        incremental=incremental,
        cache_dir=config.get_data_path(),
    )

    if parallel:
        results, skipped_count = validator.validate_directory_parallel(
            vault_path,
            show_progress=True,
            collect_fixes=fix,
            max_workers=workers,
        )
    else:
        results, skipped_count = validator.validate_directory(
            vault_path,
            show_progress=True,
            collect_fixes=fix,
        )

    # Apply fixes if requested
    fixes_applied = 0
    if fix:
        console.print("\n[cyan]Applying auto-fixes...[/cyan]")
        for result in results:
            if result.get("fixes") and result.get("filepath"):
                safe_fixes = [f for f in result["fixes"] if f.safe]
                if safe_fixes:
                    count, descriptions = validator.apply_fixes(
                        result["filepath"], safe_fixes
                    )
                    fixes_applied += count
                    if verbose and descriptions:
                        for desc in descriptions:
                            console.print(f"  [green]+[/green] {desc}")

        if fixes_applied:
            console.print(f"[green]Applied {fixes_applied} auto-fix(es)[/green]")

    # Show summary
    console.print("\n" + validator.generate_report(results, use_colors=not no_colors))

    if skipped_count > 0:
        console.print(f"[dim]Skipped {skipped_count} unchanged file(s)[/dim]")

    # Show per-file details if verbose
    if verbose:
        files_with_issues = [r for r in results if r.get("issues")]
        if files_with_issues:
            console.print("\n[bold]Files with issues:[/bold]")
            for result in files_with_issues:
                report_text = ReportGenerator.generate_console_report(
                    result["file"],
                    result["issues"],
                    result.get("passed", []),
                    use_colors=not no_colors,
                )
                console.print(report_text)

    # Generate markdown report if requested
    if report:
        md_report = validator.generate_markdown_report(results)
        report.write_text(md_report, encoding="utf-8")
        console.print(f"\n[green]Report written to:[/green] {report}")

    # Write log file to data_dir (not vault)
    log_dir = config.get_validation_log_dir()
    log_path = validator.write_log_file(results, log_dir, skipped_count)
    console.print(f"[dim]Log written to: {log_path}[/dim]")

    # Exit with error code if critical issues
    critical_count = sum(len(r["issues"].get(Severity.CRITICAL, [])) for r in results)
    if critical_count > 0:
        console.print(
            f"\n[bold red]Found {critical_count} critical issue(s) - "
            "please fix before syncing[/bold red]"
        )
        raise typer.Exit(code=1)

    logger.info("validate_all_completed", files=len(results), fixes=fixes_applied)


@validate_app.command(name="note")
def validate_note(
    note_path: Annotated[
        Path,
        typer.Argument(help="Path to note file", exists=True),
    ],
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Apply safe auto-fixes"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
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
    """Validate a single Q&A note."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.validation import NoteValidator, ReportGenerator, Severity

    logger.info("validate_note_started", path=str(note_path))

    # Find vault root
    vault_path = config.vault_path / config.source_dir
    if not vault_path.exists():
        # Try to use the note's parent directory
        vault_path = note_path.parent.parent
        console.print(
            f"[yellow]Warning: Using detected vault root: {vault_path}[/yellow]"
        )

    validator = NoteValidator(vault_root=vault_path)

    result = validator.validate_file(note_path, collect_fixes=fix)

    if not result["success"]:
        console.print(f"[bold red]Error:[/bold red] {result.get('error')}")
        raise typer.Exit(code=1)

    # Generate report
    report = ReportGenerator.generate_console_report(
        str(note_path),
        result["issues"],
        result.get("passed", []),
        use_colors=True,
    )
    console.print(report)

    # Apply fixes if requested
    if fix and result.get("fixes"):
        safe_fixes = [f for f in result["fixes"] if f.safe]
        if safe_fixes:
            console.print("\n[cyan]Applying auto-fixes...[/cyan]")
            count, descriptions = validator.apply_fixes(note_path, safe_fixes)
            for desc in descriptions:
                console.print(f"  [green]+[/green] {desc}")
            console.print(f"\n[green]Applied {count} auto-fix(es)[/green]")

            # Re-validate after fixes
            if verbose:
                console.print("\n[cyan]Re-validating after fixes...[/cyan]")
                result = validator.validate_file(note_path, collect_fixes=False)
                report = ReportGenerator.generate_console_report(
                    str(note_path),
                    result["issues"],
                    result.get("passed", []),
                    use_colors=True,
                )
                console.print(report)

    # Exit with error code if critical issues
    critical_count = len(result["issues"].get(Severity.CRITICAL, []))
    if critical_count > 0:
        raise typer.Exit(code=1)

    logger.info("validate_note_completed", path=str(note_path))


@validate_app.command(name="dir")
def validate_dir(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory to validate", exists=True),
    ],
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Apply safe auto-fixes"),
    ] = False,
    incremental: Annotated[
        bool,
        typer.Option("--incremental", help="Only validate changed files"),
    ] = False,
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel", "-p", help="Use parallel processing for faster validation"
        ),
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (default: CPU count, max 8)",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Generate markdown report to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show per-file details"),
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
    """Validate all Q&A notes in a specific directory."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.validation import NoteValidator, ReportGenerator, Severity

    logger.info("validate_dir_started", directory=str(directory), parallel=parallel)

    # Use the directory's parent as vault root for link resolution
    vault_path = config.vault_path / config.source_dir
    if not vault_path.exists():
        vault_path = directory.parent

    mode_str = "parallel" if parallel else "sequential"
    console.print(f"\n[cyan]Validating notes in:[/cyan] {directory} ({mode_str})")

    validator = NoteValidator(
        vault_root=vault_path,
        incremental=incremental,
        cache_dir=config.get_data_path(),
    )

    if parallel:
        results, skipped_count = validator.validate_directory_parallel(
            directory,
            show_progress=True,
            collect_fixes=fix,
            max_workers=workers,
        )
    else:
        results, skipped_count = validator.validate_directory(
            directory,
            show_progress=True,
            collect_fixes=fix,
        )

    # Apply fixes if requested
    fixes_applied = 0
    if fix:
        console.print("\n[cyan]Applying auto-fixes...[/cyan]")
        for result in results:
            if result.get("fixes") and result.get("filepath"):
                safe_fixes = [f for f in result["fixes"] if f.safe]
                if safe_fixes:
                    count, _ = validator.apply_fixes(result["filepath"], safe_fixes)
                    fixes_applied += count

        if fixes_applied:
            console.print(f"[green]Applied {fixes_applied} auto-fix(es)[/green]")

    # Show summary
    console.print("\n" + validator.generate_report(results, use_colors=True))

    if skipped_count > 0:
        console.print(f"[dim]Skipped {skipped_count} unchanged file(s)[/dim]")

    # Show per-file details if verbose
    if verbose:
        files_with_issues = [r for r in results if r.get("issues")]
        if files_with_issues:
            console.print("\n[bold]Files with issues:[/bold]")
            for result in files_with_issues:
                report_text = ReportGenerator.generate_console_report(
                    result["file"],
                    result["issues"],
                    result.get("passed", []),
                    use_colors=True,
                )
                console.print(report_text)

    # Generate markdown report if requested
    if report:
        md_report = validator.generate_markdown_report(results)
        report.write_text(md_report, encoding="utf-8")
        console.print(f"\n[green]Report written to:[/green] {report}")

    # Exit with error code if critical issues
    critical_count = sum(len(r["issues"].get(Severity.CRITICAL, [])) for r in results)
    if critical_count > 0:
        console.print(
            f"\n[bold red]Found {critical_count} critical issue(s)[/bold red]"
        )
        raise typer.Exit(code=1)

    logger.info("validate_dir_completed", files=len(results), fixes=fixes_applied)


@validate_app.command(name="stats")
def validate_stats(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Show validation cache statistics."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.validation import HashTracker

    vault_path = config.vault_path / config.source_dir
    tracker = HashTracker(vault_path, cache_dir=config.get_data_path())
    stats = tracker.get_stats()

    console.print("\n[bold cyan]Validation Cache Statistics[/bold cyan]\n")
    console.print(f"  Total cached: {stats['total_cached']} files")
    console.print(f"  Passed: {stats['passed']} files")
    console.print(f"  Failed: {stats['failed']} files")
    console.print(f"\n  Cache location: {tracker.cache_file}")

    logger.info("validate_stats_shown", **stats)


@validate_app.command(name="clear-cache")
def validate_clear_cache(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.yaml", exists=True),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
    ] = "INFO",
) -> None:
    """Clear the validation cache."""
    config, logger = get_config_and_logger(config_path, log_level)

    from obsidian_anki_sync.validation import HashTracker

    vault_path = config.vault_path / config.source_dir
    tracker = HashTracker(vault_path, cache_dir=config.get_data_path())

    # Get stats before clearing
    stats = tracker.get_stats()
    cleared_count = stats["total_cached"]

    tracker.clear_cache()

    console.print(
        f"\n[green]Cleared validation cache ({cleared_count} entries)[/green]"
    )

    logger.info("validate_cache_cleared", entries_cleared=cleared_count)
