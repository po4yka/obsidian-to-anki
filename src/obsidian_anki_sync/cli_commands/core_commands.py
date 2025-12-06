"""Core CLI commands: sync, test-run, lint, validate, init, check."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..utils.preflight import run_preflight_checks
from .init_handler import run_init
from .shared import console, get_config_and_logger
from .sync_handler import run_sync
from .test_run_handler import run_test_run
from .validate_commands import run_lint_note, run_validate


def register(app: typer.Typer) -> None:
    """Register core commands on the given Typer app."""

    @app.command()
    def sync(
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Preview changes without applying",
            ),
        ] = False,
        incremental: Annotated[
            bool,
            typer.Option(
                "--incremental",
                help="Only process new notes not yet synced (skip existing notes)",
            ),
        ] = False,
        no_index: Annotated[
            bool,
            typer.Option(
                "--no-index",
                help="Skip indexing phase (not recommended)",
            ),
        ] = False,
        sample: Annotated[
            int | None,
            typer.Option(
                "--sample",
                help="Sync only N random notes (for testing)",
                min=1,
            ),
        ] = None,
        resume: Annotated[
            str | None,
            typer.Option(
                "--resume",
                help="Resume a previous interrupted sync by session ID",
            ),
        ] = None,
        no_resume: Annotated[
            bool,
            typer.Option(
                "--no-resume",
                help="Disable automatic resume of incomplete syncs",
            ),
        ] = False,
        use_queue: Annotated[
            bool,
            typer.Option(
                "--use-queue",
                help="Use Redis queue for parallel processing",
            ),
        ] = False,
        redis_url: Annotated[
            str,
            typer.Option(
                "--redis-url",
                help="Redis URL for task queue",
            ),
        ] = "redis://localhost:6379",
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Show all log messages on terminal (for debugging)",
            ),
        ] = False,
    ) -> None:
        """Synchronize Obsidian notes to Anki cards."""
        start_time = time.time()
        config, logger = get_config_and_logger(config_path, log_level, verbose=verbose)

        logger.info(
            "cli_command_started",
            command="sync",
            dry_run=dry_run,
            incremental=incremental,
            no_index=no_index,
            sample_size=sample,
            resume=resume,
            no_resume=no_resume,
            config_path=str(config_path) if config_path else None,
            log_level=log_level,
            verbose=verbose,
            vault_path=str(config.vault_path) if config.vault_path else None,
        )

        try:
            if use_queue:
                config.enable_queue = True
                config.redis_url = redis_url

            run_sync(
                config=config,
                logger=logger,
                dry_run=dry_run,
                incremental=incremental,
                no_index=no_index,
                sample_size=sample,
                resume=resume,
                no_resume=no_resume,
            )

            duration = time.time() - start_time
            logger.info(
                "cli_command_completed",
                command="sync",
                duration=round(duration, 2),
                success=True,
            )
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                "cli_command_failed",
                command="sync",
                duration=round(duration, 2),
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )
            raise

    @app.command(name="test-run")
    def test_run(
        count: Annotated[
            int,
            typer.Option(
                "--count",
                min=1,
                help="Number of random notes to process",
            ),
        ] = 10,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run/--no-dry-run",
                help="Preview changes without applying (default: --dry-run)",
            ),
        ] = True,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
        index: Annotated[
            bool,
            typer.Option(
                "--index/--no-index",
                help="Build the full vault index before sampling (default: --no-index)",
            ),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Show all log messages on terminal (for debugging)",
            ),
        ] = False,
    ) -> None:
        """Run a sample by processing N random notes."""
        config, logger = get_config_and_logger(config_path, log_level, verbose=verbose)

        run_test_run(
            config=config,
            logger=logger,
            count=count,
            dry_run=dry_run,
            index=index,
        )

    @app.command(name="lint-note")
    def lint_note(
        note_path: Annotated[
            Path, typer.Argument(help="Path to note file (markdown)", exists=True)
        ],
        enforce_bilingual: Annotated[
            bool,
            typer.Option(
                "--enforce-bilingual/--allow-partial",
                help="Require complete Q/A pairs for every language tag.",
            ),
        ] = True,
        check_code_fences: Annotated[
            bool,
            typer.Option(
                "--check-code-fences/--skip-code-fences",
                help="Validate that ``` fences are balanced.",
            ),
        ] = True,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
    ) -> None:
        """Lint a note for bilingual completeness and formatting issues."""
        config, logger = get_config_and_logger(config_path, log_level)

        run_lint_note(
            config=config,
            logger=logger,
            note_path=note_path,
            enforce_bilingual=enforce_bilingual,
            check_code_fences=check_code_fences,
        )

    @app.command()
    def validate(
        note_path: Annotated[Path, typer.Argument(help="Path to note file", exists=True)],
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Validate note structure and APF compliance."""
        config, logger = get_config_and_logger(config_path, log_level)

        run_validate(
            config=config,
            logger=logger,
            note_path=note_path,
        )

    @app.command()
    def init(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Initialize configuration and database."""
        config, logger = get_config_and_logger(config_path, log_level)

        run_init(config=config, logger=logger)

    @app.command(name="check")
    def check_setup(
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
        skip_anki: Annotated[
            bool,
            typer.Option("--skip-anki", help="Skip Anki connectivity check"),
        ] = False,
        skip_llm: Annotated[
            bool,
            typer.Option("--skip-llm", help="Skip LLM provider connectivity check"),
        ] = False,
    ) -> None:
        """Run pre-flight checks to validate your setup."""
        config, logger = get_config_and_logger(config_path, log_level)

        logger.info("check_setup_started")

        console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

        _passed, results = run_preflight_checks(
            config, check_anki=not skip_anki, check_llm=not skip_llm
        )

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

        errors = [r for r in results if not r.passed and r.severity == "error"]
        warnings = [r for r in results if not r.passed and r.severity == "warning"]
        passed_checks = [r for r in results if r.passed]

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
            console.print(
                "[yellow]Fix the errors above before running sync.[/yellow]\n"
            )
            logger.error(
                "check_setup_failed", errors=len(errors), warnings=len(warnings)
            )
            raise typer.Exit(code=1)
        if warnings:
            console.print(
                "[bold yellow]WARNING: Setup validation passed with warnings.[/bold yellow]"
            )
            console.print(
                "[dim]You may proceed, but some features may not work as expected.[/dim]\n"
            )
            logger.warning(
                "check_setup_completed_with_warnings",
                warnings=len(warnings),
                errors=len(errors),
            )
        else:
            console.print("[green]Setup validation passed![/green]\n")
            logger.info("check_setup_completed", warnings=0, errors=0)


