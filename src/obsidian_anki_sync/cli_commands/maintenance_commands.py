"""Maintenance CLI commands such as code formatting."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .shared import console, get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register maintenance commands on the given Typer app."""

    @app.command(name="format")
    def run_format(
        check: Annotated[
            bool,
            typer.Option(
                "--check", help="Run formatters in check mode (no modifications)"
            ),
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
        """Run code formatters (ruff + isort)."""
        _config, logger = get_config_and_logger(config_path, log_level)
        logger.info("format_started", check=check)

        paths = ["src", "tests"]
        commands = [
            (["ruff", "format"], [] if not check else ["--check"], paths),
            (["isort"], ["--check-only"] if check else [], paths),
        ]

        try:
            for base, extra, target_paths in commands:
                cmd = base + extra + target_paths
                logger.debug("format_command", command=cmd)

                mode = "Checking" if check else "Formatting"
                tool = base[0]
                console.print(f"[cyan]{mode} with {tool}...[/cyan]")

                subprocess.run(cmd, check=True)

            console.print("[green] Format completed successfully[/green]")
            logger.info("format_completed", check=check)
        except subprocess.CalledProcessError as exc:
            logger.error("format_failed", returncode=exc.returncode, cmd=exc.cmd)
            console.print(f"[bold red]Error:[/bold red] Formatter failed: {exc.cmd}")
            raise typer.Exit(code=1)


