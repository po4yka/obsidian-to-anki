"""Format command implementation logic."""

import subprocess
from typing import Any

import typer

from obsidian_anki_sync.config import Config

from .shared import console


def run_format(config: Config, logger: Any, check: bool) -> None:
    """Execute the format operation.

    Args:
        config: Configuration object
        logger: Logger instance
        check: Run formatters in check mode (no modifications)

    Raises:
        typer.Exit: On format failure
    """
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

            # Display what we're running
            mode = "Checking" if check else "Formatting"
            tool = base[0]
            console.print(f"[cyan]{mode} with {tool}...[/cyan]")

            subprocess.run(cmd, check=True)

        console.print("[green] Format completed successfully[/green]")
        logger.info("format_completed", check=check)
    except subprocess.CalledProcessError as exc:
        logger.error("format_failed", returncode=exc.returncode, cmd=exc.cmd)
        console.print(
            f"[bold red]Error:[/bold red] Formatter failed: {exc.cmd}")
        raise typer.Exit(code=1)
