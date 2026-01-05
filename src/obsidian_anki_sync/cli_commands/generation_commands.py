"""Card generation CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .generate_handler import run_generate_cards
from .shared import get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register generation commands on the given Typer app."""

    @app.command(name="generate")
    def generate_cards(
        term: Annotated[
            str,
            typer.Argument(help="Term or phrase to generate cards for"),
        ],
        count: Annotated[
            int,
            typer.Option(
                "--count",
                "-c",
                help="Number of card examples to generate",
            ),
        ] = 5,
        prompt: Annotated[
            Path,
            typer.Option(
                "--prompt",
                "-p",
                help="Path to prompt template file",
            ),
        ] = Path("prompts/card_generator_prompt.txt"),
        model: Annotated[
            str | None,
            typer.Option(
                "--model",
                "-m",
                help="LLM model to use (overrides config default)",
            ),
        ] = None,
        temperature: Annotated[
            float,
            typer.Option(
                "--temperature",
                "-t",
                help="Sampling temperature for generation",
            ),
        ] = 0.2,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Preview generated cards without importing to Anki",
            ),
        ] = False,
        output: Annotated[
            Path | None,
            typer.Option(
                "--output",
                "-o",
                help="Write generated cards to file instead of importing",
            ),
        ] = None,
        copy_mode: Annotated[
            bool,
            typer.Option(
                "--copy-mode",
                help="Copy prompt to clipboard and accept manual LLM response",
            ),
        ] = False,
        log: Annotated[
            Path | None,
            typer.Option("--log", help="Optional log file for LLM responses"),
        ] = None,
        very_verbose: Annotated[
            bool,
            typer.Option(
                "--very-verbose",
                help="Log full LLM responses to log file (auto-enables --log)",
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
        """Generate multiple card examples for a term and interactively select which to keep."""
        config, logger = get_config_and_logger(
            config_path, log_level, log_file=log, very_verbose=very_verbose
        )

        run_generate_cards(
            config=config,
            logger=logger,
            term=term,
            prompt=prompt,
            count=count,
            model=model,
            temperature=temperature,
            dry_run=dry_run,
            output=output,
            copy_mode=copy_mode,
            log_file=log,
            very_verbose=very_verbose,
        )
