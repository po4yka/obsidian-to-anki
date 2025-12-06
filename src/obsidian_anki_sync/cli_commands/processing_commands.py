"""Processing CLI commands: process-file and direct Anki queries."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..providers.factory import ProviderFactory
from .process_file import (
    get_processed_slugs,
    load_cards_from_file,
    process_card_with_llm,
    save_cards_to_file,
)
from .shared import console, get_config_and_logger


def register(app: typer.Typer) -> None:
    """Register processing commands on the given Typer app."""

    @app.command(name="process-file")
    def process_file(
        input: Annotated[
            Path,
            typer.Argument(help="Path to input YAML or CSV file"),
        ],
        output: Annotated[
            Path,
            typer.Option("--output", "-o", help="Output file path (required)"),
        ],
        prompt: Annotated[
            Path,
            typer.Option("--prompt", "-p", help="Path to prompt template file"),
        ],
        field: Annotated[
            str | None,
            typer.Option(
                "--field",
                help="Field name to update (single field mode)",
            ),
        ] = None,
        json_mode: Annotated[
            bool,
            typer.Option("--json", help="Expect JSON response and merge all fields"),
        ] = False,
        model: Annotated[
            str | None,
            typer.Option("--model", "-m", help="LLM model to use"),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Preview without making API calls"),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Re-process all rows, ignoring existing output",
            ),
        ] = False,
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Limit number of new rows to process"),
        ] = None,
        require_result_tag: Annotated[
            bool,
            typer.Option(
                "--require-result-tag",
                help="Extract content from <result></result> tags only",
            ),
        ] = False,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option(
                "--log-level",
                help="Log level (DEBUG, INFO, WARN, ERROR)",
            ),
        ] = "INFO",
    ) -> None:
        """Process cards from file with LLM and save results."""
        config, logger = get_config_and_logger(config_path, log_level)

        if not input.exists():
            console.print(
                f"\n[bold red]Error:[/bold red] Input file not found: {input}"
            )
            raise typer.Exit(code=1)

        if field and json_mode:
            console.print(
                "\n[bold red]Error:[/bold red] --field and --json are mutually exclusive"
            )
            raise typer.Exit(code=1)

        if not field and not json_mode:
            console.print(
                "\n[bold red]Error:[/bold red] Must specify either --field or --json"
            )
            raise typer.Exit(code=1)

        if not prompt.exists():
            console.print(
                f"\n[bold red]Error:[/bold red] Prompt file not found: {prompt}"
            )
            raise typer.Exit(code=1)

        with prompt.open("r", encoding="utf-8") as file_handle:
            prompt_template = file_handle.read()

        model_name = model or config.default_llm_model or "deepseek/deepseek-v3.2"
        if not model_name:
            console.print(
                "\n[bold red]Error:[/bold red] No model specified and no default configured"
            )
            raise typer.Exit(code=1)

        logger.info(
            "process_file_started",
            input_path=str(input),
            output_path=str(output),
            model=model_name,
            field=field,
            json_mode=json_mode,
        )

        try:
            console.print(f"\n[cyan]Loading cards from {input}...[/cyan]")
            all_cards = load_cards_from_file(input)
            console.print(f"[green]Found {len(all_cards)} cards[/green]")

            input_file_format = input.suffix.lower()
            if input_file_format == ".csv":
                output_file_format = "csv"
            else:
                output_file_format = "yaml"

            processed_slugs = set()
            processed_cards = []
            if output.exists() and not force:
                processed_slugs = get_processed_slugs(output, output_file_format)
                processed_cards = load_cards_from_file(output)
                console.print(
                    f"[cyan]Found {len(processed_cards)} already processed cards[/cyan]"
                )

            cards_to_process = []
            for card in all_cards:
                slug = card.get("slug", "")
                if force or slug not in processed_slugs:
                    cards_to_process.append(card)

            if limit:
                cards_to_process = cards_to_process[:limit]

            if not cards_to_process:
                console.print("\n[yellow]No new cards to process.[/yellow]")
                return

            console.print(
                f"\n[cyan]Processing {len(cards_to_process)} cards...[/cyan]"
            )

            if dry_run:
                console.print(
                    "[yellow]Dry run mode: No API calls will be made[/yellow]"
                )
                for card in cards_to_process[:5]:
                    console.print(f"  Would process: {card.get('slug', 'unknown')}")
                return

            provider = ProviderFactory.create_from_config(config)
            if hasattr(provider, "client"):
                llm_client = provider.client
            elif hasattr(provider, "get_client"):
                llm_client = provider.get_client()
            else:
                llm_client = provider

            processed_cards_output = []
            success_count = 0
            error_count = 0

            for i, card in enumerate(cards_to_process, 1):
                try:
                    console.print(
                        f"  [{i}/{len(cards_to_process)}] Processing {card.get('slug', 'unknown')}..."
                    )
                    updated_card = process_card_with_llm(
                        card=card.copy(),
                        prompt_template=prompt_template,
                        llm_client=llm_client,
                        model=model_name,
                        field_name=field,
                        json_mode=json_mode,
                        require_result_tag=require_result_tag,
                    )

                    processed_cards_output.append(updated_card)
                    if "_error" not in updated_card:
                        success_count += 1
                    else:
                        error_count += 1

                    if i % 10 == 0:
                        all_processed = processed_cards + processed_cards_output
                        save_cards_to_file(all_processed, output, output_file_format)
                        console.print(
                            f"  [dim]Saved progress ({i}/{len(cards_to_process)})[/dim]"
                        )

                except Exception as exc:
                    logger.error(
                        "card_processing_error",
                        error=str(exc),
                        slug=card.get("slug"),
                    )
                    error_count += 1
                    card["_error"] = str(exc)
                    processed_cards_output.append(card)

            all_processed = processed_cards + processed_cards_output
            save_cards_to_file(all_processed, output, output_file_format)

            console.print("\n[bold green]Processing complete![/bold green]")
            console.print(f"  Success: {success_count}")
            console.print(f"  Errors: {error_count}")
            console.print(f"  Output: {output}")

            logger.info(
                "process_file_completed",
                total=len(cards_to_process),
                success=success_count,
                errors=error_count,
            )

        except Exception as exc:
            logger.error("process_file_failed", error=str(exc))
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1)

    @app.command(name="query")
    def query_anki(
        action: Annotated[
            str,
            typer.Argument(
                help="AnkiConnect API action name (e.g., 'deckNames', 'findNotes')"
            ),
        ],
        params: Annotated[
            str | None, typer.Argument(help="JSON string of parameters (optional)")
        ] = None,
        config_path: Annotated[
            Path | None,
            typer.Option("--config", help="Path to config.yaml", exists=True),
        ] = None,
        log_level: Annotated[
            str,
            typer.Option("--log-level", help="Log level (DEBUG, INFO, WARN, ERROR)"),
        ] = "INFO",
    ) -> None:
        """Query AnkiConnect API directly."""
        from .anki_handler import run_query_anki

        config, logger = get_config_and_logger(config_path, log_level)
        run_query_anki(config=config, logger=logger, action=action, params=params)


