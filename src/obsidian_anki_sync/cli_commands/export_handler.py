"""Export command implementation logic."""

import asyncio
import inspect
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator
from obsidian_anki_sync.anki.exporter import export_cards_to_apkg
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.obsidian.parser import discover_notes, parse_note
from obsidian_anki_sync.sync.state_db import StateDB
from obsidian_anki_sync.utils.preflight import run_preflight_checks

from .shared import console

if TYPE_CHECKING:
    from obsidian_anki_sync.models import Card


def run_export(
    config: Config,
    logger: Any,
    output: Path | None,
    deck_name: str | None,
    deck_description: str,
    use_langgraph: bool | None,
    sample_size: int | None,
) -> None:
    """Execute the export operation.

    Args:
        config: Configuration object
        logger: Logger instance
        output: Output .apkg file path
        deck_name: Name for the exported deck
        deck_description: Description for the deck
        use_langgraph: Override LangGraph setting
        sample_size: Export only N random notes

    Raises:
        typer.Exit: On export failure
    """
    # Override LangGraph setting if CLI flag is provided
    if use_langgraph is not None:
        config.use_langgraph = use_langgraph
        # Enable PydanticAI when using LangGraph
        config.use_pydantic_ai = use_langgraph
        logger.info("langgraph_system_override", use_langgraph=use_langgraph)

    # Determine output path
    output_path = output or config.export_output_path or Path("output.apkg")

    # Determine deck name
    final_deck_name = deck_name or config.export_deck_name or config.anki_deck_name

    # Determine deck description
    final_description = deck_description or config.export_deck_description or ""

    logger.info(
        "export_started",
        output_path=str(output_path),
        deck_name=final_deck_name,
        sample_size=sample_size,
    )

    # Run pre-flight checks (skip Anki since we're exporting to file)
    console.print("\n[bold cyan]Running pre-flight checks...[/bold cyan]\n")

    _passed, results = run_preflight_checks(
        config, check_anki=False, check_llm=True)

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
        # Generate cards by processing notes
        console.print("\n[cyan]Generating cards from Obsidian notes...[/cyan]")

        with StateDB(config.db_path) as _:
            # Use a dummy Anki client (won't connect)

            # We don't need AnkiConnect for export, but SyncEngine expects it
            # We'll use the sync engine's card generation logic
            note_paths = discover_notes(config.vault_path, config.source_dir)

            if sample_size:
                note_paths = random.sample(
                    note_paths, min(sample_size, len(note_paths))
                )

            console.print(
                f"[cyan]Processing {len(note_paths)} notes...[/cyan]")

            # Generate cards using the agent system
            cards: list[Card] = []

            # Always use agent system for card generation
            if not config.use_langgraph and not config.use_pydantic_ai:
                logger.info(
                    "enabling_agent_system",
                    reason="Agent system is required",
                )
                config.use_langgraph = True
                config.use_pydantic_ai = True

            console.print(
                "[cyan]Using LangGraph agent system for generation...[/cyan]"
            )
            orchestrator = LangGraphOrchestrator(config)

            for note_path_tuple in note_paths:
                try:
                    note_path, note_content = note_path_tuple
                    metadata, qa_pairs = parse_note(note_path)
                    # Handle both sync and async orchestrators
                    if inspect.iscoroutinefunction(orchestrator.process_note):
                        result = asyncio.run(
                            orchestrator.process_note(
                                note_content, metadata, qa_pairs, note_path
                            )
                        )
                    else:
                        result = orchestrator.process_note(
                            note_content, metadata, qa_pairs, note_path
                        )

                    if result.success and result.generation:
                        generated_cards = result.generation.cards
                        # Convert GeneratedCard to Card using orchestrator's method
                        converted_cards = orchestrator.convert_to_cards(
                            generated_cards, metadata, qa_pairs, note_path
                        )
                        cards.extend(converted_cards)
                        console.print(
                            f"  [green][/green] {metadata.title} "
                            f"({len(generated_cards)} cards)"
                        )
                    else:
                        error_msg = "Pipeline failed"
                        if result.post_validation:
                            error_msg = (
                                result.post_validation.error_details or error_msg
                            )
                        console.print(
                            f"  [red][/red] {metadata.title}: {error_msg}"
                        )
                except Exception as e:
                    console.print(f"  [red][/red] {note_path.name}: {e}")

            if not cards:
                console.print(
                    "\n[yellow]No cards generated. Exiting.[/yellow]")
                return

            # Export to .apkg
            console.print(
                f"\n[cyan]Exporting {len(cards)} cards to {output_path}...[/cyan]"
            )

            export_cards_to_apkg(
                cards=cards,
                output_path=output_path,
                deck_name=final_deck_name,
                deck_description=final_description,
            )

            console.print(
                f"\n[bold green] Successfully exported {len(cards)} cards "
                f"to {output_path}[/bold green]"
            )

            console.print(
                "\n[cyan]Import this file into Anki to add the cards.[/cyan]")

            logger.info(
                "export_completed",
                output_path=str(output_path),
                card_count=len(cards),
            )

    except Exception as e:
        logger.error("export_failed", error=str(e))
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
