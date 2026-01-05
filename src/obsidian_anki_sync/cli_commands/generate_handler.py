"""Generate command implementation logic."""

import tempfile
from pathlib import Path
from typing import Any

import typer
import yaml

from obsidian_anki_sync.anki.client import AnkiClient
from obsidian_anki_sync.anki.importer import import_cards_from_yaml
from obsidian_anki_sync.config import Config
from obsidian_anki_sync.prompts.template_parser import parse_template_file
from obsidian_anki_sync.utils.card_selector import CardCandidate
from obsidian_anki_sync.utils.clipboard import copy_to_clipboard

from .shared import console


def run_generate_cards(
    config: Config,
    logger: Any,
    term: str,
    prompt: Path,
    count: int,
    model: str | None,
    temperature: float,
    dry_run: bool,
    output: Path | None,
    copy_mode: bool,
    log_file: Path | None,
    very_verbose: bool,
) -> None:
    """Execute the generate-cards operation.

    Args:
        config: Configuration object
        logger: Logger instance
        term: Term or phrase to generate cards for
        prompt: Path to prompt template file
        count: Number of card examples to generate
        model: LLM model to use
        temperature: LLM temperature
        dry_run: Display cards without importing
        output: Export cards to file instead of importing
        copy_mode: Copy prompt to clipboard for manual LLM interaction
        log_file: Generate log file with detailed debug information
        very_verbose: Log full LLM responses to log file

    Raises:
        typer.Exit: On generate-cards failure
    """
    if not prompt.exists():
        console.print(f"\n[bold red]Error:[/bold red] Prompt file not found: {prompt}")
        raise typer.Exit(code=1)

    # Parse prompt template
    try:
        template = parse_template_file(prompt)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] Failed to parse template: {e}")
        raise typer.Exit(code=1)

    # Determine model
    model_name = model or config.default_llm_model or "deepseek/deepseek-v3.2"
    if not model_name:
        console.print(
            "\n[bold red]Error:[/bold red] No model specified and no default configured"
        )
        raise typer.Exit(code=1)

    logger.info(
        "generate_cards_started",
        term=term,
        count=count,
        model=model_name,
        template_path=str(prompt),
    )

    # Build prompt with substitutions
    prompt_text = template.substitute(term=term, count=count)

    # Copy mode: copy to clipboard and wait for manual input
    if copy_mode:
        console.print("\n[cyan]Copy mode: Prompt copied to clipboard[/cyan]")
        if copy_to_clipboard(prompt_text):
            console.print("[green]Prompt copied![/green]")
            console.print("\n[yellow]Instructions:[/yellow]")
            console.print(
                "1. Paste the prompt into your LLM interface (OpenRouter, Ollama, etc.)"
            )
            console.print("2. Copy the complete JSON response")
            console.print("3. Paste it here and press Enter")
            console.print("4. Type 'END' on a new line to finish\n")

            # Wait for user input
            lines = []
            console.print(
                "[cyan]Paste JSON response (type 'END' on new line to finish):[/cyan]"
            )
            while True:
                try:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[yellow]Input cancelled[/yellow]")
                    return

            response_text = "\n".join(lines)
        else:
            console.print(
                "[yellow]Clipboard not available, displaying prompt:[/yellow]"
            )
            console.print(f"\n[dim]{prompt_text}[/dim]\n")
            response_text = typer.prompt("[cyan]Paste LLM response[/cyan]", default="")
    else:
        # Normal mode: call LLM
        from obsidian_anki_sync.providers.factory import ProviderFactory

        provider = ProviderFactory.create_from_config(config)
        if hasattr(provider, "client"):
            llm_client = provider.client
        else:
            llm_client = provider

        console.print(
            f"\n[cyan]Generating {count} card candidates for '{term}'...[/cyan]"
        )

        try:
            if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
                # OpenAI-style client
                response = llm_client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                response_text = response.choices[0].message.content or ""
            else:
                # Generic interface
                response_text = llm_client.generate(
                    prompt_text, model=model_name, temperature=temperature
                )

        except Exception as e:
            logger.error("card_generation_failed", error=str(e))
            console.print(f"\n[bold red]Error generating cards:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Parse JSON response
    import json

    try:
        cards_data = json.loads(response_text)
        if not isinstance(cards_data, list):
            cards_data = [cards_data]
    except json.JSONDecodeError as e:
        console.print(f"\n[bold red]Error:[/bold red] Invalid JSON response: {e}")
        console.print(f"[yellow]Response was:[/yellow]\n{response_text[:200]}...")
        raise typer.Exit(code=1)

    # Convert to CardCandidate objects
    candidates = []
    for i, card_data in enumerate(cards_data):
        if not isinstance(card_data, dict):
            continue

        # Map fields using template field_map if available
        fields = {}
        if template.field_map:
            for template_key, anki_field in template.field_map.items():
                if template_key in card_data:
                    fields[anki_field] = card_data[template_key]
        else:
            fields = card_data

        candidate = CardCandidate(index=i, fields=fields)
        candidates.append(candidate)

    if not candidates:
        console.print("\n[yellow]No valid cards generated.[/yellow]")
        return

    # Check for duplicates if we have a deck
    if template.deck and not dry_run:
        try:
            client = AnkiClient(config.anki_connect_url)
            note_ids = client.find_notes(f'deck:"{template.deck}"')
            if note_ids:
                notes_info = client.notes_info(note_ids)
                # Simple duplicate check based on field content
                for candidate in candidates:
                    for note_info in notes_info:
                        note_fields = note_info.get("fields", {})
                        # Check if any field matches
                        for field_name, field_value in candidate.fields.items():
                            if field_name in note_fields:
                                if field_value[:50] in note_fields[field_name][:50]:
                                    candidate.is_duplicate = True
                                    candidate.duplicate_reason = (
                                        f"Similar to existing card in {template.deck}"
                                    )
                                    break
        except Exception as e:
            logger.warning("duplicate_check_failed", error=str(e))

    # Interactive selection
    if not dry_run and not output:
        from obsidian_anki_sync.utils.card_selector import select_cards_interactive

        selected_indices = select_cards_interactive(
            candidates, title=f"Select cards to add for '{term}'"
        )

        if not selected_indices:
            console.print("\n[yellow]No cards selected. Exiting.[/yellow]")
            return

        selected_candidates = [candidates[i] for i in selected_indices]

        # Quality check if configured
        if template.quality_check:
            console.print("\n[cyan]Running quality checks...[/cyan]")
            from obsidian_anki_sync.providers.factory import ProviderFactory
            from obsidian_anki_sync.utils.quality_check import run_quality_check

            quality_provider = ProviderFactory.create_from_config(config)
            if hasattr(quality_provider, "client"):
                quality_client = quality_provider.client
            else:
                quality_client = quality_provider

            for candidate in selected_candidates:
                quality_result = run_quality_check(
                    card_fields=candidate.fields,
                    quality_config=template.quality_check,
                    llm_client=quality_client,
                )
                candidate.quality_score = quality_result["score"]
                candidate.quality_reason = quality_result["reason"]

                if not quality_result["is_valid"] or quality_result["score"] < 0.7:
                    console.print(
                        f"  [yellow]Card {candidate.index + 1} flagged:[/yellow] {quality_result['reason']}"
                    )

        # Import selected cards
        if template.deck:
            # Convert to YAML format for import
            cards_for_import = []
            for candidate in selected_candidates:
                card_data = {
                    "noteType": template.note_type or "APF::Simple",
                    **candidate.fields,
                }
                cards_for_import.append(card_data)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(cards_for_import, f)
                temp_path = Path(f.name)

            try:
                client = AnkiClient(config.anki_connect_url)
                result = import_cards_from_yaml(
                    client=client,
                    input_path=temp_path,
                    deck_name=template.deck,
                    note_type=template.note_type,
                )
                console.print(
                    f"\n[bold green]Successfully added {result['created']} card(s)![/bold green]"
                )
            finally:
                temp_path.unlink()

    elif output:
        # Export to file
        cards_for_export = []
        for candidate in candidates:
            card_data = {
                "noteType": template.note_type or "APF::Simple",
                **candidate.fields,
            }
            cards_for_export.append(card_data)

        with output.open("w", encoding="utf-8") as f:
            yaml.dump(cards_for_export, f, default_flow_style=False, allow_unicode=True)

        console.print(
            f"\n[bold green]Exported {len(cards_for_export)} cards to {output}[/bold green]"
        )
    else:
        # Dry run: just display
        console.print(f"\n[cyan]Generated {len(candidates)} card candidates:[/cyan]\n")
        for candidate in candidates:
            console.print(f"[bold]Card {candidate.index + 1}:[/bold]")
            for key, value in candidate.fields.items():
                console.print(
                    f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}"
                )
            console.print()

    logger.info("generate_cards_completed", term=term, count=len(candidates))
