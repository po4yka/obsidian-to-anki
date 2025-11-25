"""File-based processing with LLM and resume support."""

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import DeckImportError
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_cards_from_file(input_path: Path) -> list[dict[str, Any]]:
    """
    Load cards from YAML or CSV file.

    Args:
        input_path: Path to input file

    Returns:
        List of card dictionaries

    Raises:
        DeckImportError: If file cannot be loaded
    """
    # Security: Check file size to prevent DoS attacks
    try:
        file_size = input_path.stat().st_size
        max_file_size = 50 * 1024 * 1024  # 50MB limit for import files
        if file_size > max_file_size:
            raise DeckImportError(
                f"File too large: {input_path} ({file_size} bytes). "
                f"Maximum allowed size is {max_file_size} bytes."
            )
    except OSError as e:
        raise DeckImportError(f"Cannot check file size: {e}")

    file_format = input_path.suffix.lower()

    if file_format in (".yaml", ".yml"):
        with input_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, list):
                raise DeckImportError("YAML file must contain a list of cards")
            return data
    elif file_format == ".csv":
        cards = []
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cards.append(dict(row))
        return cards
    else:
        raise DeckImportError(f"Unsupported file format: {file_format}")


def save_cards_to_file(
    cards: list[dict[str, Any]], output_path: Path, file_format: str
) -> None:
    """
    Save cards to YAML or CSV file.

    Args:
        cards: List of card dictionaries
        output_path: Path to output file
        file_format: Output format ('yaml' or 'csv')
    """
    if file_format.lower() == "csv":
        if not cards:
            # Create empty CSV
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["noteId", "slug", "noteType", "tags"])
            return

        # Get all field names
        all_fields = set()
        for card in cards:
            all_fields.update(card.keys())

        field_names = sorted(all_fields)

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=field_names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cards)
    else:
        # YAML
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                cards, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )


def get_processed_slugs(
    output_path: Path, file_format: str
) -> set[str]:  # noqa: ARG001
    """
    Get set of already processed card slugs from output file.

    Args:
        output_path: Path to output file
        file_format: File format ('yaml' or 'csv') - kept for API compatibility

    Returns:
        Set of processed slugs
    """
    if not output_path.exists():
        return set()

    try:
        cards = load_cards_from_file(output_path)
        return {card.get("slug", "") for card in cards if card.get("slug")}
    except (DeckImportError, yaml.YAMLError, csv.Error) as e:
        logger.warning("failed_to_load_existing_output", error=str(e))
        return set()


def process_card_with_llm(
    card: dict[str, Any],
    prompt_template: str,
    llm_client: Any,
    model: str,
    field_name: str | None = None,
    json_mode: bool = False,
    require_result_tag: bool = False,
) -> dict[str, Any]:
    """
    Process a single card with LLM.

    Args:
        card: Card dictionary
        prompt_template: Prompt template with {field_name} placeholders
        llm_client: LLM client instance
        model: Model name to use
        field_name: Field to update (if single field mode)
        json_mode: Whether to expect JSON response
        require_result_tag: Whether to extract content from <result> tags

    Returns:
        Updated card dictionary
    """
    # Substitute variables in prompt
    prompt = prompt_template
    for key, value in card.items():
        if isinstance(value, (str, int, float)):
            prompt = prompt.replace(f"{{{key}}}", str(value))

    # Call LLM
    try:
        if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            # OpenAI-style client
            from ..utils.llm_logging import log_llm_request, log_llm_success

            start_time = log_llm_request(
                model=model,
                operation="process_file",
                prompt_length=len(prompt),
            )

            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            result_text = response.choices[0].message.content or ""

            # Log success with cost tracking
            usage = response.usage
            log_llm_success(
                model=model,
                operation="process_file",
                start_time=start_time,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                response_length=len(result_text),
            )
        else:
            # Generic client interface
            result_text = llm_client.generate(prompt, model=model)

        # Extract result if required
        if require_result_tag:
            from ..utils.result_extractor import extract_result_tag

            result_text = extract_result_tag(result_text, require_tag=True)

        # Update card
        if json_mode:
            # Parse JSON and merge fields
            try:
                result_data = json.loads(result_text)
                if isinstance(result_data, dict):
                    card.update(result_data)
            except json.JSONDecodeError:
                logger.warning("invalid_json_response", slug=card.get("slug"))
        elif field_name:
            # Update single field
            card[field_name] = result_text
        else:
            # Default: update first field or add to a generic field
            if not card:
                card["processed"] = result_text
            else:
                first_key = next(iter(card.keys()))
                card[first_key] = result_text

    except (ValueError, KeyError, AttributeError, RuntimeError, TypeError) as e:
        logger.error("llm_processing_failed",
                     error=str(e), slug=card.get("slug"))
        card["_error"] = str(e)

    return card
