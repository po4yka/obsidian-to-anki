"""JSON cleaning and repair utilities for OpenRouter responses."""

import json

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


def clean_json_response(text: str, model: str) -> str:
    """Clean JSON response from DeepSeek and similar models.

    DeepSeek often wraps JSON in markdown fences, adds reasoning text,
    or appends special tokens. This method extracts the clean JSON.

    Args:
        text: Raw response text
        model: Model identifier for logging

    Returns:
        Cleaned JSON text
    """
    # Track warnings per call to avoid duplicate warnings for the same text
    # Use a simple identifier based on text start and length
    text_id = f"{hash(text[:100])}:{len(text)}"
    warned_text_ids = set()  # Local set per call

    cleaned = text.strip()

    # Remove markdown code fences if present (must be done first)
    if cleaned.startswith("```json"):
        end_fence = cleaned.rfind("```")
        if end_fence > 6:  # Found closing fence after opening
            cleaned = cleaned[7:end_fence].strip()
            logger.debug(
                "removed_json_markdown_fence",
                model=model,
                original_length=len(text),
                cleaned_length=len(cleaned),
            )
    elif cleaned.startswith("```"):
        end_fence = cleaned.rfind("```")
        if end_fence > 3:
            cleaned = cleaned[3:end_fence].strip()
            logger.debug(
                "removed_generic_markdown_fence",
                model=model,
                original_length=len(text),
                cleaned_length=len(cleaned),
            )

    # Remove DeepSeek special tokens like <｜begin▁of▁sentence｜>
    if "<｜" in cleaned:
        token_pos = cleaned.find("<｜")
        if token_pos > 0:
            cleaned = cleaned[:token_pos].strip()
            logger.debug(
                "removed_special_tokens",
                model=model,
                cleaned_length=len(cleaned),
            )

    # Extract JSON if response contains reasoning/explanatory text before JSON
    if not cleaned.startswith("{"):
        first_brace = cleaned.find("{")
        if first_brace != -1:
            cleaned = cleaned[first_brace:]
            logger.debug(
                "extracted_json_from_text",
                model=model,
                original_length=len(text),
                cleaned_length=len(cleaned),
            )

    # Find the actual end of the JSON object by counting braces
    # This handles cases where there's extra text after the JSON closes
    if cleaned.startswith("{"):
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        last_valid_pos = 0
        truncation_detected = False

        for i, char in enumerate(cleaned):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                last_valid_pos = i + 1
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                    last_valid_pos = i + 1
                elif char == "}":
                    brace_count -= 1
                    last_valid_pos = i + 1
                    if brace_count == 0 and bracket_count == 0:
                        # Found the end of the JSON object (all structures closed)
                        if i + 1 < len(cleaned):
                            cleaned = cleaned[: i + 1]
                            logger.debug(
                                "trimmed_extra_data_after_json",
                                model=model,
                                original_length=len(text),
                                cleaned_length=len(cleaned),
                            )
                        break
                elif char == "[":
                    bracket_count += 1
                    last_valid_pos = i + 1
                elif char == "]":
                    bracket_count -= 1
                    last_valid_pos = i + 1
                    # If we close a bracket but still have open braces, check if we're near the end
                    # This could indicate truncation, but only warn if we're close to the end of text
                    # (Arrays can legitimately close before their containing objects in valid JSON)
                    if bracket_count == 0 and brace_count > 0:
                        # Only warn if we're in the last 10% of the text (likely truncation)
                        # or if there's very little text remaining
                        remaining_chars = len(cleaned) - i - 1
                        is_near_end = remaining_chars < max(100, len(cleaned) * 0.1)

                        if is_near_end:
                            # Only log warning once per unique text (avoid duplicates)
                            warning_key = f"premature_array_close:{text_id}"
                            if warning_key not in warned_text_ids:
                                # Array closed but object(s) still open near end of text - likely truncation
                                logger.warning(
                                    "detected_premature_array_close",
                                    model=model,
                                    brace_count=brace_count,
                                    position=i,
                                    remaining_chars=remaining_chars,
                                    total_length=len(cleaned),
                                    note="Array closed before containing object near end of text - likely truncation",
                                )
                                warned_text_ids.add(warning_key)
                            truncation_detected = True
                        # Don't break here - let repair handle it if needed
                else:
                    last_valid_pos = i + 1

        # If we ended while still in a string or with unclosed braces/brackets, try to repair
        if in_string or brace_count > 0 or bracket_count > 0 or truncation_detected:
            logger.warning(
                "detected_truncated_json",
                model=model,
                in_string=in_string,
                brace_count=brace_count,
                bracket_count=bracket_count,
                truncation_detected=truncation_detected,
                original_length=len(text),
                cleaned_length=len(cleaned),
            )
            # Try to repair the truncated JSON
            # Use text up to last_valid_pos, but if that's 0, use the whole cleaned text
            text_to_repair = cleaned[:last_valid_pos] if last_valid_pos > 0 else cleaned
            cleaned = repair_truncated_json(text_to_repair)

    return cleaned


def repair_truncated_array(text: str) -> str:
    """Attempt to repair JSON with prematurely closed arrays.

    This handles cases where an array closes before its containing object,
    which typically indicates response truncation.

    Args:
        text: JSON text with potential array truncation

    Returns:
        Repaired JSON text
    """
    if not text.strip() or not text.startswith("{"):
        return text

    # If text ends with ]" and we have unclosed objects, try to repair
    if text.rstrip().endswith(']"'):
        # Count braces and brackets to see if we need to add closing braces
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1

        # If we have unclosed braces but closed brackets, add closing braces
        if brace_count > 0 and bracket_count == 0:
            repaired = text.rstrip()
            # Remove the closing ]" and add proper closing
            if repaired.endswith(']"'):
                repaired = repaired[:-2].rstrip()
                # Add closing braces for any open objects
                repaired += "}" * brace_count
                repaired += ']"'
                logger.debug(
                    "repaired_array_truncation",
                    original_length=len(text),
                    repaired_length=len(repaired),
                )
                return repaired

    return text


def repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open structures.

    Args:
        text: Potentially truncated JSON text

    Returns:
        Repaired JSON text (may still be invalid if too corrupted)
    """
    if not text.strip():
        return "{}"

    # First try array-specific repair
    repaired = repair_truncated_array(text)
    if repaired != text:
        return repaired

    repaired = text.rstrip()

    # Track state
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0
    last_key_pos = -1  # Track where we last saw a key (before ':')

    for i, char in enumerate(repaired):
        if escape_next:
            escape_next = False
            last_valid_pos = i + 1
            continue

        if char == "\\":
            escape_next = True
            last_valid_pos = i + 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            if not in_string:
                # String closed, this is a valid position
                last_valid_pos = i + 1
            else:
                # String opened
                last_valid_pos = i + 1
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
                last_valid_pos = i + 1
            elif char == "}":
                brace_count -= 1
                last_valid_pos = i + 1
            elif char == "[":
                bracket_count += 1
                last_valid_pos = i + 1
            elif char == "]":
                bracket_count -= 1
                last_valid_pos = i + 1
            elif char == ":":
                # Found a key-value separator, mark this position
                last_key_pos = i
                last_valid_pos = i + 1
            elif char == ",":
                # Found a separator, this is a valid position
                last_valid_pos = i + 1
            elif char not in (" ", "\n", "\t", "\r"):
                # Valid JSON character
                last_valid_pos = i + 1

    # Truncate to last valid position
    repaired = repaired[:last_valid_pos]

    # If we're in a string, we need to close it
    if in_string:
        # Just close the string
        repaired += '"'

    # Check for incomplete values after colons (common truncation pattern)
    # Look for patterns like "key": "" or "key": "incomplete
    if last_key_pos >= 0 and last_key_pos < len(repaired):
        # Find the colon position
        colon_pos = repaired.find(":", last_key_pos)
        if colon_pos != -1:
            # Get everything after the colon
            after_colon = repaired[colon_pos + 1 :].lstrip()

            # Check if we have an incomplete string value
            if after_colon.startswith('"'):
                # Count quotes in the value part
                quote_count = after_colon.count('"')
                # If odd number of quotes, string is not closed
                if quote_count % 2 == 1:
                    # String is not closed, close it
                    if not repaired.endswith('"'):
                        repaired += '"'
                # If we have "" but nothing after, it's complete (empty string)
                elif after_colon == '""':
                    pass  # Complete empty string
                # If we have "something but no closing quote
                elif quote_count == 1:
                    # Only opening quote, close it
                    if not repaired.endswith('"'):
                        repaired += '"'
            elif not after_colon:
                # No value after colon, add empty string
                repaired += ' ""'
            elif after_colon and not (
                after_colon.startswith(('"', "[", "{"))
                or (after_colon and after_colon[0].isdigit())
                or after_colon in ("true", "false", "null")
            ):
                # Invalid value start, add empty string
                repaired += ' ""'

    # Before closing structures, handle the case where we have patterns like "key": ""]}}
    # This means an object inside an array is incomplete - we need to close the object first
    # Check if the text ends with premature closing brackets/braces
    trimmed_end = repaired.rstrip()
    if trimmed_end.endswith("]") and brace_count > 0:
        # We have a ] but still open braces - this means object wasn't closed before array
        # Close the object(s) first
        while brace_count > 0:
            repaired += "}"
            brace_count -= 1

    # Handle incomplete array/object values
    # If we're in the middle of a value after a colon, we need to complete it
    if last_key_pos >= 0:
        after_colon = repaired[last_key_pos + 1 :].lstrip()
        # If after colon is empty or incomplete, add a placeholder
        if not after_colon or (
            after_colon.startswith('"') and after_colon.count('"') == 1
        ):
            # Incomplete string value - close it if needed
            if after_colon.startswith('"') and not repaired.endswith('"'):
                repaired += '"'
            elif not after_colon:
                # No value at all - add empty string
                repaired += ' ""'

    # Close any open braces (objects) first - objects must be closed before arrays
    # This handles nested structures correctly
    while brace_count > 0:
        repaired += "}"
        brace_count -= 1

    # Then close any open brackets (arrays)
    while bracket_count > 0:
        repaired += "]"
        bracket_count -= 1

    # Validate the repaired JSON
    try:
        json.loads(repaired)
        logger.debug(
            "json_repair_successful",
            original_length=len(text),
            repaired_length=len(repaired),
        )
        return repaired
    except json.JSONDecodeError as e:
        # Repair failed - try a more aggressive approach
        logger.warning(
            "json_repair_failed_attempting_aggressive",
            error=str(e),
            repaired_preview=repaired[:200],
        )
        # Try to extract just the root object if possible
        if repaired.startswith("{"):
            # Find the first complete top-level object
            first_brace = repaired.find("{")
            brace_depth = 0
            end_pos = -1
            for i in range(first_brace, len(repaired)):
                if repaired[i] == "{":
                    brace_depth += 1
                elif repaired[i] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                partial_json = repaired[:end_pos]
                try:
                    json.loads(partial_json)
                    logger.debug(
                        "json_repair_partial_success",
                        extracted_length=len(partial_json),
                    )
                    return partial_json
                except json.JSONDecodeError:
                    pass

        # Last resort: return minimal valid JSON
        logger.warning("json_repair_failed_using_fallback", original_preview=text[:100])
        return "{}"
