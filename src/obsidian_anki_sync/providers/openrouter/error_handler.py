"""Error handling and parsing for OpenRouter API responses."""

from typing import Any

import httpx

from obsidian_anki_sync.utils.logging import get_logger

from .models import MODELS_WITH_STRUCTURED_OUTPUT_ISSUES

logger = get_logger(__name__)


def parse_api_error_response(
    error: httpx.HTTPStatusError, model: str, json_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Parse API error response for detailed error information.

    Args:
        error: HTTP status error
        model: Model identifier
        json_schema: JSON schema used in request (if any)

    Returns:
        Dictionary with error details
    """
    error_details = {}
    error_type = None
    error_message = str(error)
    raw_response_text = error.response.text[:1000]

    try:
        error_json = error.response.json()
        error_details = error_json

        if "error" in error_json:
            error_msg = error_json.get("error", {})
            if isinstance(error_msg, dict):
                error_type = error_msg.get("type", "")
                error_message = error_msg.get("message", "")

                # Log specific error types
                if "schema" in error_type.lower() or "validation" in error_type.lower():
                    logger.error(
                        "openrouter_schema_validation_error",
                        status_code=error.response.status_code,
                        model=model,
                        error_type=error_type,
                        error_message=error_message,
                        had_json_schema=bool(json_schema),
                        schema_name=(
                            json_schema.get("name", "unknown") if json_schema else None
                        ),
                        raw_response=raw_response_text,
                    )
                elif "rate_limit" in error_type.lower():
                    logger.error(
                        "openrouter_rate_limit_error",
                        status_code=error.response.status_code,
                        model=model,
                        error_message=error_message,
                    )
                else:
                    logger.error(
                        "openrouter_400_error_details",
                        status_code=error.response.status_code,
                        model=model,
                        error_type=error_type,
                        error_message=error_message,
                        had_json_schema=bool(json_schema),
                        schema_name=(
                            json_schema.get("name", "unknown") if json_schema else None
                        ),
                        raw_response=raw_response_text,
                        error_json=error_json,
                    )
        else:
            logger.error(
                "openrouter_400_error_no_structure",
                status_code=error.response.status_code,
                model=model,
                raw_response=raw_response_text,
                error_json=error_json,
                had_json_schema=bool(json_schema),
            )
    except Exception as parse_error:
        error_details = {
            "raw_response": raw_response_text,
            "parse_error": str(parse_error),
        }
        logger.error(
            "openrouter_400_error_parse_failed",
            status_code=error.response.status_code,
            model=model,
            raw_response=raw_response_text,
            parse_error=str(parse_error),
            had_json_schema=bool(json_schema),
        )

    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_details": error_details,
        "raw_response_text": raw_response_text,
    }


def should_fallback_to_basic_json(
    model: str, status_code: int, json_schema: dict[str, Any] | None
) -> bool:
    """Determine if we should fallback to basic JSON mode.

    Args:
        model: Model identifier
        status_code: HTTP status code
        json_schema: JSON schema used in request (if any)

    Returns:
        True if should fallback to basic JSON mode
    """
    # If structured output failed with 400, check if model is known to have issues
    if json_schema and status_code == 400:
        return model in MODELS_WITH_STRUCTURED_OUTPUT_ISSUES
    return False
