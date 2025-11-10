"""Schema Validation Tool for Anki model compatibility.

This is a non-LLM tool that validates ProposedCard against Anki model schemas
using pure Python logic for reliability and determinism.
"""

import re
from typing import Any, Optional

from obsidian_anki_sync.agents.langchain.models import (
    CardType,
    ProposedCard,
    SchemaValidationError,
    SchemaValidationResult,
    SchemaValidationWarning,
    ValidationErrorCode,
    ValidationWarningCode,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


# Known Anki model schemas (can be extended)
KNOWN_MODELS = {
    "APF::Simple": {
        "fields": ["Front", "Back", "Additional", "Manifest"],
        "required_fields": ["Front", "Back"],
        "card_types": [CardType.BASIC],
    },
    "APF: Simple (3.0.0)": {
        "fields": ["Front", "Back", "Additional", "Manifest"],
        "required_fields": ["Front", "Back"],
        "card_types": [CardType.BASIC],
    },
    "APF::Missing (Cloze)": {
        "fields": ["Text", "Extra", "Manifest"],
        "required_fields": ["Text"],
        "card_types": [CardType.CLOZE],
    },
    "APF::Missing": {
        "fields": ["Text", "Extra", "Manifest"],
        "required_fields": ["Text"],
        "card_types": [CardType.CLOZE],
    },
    "APF::Draw": {
        "fields": ["Prompt", "Drawing", "Notes", "Manifest"],
        "required_fields": ["Prompt", "Drawing"],
        "card_types": [CardType.BASIC],
    },
    # Generic models for the LangChain system
    "InterviewBasic": {
        "fields": ["Front", "Back", "Extra", "Hint"],
        "required_fields": ["Front", "Back"],
        "card_types": [CardType.BASIC],
    },
    "InterviewCloze": {
        "fields": ["Text", "Extra", "Hint"],
        "required_fields": ["Text"],
        "card_types": [CardType.CLOZE],
    },
}


# Default length constraints (characters)
DEFAULT_MAX_FRONT_LENGTH = 500
DEFAULT_MAX_BACK_LENGTH = 2000
DEFAULT_MAX_FIELD_LENGTH = 4000
DEFAULT_MAX_TAGS = 20


class SchemaValidatorTool:
    """Tool for validating ProposedCard against Anki model schema.

    This is a non-LLM tool that uses pure Python validation for reliability.
    It can optionally fetch model information from AnkiConnect if a client
    is provided, otherwise it falls back to known model definitions.
    """

    def __init__(
        self,
        anki_client: Optional[Any] = None,
        max_front_length: int = DEFAULT_MAX_FRONT_LENGTH,
        max_back_length: int = DEFAULT_MAX_BACK_LENGTH,
        max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
        max_tags: int = DEFAULT_MAX_TAGS,
        strict_mode: bool = False,
    ):
        """Initialize Schema Validator Tool.

        Args:
            anki_client: Optional AnkiConnect client for dynamic model fetching
            max_front_length: Maximum length for Front field
            max_back_length: Maximum length for Back field
            max_field_length: Maximum length for any field
            max_tags: Maximum number of tags
            strict_mode: If True, warnings become errors
        """
        self.anki_client = anki_client
        self.max_front_length = max_front_length
        self.max_back_length = max_back_length
        self.max_field_length = max_field_length
        self.max_tags = max_tags
        self.strict_mode = strict_mode

        # Cache for dynamically fetched models
        self._model_cache: dict[str, dict] = {}

    def validate(self, proposed_card: ProposedCard) -> SchemaValidationResult:
        """Validate a ProposedCard against Anki model schema.

        Args:
            proposed_card: The card to validate

        Returns:
            SchemaValidationResult with validation status, errors, and warnings
        """
        errors: list[SchemaValidationError] = []
        warnings: list[SchemaValidationWarning] = []

        logger.debug(
            "schema_validation_start",
            slug=proposed_card.slug,
            model=proposed_card.model_name,
            card_type=proposed_card.card_type,
        )

        # 1. Validate model exists and get schema
        model_schema = self._get_model_schema(proposed_card.model_name)
        if not model_schema:
            errors.append(
                SchemaValidationError(
                    code=ValidationErrorCode.UNKNOWN_FIELD,
                    field=None,
                    message=f"Unknown Anki model: {proposed_card.model_name}. "
                    "Model must exist in Anki or be defined in known models.",
                )
            )
            return SchemaValidationResult(valid=False, errors=errors, warnings=warnings)

        # 2. Validate required fields are present and non-empty
        for required_field in model_schema.get("required_fields", []):
            if required_field not in proposed_card.fields:
                errors.append(
                    SchemaValidationError(
                        code=ValidationErrorCode.MISSING_REQUIRED_FIELD,
                        field=required_field,  # type: ignore
                        message=f"Required field '{required_field}' is missing.",
                    )
                )
            elif not proposed_card.fields[required_field].strip():
                errors.append(
                    SchemaValidationError(
                        code=ValidationErrorCode.MISSING_REQUIRED_FIELD,
                        field=required_field,  # type: ignore
                        message=f"Required field '{required_field}' is empty.",
                    )
                )

        # 3. Validate no unknown fields
        allowed_fields = set(model_schema.get("fields", []))
        for field_name in proposed_card.fields.keys():
            if field_name not in allowed_fields:
                errors.append(
                    SchemaValidationError(
                        code=ValidationErrorCode.UNKNOWN_FIELD,
                        field=field_name,  # type: ignore
                        message=f"Field '{field_name}' not defined in model '{proposed_card.model_name}'. "
                        f"Allowed fields: {', '.join(allowed_fields)}",
                    )
                )

        # 4. Validate Cloze syntax for Cloze cards
        if proposed_card.card_type == CardType.CLOZE:
            cloze_errors = self._validate_cloze_syntax(proposed_card)
            errors.extend(cloze_errors)

        # 5. Validate length constraints
        length_warnings = self._validate_field_lengths(proposed_card)
        warnings.extend(length_warnings)

        # 6. Validate tags
        tag_warnings = self._validate_tags(proposed_card)
        warnings.extend(tag_warnings)

        # 7. In strict mode, convert warnings to errors
        if self.strict_mode:
            for warning in warnings:
                errors.append(
                    SchemaValidationError(
                        code=ValidationErrorCode.LENGTH_EXCEEDED,
                        field=warning.field,
                        message=f"[STRICT] {warning.message}",
                    )
                )
            warnings = []

        valid = len(errors) == 0

        logger.info(
            "schema_validation_complete",
            slug=proposed_card.slug,
            valid=valid,
            errors=len(errors),
            warnings=len(warnings),
        )

        return SchemaValidationResult(valid=valid, errors=errors, warnings=warnings)

    def _get_model_schema(self, model_name: str) -> Optional[dict]:
        """Get schema for an Anki model.

        First checks the cache, then tries AnkiConnect if available,
        then falls back to known models.

        Args:
            model_name: Name of the Anki model

        Returns:
            Model schema dict or None if not found
        """
        # Check cache
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        # Check known models
        if model_name in KNOWN_MODELS:
            schema = KNOWN_MODELS[model_name]
            self._model_cache[model_name] = schema
            return schema

        # Try fetching from AnkiConnect
        if self.anki_client:
            try:
                field_names = self.anki_client.get_model_field_names(model_name)
                # Create a dynamic schema
                schema = {
                    "fields": field_names,
                    "required_fields": [field_names[0]] if field_names else [],
                    "card_types": [CardType.BASIC],  # Default assumption
                }
                self._model_cache[model_name] = schema
                logger.info(
                    "model_schema_fetched",
                    model=model_name,
                    fields=field_names,
                )
                return schema
            except Exception as e:
                logger.warning(
                    "model_schema_fetch_failed",
                    model=model_name,
                    error=str(e),
                )

        return None

    def _validate_cloze_syntax(
        self, proposed_card: ProposedCard
    ) -> list[SchemaValidationError]:
        """Validate Cloze deletion syntax in card fields.

        Checks for:
        - Valid {{c1::text}} format
        - At least one cloze deletion
        - Sequential numbering

        Args:
            proposed_card: The card to validate

        Returns:
            List of validation errors
        """
        errors: list[SchemaValidationError] = []

        # Find the text field (could be "Text", "Front", etc.)
        text_field = None
        text_content = ""
        for field_name in ["Text", "Front"]:
            if field_name in proposed_card.fields:
                text_field = field_name
                text_content = proposed_card.fields[field_name]
                break

        if not text_field:
            errors.append(
                SchemaValidationError(
                    code=ValidationErrorCode.INVALID_CLOZE_FORMAT,
                    field=None,
                    message="Cloze card must have 'Text' or 'Front' field with cloze deletions.",
                )
            )
            return errors

        # Find all cloze deletions
        cloze_pattern = r"\{\{c(\d+)::(.*?)\}\}"
        matches = re.findall(cloze_pattern, text_content)

        if not matches:
            errors.append(
                SchemaValidationError(
                    code=ValidationErrorCode.INVALID_CLOZE_FORMAT,
                    field=text_field,  # type: ignore
                    message=f"Cloze card must contain at least one cloze deletion ({{{{c1::text}}}}) in '{text_field}' field.",
                )
            )
            return errors

        # Check for sequential numbering (1, 2, 3, ...)
        cloze_numbers = sorted(set(int(num) for num, _ in matches))
        if cloze_numbers[0] != 1:
            errors.append(
                SchemaValidationError(
                    code=ValidationErrorCode.INVALID_CLOZE_FORMAT,
                    field=text_field,  # type: ignore
                    message=f"Cloze deletions must start from 1 (found: {cloze_numbers[0]}).",
                )
            )

        # Check for gaps in numbering
        for i, num in enumerate(cloze_numbers[:-1], start=1):
            if cloze_numbers[i] - num > 1:
                errors.append(
                    SchemaValidationError(
                        code=ValidationErrorCode.INVALID_CLOZE_FORMAT,
                        field=text_field,  # type: ignore
                        message=f"Cloze deletions have gaps in numbering: ...{num}, {cloze_numbers[i]}...",
                    )
                )
                break

        return errors

    def _validate_field_lengths(
        self, proposed_card: ProposedCard
    ) -> list[SchemaValidationWarning]:
        """Validate field length constraints.

        Args:
            proposed_card: The card to validate

        Returns:
            List of validation warnings
        """
        warnings: list[SchemaValidationWarning] = []

        for field_name, field_value in proposed_card.fields.items():
            field_length = len(field_value)

            # Check Front field length
            if field_name == "Front" and field_length > self.max_front_length:
                warnings.append(
                    SchemaValidationWarning(
                        code=ValidationWarningCode.STYLE_WARNING,
                        field="Front",
                        message=f"Front field is too long ({field_length} chars > {self.max_front_length} recommended). "
                        "Consider shortening for better user experience.",
                    )
                )

            # Check Back field length
            if field_name == "Back" and field_length > self.max_back_length:
                warnings.append(
                    SchemaValidationWarning(
                        code=ValidationWarningCode.BACK_TOO_LONG,
                        field="Back",
                        message=f"Back field is very long ({field_length} chars > {self.max_back_length} recommended). "
                        "Consider splitting or summarizing.",
                    )
                )

            # Check any field exceeding maximum
            if field_length > self.max_field_length:
                warnings.append(
                    SchemaValidationWarning(
                        code=ValidationWarningCode.STYLE_WARNING,
                        field=field_name,  # type: ignore
                        message=f"Field '{field_name}' exceeds maximum length ({field_length} chars > {self.max_field_length}).",
                    )
                )

        return warnings

    def _validate_tags(
        self, proposed_card: ProposedCard
    ) -> list[SchemaValidationWarning]:
        """Validate tags.

        Args:
            proposed_card: The card to validate

        Returns:
            List of validation warnings
        """
        warnings: list[SchemaValidationWarning] = []

        if len(proposed_card.tags) > self.max_tags:
            warnings.append(
                SchemaValidationWarning(
                    code=ValidationWarningCode.TOO_MANY_TAGS,
                    field=None,
                    message=f"Card has {len(proposed_card.tags)} tags (> {self.max_tags} recommended). "
                    "Too many tags can clutter the card.",
                )
            )

        # Check for empty tags
        empty_tags = [tag for tag in proposed_card.tags if not tag.strip()]
        if empty_tags:
            warnings.append(
                SchemaValidationWarning(
                    code=ValidationWarningCode.STYLE_WARNING,
                    field=None,
                    message=f"Card has {len(empty_tags)} empty tags. These will be ignored.",
                )
            )

        return warnings

    def validate_batch(
        self, proposed_cards: list[ProposedCard]
    ) -> list[SchemaValidationResult]:
        """Validate multiple cards in batch.

        Args:
            proposed_cards: List of cards to validate

        Returns:
            List of validation results in the same order
        """
        results = []
        for card in proposed_cards:
            result = self.validate(card)
            results.append(result)
        return results
