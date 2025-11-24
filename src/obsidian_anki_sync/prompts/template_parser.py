"""Parse and process prompt templates with YAML frontmatter."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import ConfigurationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Parsed prompt template with configuration."""

    deck: str | None = None
    note_type: str | None = None
    field_map: dict[str, str] | None = None
    quality_check: dict[str, Any] | None = None
    prompt_body: str = ""

    def substitute(self, **kwargs: Any) -> str:
        """
        Substitute variables in prompt body.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Prompt with variables substituted
        """
        result = self.prompt_body
        for key, value in kwargs.items():
            # Replace {key} with value
            result = result.replace(f"{{{key}}}", str(value))
        return result


def parse_template_file(template_path: Path) -> PromptTemplate:
    """
    Parse a prompt template file with YAML frontmatter.

    Args:
        template_path: Path to template file

    Returns:
        Parsed PromptTemplate

    Raises:
        ConfigurationError: If template is invalid
    """
    if not template_path.exists():
        raise ConfigurationError(f"Template file not found: {template_path}")

    with template_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Split frontmatter and body
    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)

    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        prompt_body = frontmatter_match.group(2)
    else:
        # No frontmatter, entire file is prompt body
        frontmatter_text = ""
        prompt_body = content

    # Parse YAML frontmatter
    config: dict[str, Any] = {}
    if frontmatter_text:
        try:
            config = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML frontmatter: {e}") from e

    # Extract configuration
    template = PromptTemplate(
        deck=config.get("deck"),
        note_type=config.get("noteType") or config.get("note_type"),
        field_map=config.get("fieldMap") or config.get("field_map"),
        quality_check=config.get(
            "qualityCheck") or config.get("quality_check"),
        prompt_body=prompt_body.strip(),
    )

    logger.debug(
        "template_parsed",
        template_path=str(template_path),
        has_deck=template.deck is not None,
        has_field_map=template.field_map is not None,
        has_quality_check=template.quality_check is not None,
    )

    return template


def validate_template(template: PromptTemplate) -> list[str]:
    """
    Validate a prompt template.

    Args:
        template: PromptTemplate to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not template.prompt_body:
        errors.append("Prompt body is empty")

    # Check for required placeholders if field_map is specified
    if template.field_map:
        required_placeholders = set(template.field_map.keys())
        found_placeholders = set(re.findall(
            r"\{(\w+)\}", template.prompt_body))

        missing = required_placeholders - found_placeholders
        if missing:
            errors.append(f"Missing placeholders: {', '.join(missing)}")

    return errors
