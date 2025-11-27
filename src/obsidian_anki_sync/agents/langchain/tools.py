"""LangChain tool definitions for card generation and validation.

This module defines tools that can be used by LangChain agents to perform
specific operations in the card generation pipeline.
"""

from langchain_core.tools import BaseTool

from ...apf.html_validator import validate_card_html
from ...apf.linter import validate_apf
from ...sync.slug_generator import generate_slug
from ...utils.content_hash import compute_content_hash
from ...utils.logging import get_logger

logger = get_logger(__name__)


class APFValidatorTool(BaseTool):
    """Tool for validating APF card format."""

    name: str = "apf_validator"
    description: str = "Validate APF card format and structure. Returns validation results with any errors found."

    def _run(self, apf_content: str) -> str:
        """Validate APF content.

        Args:
            apf_content: The APF content to validate

        Returns:
            Validation result as string
        """
        try:
            result = validate_apf(apf_content)
            if result.valid:
                return "APF validation passed: All cards are valid."
            else:
                errors = [f"- {error}" for error in result.errors]
                return "APF validation failed:\n" + "\n".join(errors)
        except Exception as e:
            logger.error("apf_validation_tool_error", error=str(e))
            return f"APF validation error: {e}"


class HTMLFormatterTool(BaseTool):
    """Tool for formatting and validating HTML content."""

    name: str = "html_formatter"
    description: str = (
        "Format and validate HTML content. Returns formatted HTML or validation errors."
    )

    def _run(self, html_content: str) -> str:
        """Format and validate HTML content.

        Args:
            html_content: The HTML content to format

        Returns:
            Formatted HTML or validation errors
        """
        try:
            result = validate_card_html(html_content)
            if result.valid:
                return f"HTML validation passed. Formatted content:\n{result.formatted_html}"
            else:
                errors = [f"- {error}" for error in result.errors]
                return "HTML validation failed:\n" + "\n".join(errors)
        except Exception as e:
            logger.error("html_formatter_tool_error", error=str(e))
            return f"HTML formatting error: {e}"


class SlugGeneratorTool(BaseTool):
    """Tool for generating slugs for cards."""

    name: str = "slug_generator"
    description: str = "Generate unique slugs for cards based on title and base slug. Returns generated slug."

    def _run(
        self, title: str, base_slug: str, existing_slugs: list[str] | None = None
    ) -> str:
        """Generate a unique slug.

        Args:
            title: Card title
            base_slug: Base slug for the note
            existing_slugs: List of existing slugs to avoid collisions

        Returns:
            Generated unique slug
        """
        try:
            existing_slugs = existing_slugs or []
            slug = generate_slug(title, base_slug, existing_slugs)
            return f"Generated slug: {slug}"
        except Exception as e:
            logger.error("slug_generator_tool_error", error=str(e))
            return f"Slug generation error: {e}"


class ContentHashTool(BaseTool):
    """Tool for computing content hashes."""

    name: str = "content_hash"
    description: str = (
        "Compute content hash for change detection. Returns hex hash string."
    )

    def _run(self, content: str) -> str:
        """Compute content hash.

        Args:
            content: Content to hash

        Returns:
            Hex hash string
        """
        try:
            hash_value = compute_content_hash(content)
            return f"Content hash: {hash_value}"
        except Exception as e:
            logger.error("content_hash_tool_error", error=str(e))
            return f"Content hash error: {e}"


class MetadataExtractorTool(BaseTool):
    """Tool for extracting metadata from note content."""

    name: str = "metadata_extractor"
    description: str = "Extract metadata from note content including YAML frontmatter. Returns structured metadata."

    def _run(self, note_content: str) -> str:
        """Extract metadata from note content.

        Args:
            note_content: Full note content with frontmatter

        Returns:
            Structured metadata as JSON string
        """
        try:
            import yaml

            # Split frontmatter from content
            if note_content.startswith("---"):
                parts = note_content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    try:
                        metadata = yaml.safe_load(frontmatter)
                        return f"Extracted metadata: {metadata}"
                    except yaml.YAMLError as e:
                        return f"YAML parsing error: {e}"

            return "No YAML frontmatter found"
        except Exception as e:
            logger.error("metadata_extractor_tool_error", error=str(e))
            return f"Metadata extraction error: {e}"


class CardTemplateTool(BaseTool):
    """Tool for generating card templates."""

    name: str = "card_template"
    description: str = "Generate APF card templates for different card types (Simple, Missing, Draw, Cloze)."

    def _run(
        self,
        card_type: str,
        question: str,
        answer: str,
        slug: str,
        language: str = "en",
        tags: list[str] | None = None,
    ) -> str:
        """Generate a card template.

        Args:
            card_type: Type of card (Simple, Missing, Draw, Cloze)
            question: Question text
            answer: Answer text
            slug: Card slug
            language: Language code
            tags: List of tags

        Returns:
            Generated card template
        """
        try:
            tags = tags or []
            tags_str = " ".join(tags)

            template = f"""<!-- Card 1 | slug: {slug} | CardType: {card_type} | Tags: {tags_str} -->

<!-- Title -->
{question}

<!-- Key point -->
{answer}

<!-- Key point notes -->
<ul>
  <li>Main concept</li>
</ul>

<!-- manifest: {{"slug":"{slug}","lang":"{language}","type":"{card_type}","tags":{tags}}} -->"""

            return f"Generated {card_type} card template:\n{template}"

        except Exception as e:
            logger.error("card_template_tool_error", error=str(e))
            return f"Card template generation error: {e}"


class QAExtractorTool(BaseTool):
    """Tool for extracting Q&A pairs from note content."""

    name: str = "qa_extractor"
    description: str = (
        "Extract Q&A pairs from note content. Returns structured Q&A data."
    )

    def _run(self, note_content: str, language_tags: list[str] | None = None) -> str:
        """Extract Q&A pairs from note content.

        Args:
            note_content: Note content to parse
            language_tags: List of language tags

        Returns:
            Extracted Q&A pairs
        """
        try:
            from ...obsidian.parser import parse_note_content

            language_tags = language_tags or ["en"]
            qa_pairs = parse_note_content(note_content, language_tags)

            if qa_pairs:
                result = f"Extracted {len(qa_pairs)} Q&A pairs:\n"
                for i, qa in enumerate(qa_pairs, 1):
                    result += f"{i}. Q: {qa.question_en[:50]}...\n   A: {qa.answer_en[:50]}...\n"
                return result
            else:
                return "No Q&A pairs found in content"

        except Exception as e:
            logger.error("qa_extractor_tool_error", error=str(e))
            return f"Q&A extraction error: {e}"


# Tool registry for easy access
TOOL_REGISTRY = {
    "apf_validator": APFValidatorTool,
    "html_formatter": HTMLFormatterTool,
    "slug_generator": SlugGeneratorTool,
    "content_hash": ContentHashTool,
    "metadata_extractor": MetadataExtractorTool,
    "card_template": CardTemplateTool,
    "qa_extractor": QAExtractorTool,
}


def get_tool(tool_name: str, **kwargs) -> BaseTool:
    """Get a tool instance by name.

    Args:
        tool_name: Name of the tool
        **kwargs: Additional arguments for tool initialization

    Returns:
        Tool instance

    Raises:
        ValueError: If tool name is not found
    """
    if tool_name not in TOOL_REGISTRY:
        available_tools = list(TOOL_REGISTRY.keys())
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {available_tools}"
        )

    tool_class = TOOL_REGISTRY[tool_name]
    return tool_class(**kwargs)


def get_tools_for_agent(agent_type: str) -> list[BaseTool]:
    """Get recommended tools for a specific agent type.

    Args:
        agent_type: Type of agent (generator, validator, etc.)

    Returns:
        List of recommended tools
    """
    tool_sets = {
        "generator": [
            "card_template",
            "slug_generator",
            "html_formatter",
            "apf_validator",
        ],
        "validator": [
            "apf_validator",
            "html_formatter",
            "content_hash",
        ],
        "pre_validator": [
            "metadata_extractor",
            "qa_extractor",
            "content_hash",
        ],
        "post_validator": [
            "apf_validator",
            "html_formatter",
        ],
        "enrichment": [
            "metadata_extractor",
            "card_template",
        ],
    }

    tool_names = tool_sets.get(agent_type, [])
    return [get_tool(name) for name in tool_names]
