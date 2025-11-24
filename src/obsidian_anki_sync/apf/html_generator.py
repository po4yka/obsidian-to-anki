"""Structured HTML generation with templates and validation.

Provides template-based APF HTML generation with built-in validation
to ensure consistent, well-formed HTML output.
"""

import html
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from html import escape

from ..utils.logging import get_logger
from .html_validator import validate_card_html

logger = get_logger(__name__)


@dataclass
class CardTemplate:
    """Template for generating APF card HTML."""
    card_type: str
    sections: Dict[str, str]

    def render(self, data: Dict[str, Any]) -> str:
        """Render the template with provided data."""
        html_parts = []

        # Render each section
        for section_name, template in self.sections.items():
            if section_name in data and data[section_name]:
                rendered = self._render_section(template, data)
                if rendered:
                    html_parts.append(rendered)

        return '\n'.join(html_parts)

    def _render_section(self, template: str, data: Dict[str, Any]) -> str:
        """Render a single section template."""
        # Simple template substitution
        result = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        return result


@dataclass
class GenerationResult:
    """Result of HTML generation with validation."""
    html: str
    is_valid: bool
    validation_errors: List[str]
    warnings: List[str]


class HTMLTemplateGenerator:
    """Structured HTML generation for APF cards with validation."""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, CardTemplate]:
        """Initialize available card templates."""
        return {
            'simple': CardTemplate(
                card_type='Simple',
                sections={
                    'header': '<!-- Card {card_index} | slug: {slug} | CardType: Simple | Tags: {tags} -->',
                    'title': '\n<!-- Title -->\n{title}',
                    'question': '\n<!-- Question -->\n{question}',
                    'answer': '\n<!-- Answer -->\n{answer}',
                    'key_points': '\n<!-- Key point -->\n{key_points}',
                    'notes': '\n<!-- Other notes -->\n{other_notes}' if '{other_notes}' else '',
                    'references': '\n<!-- References -->\n{references}' if '{references}' else '',
                }
            ),
            'code_block': CardTemplate(
                card_type='Simple',
                sections={
                    'header': '<!-- Card {card_index} | slug: {slug} | CardType: Simple | Tags: {tags} -->',
                    'title': '\n<!-- Title -->\n{title}',
                    'code_sample': '\n<!-- Sample (code block) -->\n{code_sample}',
                    'key_points': '\n<!-- Key point -->\n{key_points}',
                    'notes': '\n<!-- Other notes -->\n{other_notes}' if '{other_notes}' else '',
                }
            ),
            'cloze': CardTemplate(
                card_type='Missing',  # Cloze cards use "Missing" type
                sections={
                    'header': '<!-- Card {card_index} | slug: {slug} | CardType: Missing | Tags: {tags} -->',
                    'title': '\n<!-- Title -->\n{title}',
                    'content': '\n<!-- Content -->\n{content}',
                }
            )
        }

    def generate_card_html(self, card_data: Dict[str, Any], template_name: str = 'simple') -> GenerationResult:
        """
        Generate APF HTML for a card using templates.

        Args:
            card_data: Card data dictionary
            template_name: Template to use ('simple', 'code_block', 'cloze')

        Returns:
            GenerationResult with HTML and validation info
        """
        # Select appropriate template
        template = self.templates.get(template_name, self.templates['simple'])

        # Pre-process data
        processed_data = self._preprocess_card_data(card_data)

        # Generate HTML
        html_content = template.render(processed_data)

        # Validate and fix
        validation_errors = validate_card_html(html_content)
        warnings = []

        if validation_errors:
            html_content, fix_warnings = self._auto_fix_html_issues(
                html_content, validation_errors)
            warnings.extend(fix_warnings)

            # Re-validate after fixes
            final_errors = validate_card_html(html_content)
            if final_errors:
                validation_errors.extend(final_errors)
                warnings.append(
                    "Some HTML validation issues could not be auto-fixed")

        return GenerationResult(
            html=html_content,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            warnings=warnings
        )

    def _preprocess_card_data(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess card data for template rendering."""
        processed = card_data.copy()

        # Ensure required fields
        processed.setdefault('card_index', 1)
        processed.setdefault('slug', f'card-{processed["card_index"]}')
        processed.setdefault('tags', '')
        processed.setdefault('title', 'Untitled Card')

        # Process code blocks
        if 'code_sample' in processed:
            processed['code_sample'] = self._generate_code_block(
                processed['code_sample'])

        if 'key_points' in processed:
            processed['key_points'] = self._generate_key_points(
                processed['key_points'])

        # Process text content
        for field in ['question', 'answer', 'title', 'other_notes', 'content']:
            if field in processed and processed[field]:
                processed[field] = self._escape_and_format_text(
                    processed[field])

        return processed

    def _generate_code_block(self, code_data: Any) -> str:
        """Generate properly formatted code block HTML."""
        if isinstance(code_data, str):
            # Simple code string
            return self._create_code_html(code_data, 'text')
        elif isinstance(code_data, dict):
            # Structured code data
            code = code_data.get('code', '')
            language = code_data.get('language', 'text')
            caption = code_data.get('caption', '')
            return self._create_code_html(code, language, caption)
        else:
            return '<pre><code>Invalid code data</code></pre>'

    def _create_code_html(self, code: str, language: str, caption: str = '') -> str:
        """Create properly formatted code HTML."""
        escaped_code = escape(code.strip())
        code_html = f'<pre><code class="language-{language}">{escaped_code}</code></pre>'

        if caption:
            return f'<figure>\n{code_html}\n<figcaption>{escape(caption)}</figcaption>\n</figure>'
        else:
            return code_html

    def _generate_key_points(self, key_points: Any) -> str:
        """Generate key points HTML."""
        if isinstance(key_points, str):
            # Simple string
            return f'<ul>\n<li>{escape(key_points)}</li>\n</ul>'
        elif isinstance(key_points, list):
            # List of points
            points_html = '\n'.join(
                f'<li>{escape(point)}</li>' for point in key_points)
            return f'<ul>\n{points_html}\n</ul>'
        else:
            return '<ul><li>Key points data</li></ul>'

    def _escape_and_format_text(self, text: str) -> str:
        """Escape and format text content."""
        if not text:
            return ''

        # Basic HTML escaping
        escaped = escape(text)

        # Convert basic markdown to HTML
        escaped = re.sub(r'\*\*(.*?)\*\*',
                         r'<strong>\1</strong>', escaped)  # Bold
        escaped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped)  # Italic

        # Convert line breaks
        escaped = escaped.replace('\n', '<br>')

        return escaped

    def _auto_fix_html_issues(self, html: str, errors: List[str]) -> Tuple[str, List[str]]:
        """Attempt to auto-fix common HTML validation issues."""
        fixed_html = html
        warnings = []

        for error in errors:
            if 'language- class' in error:
                # Add default language class to code elements without one
                fixed_html, lang_warnings = self._add_missing_language_classes(
                    fixed_html)
                warnings.extend(lang_warnings)
            elif 'wrap in <pre><code>' in error:
                # Wrap standalone code elements
                fixed_html, wrap_warnings = self._wrap_standalone_code(
                    fixed_html)
                warnings.extend(wrap_warnings)
            elif 'Backtick code fences detected' in error:
                # Remove markdown code fences from HTML
                fixed_html = re.sub(
                    r'```[^\n]*\n(.*?)\n```', r'<pre><code>\1</code></pre>', fixed_html, flags=re.DOTALL)
                warnings.append("Converted markdown code fences to HTML")

        return fixed_html, warnings

    def _add_missing_language_classes(self, html: str) -> Tuple[str, List[str]]:
        """Add default language classes to code elements missing them."""
        warnings = []

        def add_class(match):
            code_tag = match.group(0)
            if 'class=' not in code_tag:
                # Add default language class
                warnings.append("Added default language class to code element")
                return code_tag.replace('<code', '<code class="language-text"', 1)
            return code_tag

        # Find code tags without language classes
        pattern = r'<code(?:\s[^>]*)?>'
        fixed_html = re.sub(pattern, add_class, html)

        return fixed_html, warnings

    def _wrap_standalone_code(self, html: str) -> Tuple[str, List[str]]:
        """Wrap standalone code elements in pre tags."""
        warnings = []

        def wrap_code(match):
            code_content = match.group(1)
            warnings.append("Wrapped standalone code element in pre tags")
            return f'<pre>{match.group(0)}</pre>'

        # Find code elements not inside pre tags
        # This is a simplified version - a full implementation would need more complex parsing
        fixed_html = html
        warnings.append(
            "Standalone code wrapping not fully implemented - requires complex HTML parsing")

        return fixed_html, warnings

    def generate_full_apf_html(self, cards_data: List[Dict[str, Any]], manifest_data: Dict[str, Any]) -> str:
        """
        Generate complete APF HTML document with all cards.

        Args:
            cards_data: List of card data dictionaries
            manifest_data: Document-level metadata

        Returns:
            Complete APF HTML document
        """
        parts = []

        # Add sentinels and manifest
        parts.append("<!-- PROMPT_VERSION: apf-v2.1 -->")
        parts.append("<!-- BEGIN_CARDS -->")
        parts.append("")

        # Generate each card
        for card_data in cards_data:
            result = self.generate_card_html(card_data)
            parts.append(result.html)
            parts.append("")  # Blank line between cards

            if result.warnings:
                logger.warning("card_generation_warnings",
                               card_index=card_data.get('card_index'),
                               warnings=result.warnings)

        # Close document
        parts.append("<!-- END_CARDS -->")
        parts.append("END_OF_CARDS")

        return '\n'.join(parts)


def generate_card_html_with_validation(card_data: Dict[str, Any], template_name: str = 'simple') -> GenerationResult:
    """
    Convenience function to generate card HTML with validation.

    Args:
        card_data: Card data dictionary
        template_name: Template to use

    Returns:
        GenerationResult with validated HTML
    """
    generator = HTMLTemplateGenerator()
    return generator.generate_card_html(card_data, template_name)
