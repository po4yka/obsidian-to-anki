"""Auto-improvement agent for enhancing card quality based on assessment feedback."""

import re
import time

from obsidian_anki_sync.models import NoteMetadata
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

from .models import GeneratedCard, QualityReport


def _extract_qa_from_apf(apf_html: str) -> tuple[str, str]:
    """Extract question and answer from APF HTML.

    Args:
        apf_html: APF format HTML

    Returns:
        Tuple of (question, answer)
    """
    question = ""
    answer = ""

    # Extract Front (question)
    front_match = re.search(r'<div class="front">(.*?)</div>', apf_html, re.DOTALL)
    if front_match:
        question = re.sub(r"<[^>]+>", "", front_match.group(1)).strip()

    # Extract Back (answer)
    back_match = re.search(r'<div class="back">(.*?)</div>', apf_html, re.DOTALL)
    if back_match:
        answer = re.sub(r"<[^>]+>", "", back_match.group(1)).strip()

    return question or "Unknown question", answer or "Unknown answer"


logger = get_logger(__name__)


class CardImprover:
    """Agent for automatically improving card quality based on assessment results.

    Uses both rule-based fixes for common issues and LLM-powered improvements
    for complex content problems.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        model: str = "qwen3:14b",
        temperature: float = 0.0,
    ):
        """Initialize card improver.

        Args:
            llm_provider: LLM provider for complex improvements (optional)
            model: Model to use for LLM-powered improvements
            temperature: Sampling temperature
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature

        logger.info("card_improver_initialized", has_llm=llm_provider is not None)

    def improve_card(
        self,
        card: GeneratedCard,
        quality_report: QualityReport,
        metadata: NoteMetadata,
        auto_apply: bool = True,
    ) -> GeneratedCard:
        """Improve a card based on quality assessment results.

        Args:
            card: Original card to improve
            quality_report: Quality assessment results
            metadata: Note metadata for context
            auto_apply: Whether to apply improvements automatically

        Returns:
            Improved card (or original if no improvements applied)
        """
        if quality_report.overall_score >= 0.8:
            logger.debug(
                "card_quality_sufficient",
                slug=card.slug,
                score=quality_report.overall_score,
            )
            return card

        start_time = time.time()
        improved_card = card.model_copy(deep=True)  # Create a deep copy

        logger.info(
            "improving_card",
            slug=card.slug,
            current_score=quality_report.overall_score,
            suggestions=len(quality_report.suggestions),
        )

        # Apply rule-based fixes first (deterministic, fast)
        improved_card = self._apply_rule_based_fixes(improved_card, quality_report)

        # Apply LLM-powered improvements for complex issues
        if self.llm_provider and quality_report.overall_score < 0.6:
            improved_card = self._apply_llm_improvements(
                improved_card, quality_report, metadata
            )

        improvement_time = time.time() - start_time

        logger.info(
            "card_improved",
            slug=improved_card.slug,
            improvement_time=improvement_time,
        )

        return improved_card

    def _apply_rule_based_fixes(
        self, card: GeneratedCard, quality_report: QualityReport
    ) -> GeneratedCard:
        """Apply deterministic rule-based fixes for common issues."""
        improved = card.model_copy()

        suggestions = quality_report.suggestions

        # Extract Q&A from APF HTML for rule-based fixes
        question, _answer = _extract_qa_from_apf(improved.apf_html)

        for suggestion in suggestions:
            if "punctuation" in suggestion.lower():
                fixed_question = self._fix_punctuation(question)
                # Update APF HTML with fixed question
                improved.apf_html = improved.apf_html.replace(
                    f'<div class="front">{question}</div>',
                    f'<div class="front">{fixed_question}</div>',
                    1,
                )
                question = fixed_question
            elif "capitalization" in suggestion.lower():
                fixed_question = self._fix_capitalization(question)
                improved.apf_html = improved.apf_html.replace(
                    f'<div class="front">{question}</div>',
                    f'<div class="front">{fixed_question}</div>',
                    1,
                )
                question = fixed_question
            elif "unclosed" in suggestion.lower() and "tag" in suggestion.lower():
                improved.apf_html = self._fix_html_tags(improved.apf_html)
            elif "language class" in suggestion.lower():
                improved.apf_html = self._add_code_language_classes(improved.apf_html)
            elif "slug format" in suggestion.lower():
                improved.slug = self._fix_slug_format(improved.slug)
            elif (
                "insufficient tags" in suggestion.lower()
                or "duplicate tags" in suggestion.lower()
            ):
                # Note: GeneratedCard doesn't have tags - tags are in APF HTML/manifest
                # Skip tag-related fixes for GeneratedCard
                pass
            elif "alt text" in suggestion.lower():
                improved.apf_html = self._add_image_alt_text(improved.apf_html)

        return improved

    def _apply_llm_improvements(
        self,
        card: GeneratedCard,
        quality_report: QualityReport,
        metadata: NoteMetadata,
    ) -> GeneratedCard:
        """Apply LLM-powered improvements for complex content issues."""
        if not self.llm_provider:
            return card

        # Only apply LLM improvements for very low quality cards or specific issues
        complex_issues = [
            issue
            for issue in quality_report.suggestions
            if any(
                keyword in issue.lower()
                for keyword in [
                    "ambiguous",
                    "unclear",
                    "too complex",
                    "cognitive load",
                    "passive",
                    "active recall",
                    "context",
                ]
            )
        ]

        if not complex_issues or quality_report.overall_score > 0.5:
            return card

        try:
            prompt = self._build_improvement_prompt(card, complex_issues, metadata)

            response = self.llm_provider.generate(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
            )

            # Extract response text from dict
            response_text = (
                response.get("response", "")
                if isinstance(response, dict)
                else str(response)
            )
            improved_content = self._parse_improvement_response(response_text)
            if improved_content:
                improved_card = card.model_copy()
                # Update APF HTML with improved content
                question, answer = _extract_qa_from_apf(card.apf_html)
                new_question = improved_content.get("question", question)
                new_answer = improved_content.get("answer", answer)
                # Replace in APF HTML
                if new_question != question:
                    improved_card.apf_html = improved_card.apf_html.replace(
                        f'<div class="front">{question}</div>',
                        f'<div class="front">{new_question}</div>',
                        1,
                    )
                if new_answer != answer:
                    improved_card.apf_html = improved_card.apf_html.replace(
                        f'<div class="back">{answer}</div>',
                        f'<div class="back">{new_answer}</div>',
                        1,
                    )
                return improved_card

        except Exception as e:
            logger.warning("llm_improvement_failed", error=str(e), slug=card.slug)

        return card

    def _fix_punctuation(self, text: str) -> str:
        """Add proper punctuation to questions."""
        text = text.strip()
        if not text:
            return text

        # Add question mark if missing and text looks like a question
        question_starters = ["what", "how", "why", "when", "where", "which", "who"]
        if any(text.lower().startswith(word) for word in question_starters):
            if not text.endswith("?"):
                text += "?"

        # Add period if missing and not a question
        elif not text.endswith((".", "!", "?")):
            text += "."

        return text

    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization issues."""
        if not text:
            return text

        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else text

        return text

    def _fix_html_tags(self, text: str) -> str:
        """Fix common HTML tag issues."""
        # Fix unclosed <pre> tags
        pre_count = text.count("<pre>")
        pre_close_count = text.count("</pre>")

        if pre_count > pre_close_count:
            text += "</pre>" * (pre_count - pre_close_count)

        return text

    def _add_code_language_classes(self, text: str) -> str:
        """Add language classes to code blocks."""
        # Find code blocks without language classes
        pattern = r"(<pre><code)([^>]*>.*?)</code></pre>"

        def add_language_class(match: re.Match[str]) -> str:
            code_tag = match.group(1)
            rest = match.group(2)

            # Skip if already has language class
            if 'class="language-' in code_tag:
                return match.group(0)

            # Try to detect language from content
            code_content = rest.replace("</code>", "").replace(">", "")
            detected_lang = self._detect_code_language(code_content)

            if detected_lang:
                return f'{code_tag} class="language-{detected_lang}"{rest}'
            else:
                # Default to 'text' if can't detect
                return f'{code_tag} class="language-text"{rest}'

        return re.sub(pattern, add_language_class, text, flags=re.DOTALL)

    # type: ignore[no-untyped-def]
    def _detect_code_language(self, code: str) -> str | None:
        """Simple language detection based on keywords."""
        code_lower = code.lower()

        # Python indicators
        if any(
            keyword in code_lower for keyword in ["def ", "import ", "print(", "class "]
        ):
            return "python"

        # JavaScript indicators
        if any(
            keyword in code_lower
            for keyword in ["function ", "const ", "let ", "console.log"]
        ):
            return "javascript"

        # Java indicators
        if any(
            keyword in code_lower
            for keyword in ["public class", "system.out", "void main"]
        ):
            return "java"

        # Kotlin indicators
        if any(
            keyword in code_lower for keyword in ["fun ", "val ", "var ", "println"]
        ):
            return "kotlin"

        # SQL indicators
        if any(
            keyword in code_lower
            for keyword in ["select ", "from ", "where ", "insert "]
        ):
            return "sql"

        # Bash indicators
        if any(
            keyword in code_lower for keyword in ["#!/bin/bash", "echo ", "ls ", "cd "]
        ):
            return "bash"

        return None

    def _fix_slug_format(self, slug: str) -> str:
        """Fix slug to use only lowercase letters, numbers, and hyphens."""
        if not slug:
            return slug

        # Convert to lowercase
        slug = slug.lower()

        # Replace spaces and underscores with hyphens
        slug = re.sub(r"[_\s]+", "-", slug)

        # Remove invalid characters
        slug = re.sub(r"[^a-z0-9-]", "", slug)

        # Remove multiple consecutive hyphens
        slug = re.sub(r"-+", "-", slug)

        # Remove leading/trailing hyphens
        slug = slug.strip("-")

        return slug

    def _add_missing_tags(self, tags: list[str], question: str) -> list[str]:
        """Add relevant tags if there are fewer than 3."""
        if len(tags) >= 3:
            return tags

        new_tags = tags.copy()

        # Add programming-related tags based on content
        question_lower = question.lower()

        if "kotlin" in question_lower and "kotlin" not in new_tags:
            new_tags.append("kotlin")
        if "android" in question_lower and "android" not in new_tags:
            new_tags.append("android")
        if "function" in question_lower and "programming" not in new_tags:
            new_tags.append("programming")
        if "api" in question_lower and "api" not in new_tags:
            new_tags.append("api")

        # Ensure we have at least one language tag
        has_language_tag = any(
            tag in ["kotlin", "java", "python", "javascript", "typescript"]
            for tag in new_tags
        )
        if not has_language_tag:
            new_tags.append("programming")

        return new_tags

    def _remove_duplicate_tags(self, tags: list[str]) -> list[str]:
        """Remove duplicate tags while preserving order."""
        seen = set()
        unique_tags = []

        for tag in tags:
            if tag not in seen:
                unique_tags.append(tag)
                seen.add(tag)

        return unique_tags

    def _add_image_alt_text(self, text: str) -> str:
        """Add alt text to images that don't have it."""
        pattern = r"(<img[^>]+)>"  # Find img tags

        def add_alt(match: re.Match[str]) -> str:
            img_tag = match.group(1)
            if "alt=" in img_tag:
                return match.group(0)  # Already has alt text

            # Add generic alt text
            return f'{img_tag} alt="Diagram or code example">'

        return re.sub(pattern, add_alt, text)

    def _build_improvement_prompt(
        self,
        card: GeneratedCard,
        issues: list[str],
        metadata: NoteMetadata,
    ) -> str:
        """Build prompt for LLM-powered card improvements."""
        return f"""You are an expert flashcard author specializing in programming education.

IMPROVE THIS FLASHCARD based on the following issues:
{chr(10).join(f"- {issue}" for issue in issues)}

ORIGINAL CARD:
{card.apf_html}

Card Type: {"Missing" if "{{c" in card.apf_html else ("Draw" if "<img " in card.apf_html and "svg" in card.apf_html.lower() else "Simple")}

CONTEXT:
Note Title: {metadata.title}
Tags: {", ".join(metadata.tags) if metadata.tags else "None"}

REQUIREMENTS:
1. Fix all identified issues while maintaining the core learning objective
2. Ensure the question requires active recall (not passive recognition)
3. Keep the answer concise but complete
4. Follow APF v2.1 format guidelines
5. Use proper technical terminology

Return the improved card in this exact JSON format:
{{
  "question": "Improved question text",
  "answer": "Improved answer text"
}}

Only return the JSON, no other text or explanations."""

    def _parse_improvement_response(
        self, response: str
    ) -> dict[str, str | None] | None:
        """Parse LLM improvement response."""
        try:
            import json

            result = json.loads(response.strip())
            if isinstance(result, dict) and "question" in result:
                return result
        except (json.JSONDecodeError, KeyError):
            pass

        return None
