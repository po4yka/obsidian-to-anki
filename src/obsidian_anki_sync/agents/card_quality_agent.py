"""Enhanced card quality assessment agent with multi-dimensional evaluation."""

import re
import time

from ..models import NoteMetadata
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .models import GeneratedCard, QualityDimension, QualityReport

logger = get_logger(__name__)


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


class CardQualityAgent:
    """Enhanced agent for comprehensive card quality assessment.

    Evaluates cards across multiple dimensions using evidence-based criteria
    from learning science research and Anki best practices.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "qwen3:14b",
        temperature: float = 0.0,
    ):
        """Initialize quality assessment agent.

        Args:
            llm_provider: LLM provider for quality evaluation
            model: Model to use for assessment
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature

        logger.info("card_quality_agent_initialized", model=model)

    def assess_card_quality(
        self,
        card: GeneratedCard,
        metadata: NoteMetadata,
        context_cards: list[GeneratedCard | None] | None = None,
    ) -> QualityReport:
        """Perform comprehensive quality assessment of a card.

        Args:
            card: Card to assess
            metadata: Note metadata for context
            context_cards: Other cards from same note for coherence checking

        Returns:
            Comprehensive quality report
        """
        start_time = time.time()

        # Extract card type and Q&A from APF HTML
        card_type = "Simple"
        if "{{c" in card.apf_html:
            card_type = "Missing"
        elif "<img " in card.apf_html and "svg" in card.apf_html.lower():
            card_type = "Draw"
        question, answer = _extract_qa_from_apf(card.apf_html)
        logger.debug("assessing_card_quality", slug=card.slug, card_type=card_type)

        # Assess each quality dimension
        dimensions = self._assess_all_dimensions(card, metadata, context_cards)

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimensions)

        # Generate improvement suggestions
        suggestions = self._generate_improvements(card, dimensions)

        # Calculate confidence in assessment
        confidence = self._calculate_confidence(card, dimensions)

        assessment_time = time.time() - start_time

        report = QualityReport(
            overall_score=overall_score,
            dimensions=dimensions,
            suggestions=suggestions,
            confidence=confidence,
            assessment_time=assessment_time,
        )

        logger.info(
            "card_quality_assessed",
            slug=card.slug,
            overall_score=overall_score,
            confidence=confidence,
            assessment_time=assessment_time,
        )

        return report

    def _assess_all_dimensions(
        self,
        card: GeneratedCard,
        metadata: NoteMetadata,
        context_cards: list[GeneratedCard | None] | None = None,
    ) -> dict[str, QualityDimension]:
        """Assess card across all quality dimensions."""
        dimensions = {}

        # Content Quality (40% weight)
        dimensions["content"] = self._assess_content_quality(card, metadata)

        # Learning Science (30% weight)
        dimensions["learning_science"] = self._assess_learning_science(card)

        # Technical Quality (20% weight)
        dimensions["technical"] = self._assess_technical_quality(card)

        # Accessibility (10% weight)
        dimensions["accessibility"] = self._assess_accessibility(card)

        return dimensions

    def _assess_content_quality(
        self, card: GeneratedCard, metadata: NoteMetadata
    ) -> QualityDimension:
        """Assess content accuracy, clarity, and educational value."""
        issues = []
        score = 1.0  # Start with perfect score, deduct for issues

        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        # Check for placeholder content
        if any(
            placeholder in question.lower()
            for placeholder in ["tbd", "todo", "fill in"]
        ):
            issues.append("Contains placeholder content")
            score -= 0.3

        # Check for incomplete code examples
        if "..." in question or "..." in answer:
            issues.append("Contains incomplete code examples")
            score -= 0.2

        # Check question clarity
        if len(question.strip()) < 10:
            issues.append("Question too short or unclear")
            score -= 0.2

        # Check for proper capitalization and punctuation
        question_stripped = question.strip()
        if question_stripped and (
            not question_stripped[0].isupper()
            or not question_stripped.endswith(("?", ".", "!"))
        ):
            issues.append("Question lacks proper capitalization or punctuation")
            score -= 0.1

        # Check answer quality
        if answer:
            answer_stripped = answer.strip()
            if len(answer_stripped) < 5:
                issues.append("Answer too brief or incomplete")
                score -= 0.2

        # Verify content matches programming domain
        tech_indicators = ["function", "class", "method", "api", "code", "programming"]
        has_tech_content = any(
            indicator in question.lower() for indicator in tech_indicators
        )
        if (
            metadata.tags
            and "programming" in " ".join(metadata.tags)
            and not has_tech_content
        ):
            issues.append("Content doesn't match programming domain")
            score -= 0.2

        score = max(0.0, score)  # Ensure non-negative score

        return QualityDimension(
            score=score,
            weight=0.4,
            issues=issues,
            strengths=self._identify_content_strengths(card),
        )

    def _assess_learning_science(self, card: GeneratedCard) -> QualityDimension:
        """Assess adherence to evidence-based learning principles."""
        issues = []
        score = 1.0

        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        # Extract card type
        card_type = "Simple"
        if "{{c" in card.apf_html:
            card_type = "Missing"
        elif "<img " in card.apf_html and "svg" in card.apf_html.lower():
            card_type = "Draw"

        # Atomic principle: One concept per card
        question_words = len(question.split())
        if question_words > 25:
            issues.append("Question too long - may test multiple concepts")
            score -= 0.2

        # Active recall: Question should require memory retrieval
        passive_indicators = ["is", "are", "does", "do", "can", "should", "would"]
        question_lower = question.lower()
        passive_count = sum(1 for word in passive_indicators if word in question_lower)
        if passive_count > 2:
            issues.append(
                "Question may be passive recognition rather than active recall"
            )
            score -= 0.15

        # Cognitive load: Answer shouldn't be too long
        if answer and len(answer) > 500:
            issues.append("Answer too long - high cognitive load")
            score -= 0.2

        # Context sufficiency: Question should have enough context
        if len(question.split()) < 5:
            issues.append("Question lacks sufficient context for unambiguous recall")
            score -= 0.1

        # Cloze quality (if applicable)
        if card_type == "Missing":
            cloze_pattern = r"\{\{c\d+::[^}]+\}\}"
            clozes = re.findall(cloze_pattern, question + answer)
            if len(clozes) > 3:
                issues.append("Too many clozes - may be dependent concepts")
                score -= 0.15
            elif len(clozes) == 0:
                issues.append("Missing card type but no clozes found")
                score -= 0.3

        score = max(0.0, score)

        return QualityDimension(
            score=score,
            weight=0.3,
            issues=issues,
            strengths=self._identify_learning_strengths(card),
        )

    def _assess_technical_quality(self, card: GeneratedCard) -> QualityDimension:
        """Assess technical correctness and formatting."""
        issues = []
        score = 1.0

        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        # Check HTML structure in APF HTML
        if "<pre>" in card.apf_html and "</pre>" not in card.apf_html:
            issues.append("Unclosed <pre> tag in APF HTML")
            score -= 0.2

        # Check code block language specification
        code_blocks = re.findall(
            r"<pre><code[^>]*>(.*?)</code></pre>",
            card.apf_html,
            re.DOTALL,
        )
        for block in code_blocks:
            if not re.search(r'class="language-[^"]*"', block):
                issues.append("Code block missing language class")
                score -= 0.1

        # Check for proper slug format
        if not re.match(r"^[a-z0-9-]+$", card.slug):
            issues.append("Invalid slug format (should be lowercase with hyphens only)")
            score -= 0.1

        # Note: GeneratedCard doesn't have tags attribute - tags are in APF HTML/manifest
        # This check is skipped for GeneratedCard model

        score = max(0.0, score)

        return QualityDimension(
            score=score,
            weight=0.2,
            issues=issues,
            strengths=self._identify_technical_strengths(card),
        )

    def _assess_accessibility(self, card: GeneratedCard) -> QualityDimension:
        """Assess accessibility compliance."""
        issues = []
        score = 1.0

        # Check for images without alt text
        img_pattern = r"<img[^>]+>"
        images = re.findall(img_pattern, card.apf_html)
        for img in images:
            if "alt=" not in img:
                issues.append("Image missing alt text for screen readers")
                score -= 0.2

        # Check for sufficient color contrast (basic check)
        if "color:" in card.apf_html.lower():
            # If colors are specified, they should meet contrast requirements
            # This is a basic check - full WCAG compliance would need more analysis
            issues.append(
                "Color usage detected - verify contrast ratios meet WCAG standards"
            )
            score -= 0.1

        # Check for semantic HTML structure
        if card.apf_html.count("<p>") > 10:
            issues.append("Excessive paragraph breaks may indicate poor structure")
            score -= 0.1

        score = max(0.0, score)

        return QualityDimension(
            score=score,
            weight=0.1,
            issues=issues,
            strengths=self._identify_accessibility_strengths(card),
        )

    def _calculate_overall_score(
        self, dimensions: dict[str, QualityDimension]
    ) -> float:
        """Calculate weighted overall quality score."""
        total_weighted_score = 0.0
        total_weight = 0.0

        for dimension in dimensions.values():
            total_weighted_score += dimension.score * dimension.weight
            total_weight += dimension.weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _generate_improvements(
        self, card: GeneratedCard, dimensions: dict[str, QualityDimension]
    ) -> list[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []

        for dim_name, dimension in dimensions.items():
            if (
                dimension.score < 0.8
            ):  # Only suggest improvements for low-scoring dimensions
                suggestions.extend(
                    self._get_dimension_improvements(dim_name, dimension.issues)
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)

        return unique_suggestions

    def _get_dimension_improvements(
        self, dimension: str, issues: list[str]
    ) -> list[str]:
        """Get improvement suggestions for a specific dimension."""
        suggestions = []

        for issue in issues:
            if "placeholder" in issue.lower():
                suggestions.append(
                    "Replace placeholder content with actual code examples"
                )
            elif "incomplete" in issue.lower():
                suggestions.append("Complete partial code examples with runnable code")
            elif "too short" in issue.lower():
                suggestions.append("Add more context to make the question unambiguous")
            elif "punctuation" in issue.lower():
                suggestions.append("Add proper punctuation and capitalization")
            elif "too brief" in issue.lower():
                suggestions.append("Expand answer with more complete explanation")
            elif "doesn't match" in issue.lower():
                suggestions.append(
                    "Ensure content aligns with the specified domain/tags"
                )
            elif "too long" in issue.lower():
                suggestions.append("Split complex questions into multiple atomic cards")
            elif "passive" in issue.lower():
                suggestions.append(
                    "Rewrite as active recall question (What/How/Why format)"
                )
            elif "cognitive load" in issue.lower():
                suggestions.append("Break down complex answers into simpler components")
            elif "clozes" in issue.lower():
                suggestions.append(
                    "Review cloze deletions to ensure independent concepts"
                )
            elif "unclosed" in issue.lower():
                suggestions.append("Fix HTML tag structure")
            elif "language class" in issue.lower():
                suggestions.append("Add language specification to code blocks")
            elif "slug format" in issue.lower():
                suggestions.append(
                    "Use lowercase letters, numbers, and hyphens only in slug"
                )
            elif "insufficient tags" in issue.lower():
                suggestions.append("Add at least 3 relevant tags")
            elif "duplicate tags" in issue.lower():
                suggestions.append("Remove duplicate tags")
            elif "alt text" in issue.lower():
                suggestions.append("Add descriptive alt text for all images")
            elif "color" in issue.lower():
                suggestions.append("Verify color contrast meets WCAG AA standards")
            elif "structure" in issue.lower():
                suggestions.append("Improve semantic HTML structure")

        return suggestions

    def _calculate_confidence(
        self, card: GeneratedCard, dimensions: dict[str, QualityDimension]
    ) -> float:
        """Calculate confidence in the quality assessment."""
        # Base confidence on assessment consistency and card complexity
        issue_count = sum(len(dim.issues) for dim in dimensions.values())

        # More issues = higher confidence (easier to identify problems)
        base_confidence = min(0.9, 0.6 + (issue_count * 0.1))

        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        # Adjust for card complexity
        question_length = len(question)
        if question_length < 20:
            base_confidence *= 0.9  # Harder to assess very short questions
        elif question_length > 100:
            base_confidence *= 1.1  # Easier to assess detailed questions

        return min(1.0, max(0.5, base_confidence))

    def _identify_content_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify content-related strengths."""
        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        strengths = []

        if len(question) > 20:
            strengths.append("Good question length with sufficient context")

        if answer and len(answer) > 20:
            strengths.append("Comprehensive answer provided")

        tech_terms = ["function", "class", "method", "api", "algorithm"]
        if any(term in question.lower() for term in tech_terms):
            strengths.append("Uses appropriate technical terminology")

        return strengths

    def _identify_learning_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify learning science-related strengths."""
        # Extract Q&A from APF HTML
        question, answer = _extract_qa_from_apf(card.apf_html)

        # Extract card type
        card_type = "Simple"
        if "{{c" in card.apf_html:
            card_type = "Missing"

        strengths = []

        # Check for active recall indicators
        active_indicators = ["what", "how", "why", "explain", "describe"]
        if any(indicator in question.lower() for indicator in active_indicators):
            strengths.append("Uses active recall question format")

        # Check for appropriate length
        if 15 <= len(question.split()) <= 25:
            strengths.append("Optimal question length for focused recall")

        # Check cloze usage
        if card_type == "Missing":
            cloze_count = len(re.findall(r"\{\{c\d+::[^}]+\}\}", question))
            if 1 <= cloze_count <= 3:
                strengths.append("Appropriate number of independent clozes")

        return strengths

    def _identify_technical_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify technical quality strengths."""
        strengths = []

        # Note: GeneratedCard doesn't have tags attribute - tags are in APF HTML/manifest
        # Tag check skipped for GeneratedCard model

        if re.match(r"^[a-z0-9-]+$", card.slug):
            strengths.append("Proper slug format")

        code_blocks = re.findall(
            r"<pre><code[^>]*>(.*?)</code></pre>",
            card.apf_html,
            re.DOTALL,
        )
        if code_blocks and all(
            'class="language-' in str(block) for block in code_blocks
        ):
            strengths.append("Well-formatted code blocks with language specification")

        return strengths

    def _identify_accessibility_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify accessibility-related strengths."""
        strengths = []

        img_pattern = r'<img[^>]+alt="[^"]*"[^>]*>'
        if re.search(img_pattern, card.apf_html):
            strengths.append("Images include alt text")

        if card.apf_html.count("<p>") <= 5:
            strengths.append("Clean, well-structured HTML")

        return strengths
