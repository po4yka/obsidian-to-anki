"""Enhanced card quality assessment agent with multi-dimensional evaluation."""

import re
import time

from ..models import NoteMetadata
from .models import GeneratedCard
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .models import QualityReport, QualityDimension

logger = get_logger(__name__)


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
        context_cards: list[GeneratedCard | None] = None,
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

        logger.debug("assessing_card_quality",
                     slug=card.slug, card_type=card.card_type)

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
        context_cards: list[GeneratedCard | None] = None,
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

        # Check for placeholder content
        if any(placeholder in card.question.lower() for placeholder in ["tbd", "todo", "fill in"]):
            issues.append("Contains placeholder content")
            score -= 0.3

        # Check for incomplete code examples
        if "..." in card.question or "..." in (card.answer or ""):
            issues.append("Contains incomplete code examples")
            score -= 0.2

        # Check question clarity
        if len(card.question.strip()) < 10:
            issues.append("Question too short or unclear")
            score -= 0.2

        # Check for proper capitalization and punctuation
        question = card.question.strip()
        if not question[0].isupper() or not question.endswith(('?', '.', '!')):
            issues.append(
                "Question lacks proper capitalization or punctuation")
            score -= 0.1

        # Check answer quality
        if card.answer:
            answer = card.answer.strip()
            if len(answer) < 5:
                issues.append("Answer too brief or incomplete")
                score -= 0.2

        # Verify content matches programming domain
        tech_indicators = ["function", "class",
                           "method", "api", "code", "programming"]
        has_tech_content = any(indicator in card.question.lower()
                               for indicator in tech_indicators)
        if metadata.tags and "programming" in " ".join(metadata.tags) and not has_tech_content:
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

        # Atomic principle: One concept per card
        question_words = len(card.question.split())
        if question_words > 25:
            issues.append("Question too long - may test multiple concepts")
            score -= 0.2

        # Active recall: Question should require memory retrieval
        passive_indicators = ["is", "are", "does",
                              "do", "can", "should", "would"]
        question_lower = card.question.lower()
        passive_count = sum(
            1 for word in passive_indicators if word in question_lower)
        if passive_count > 2:
            issues.append(
                "Question may be passive recognition rather than active recall")
            score -= 0.15

        # Cognitive load: Answer shouldn't be too long
        if card.answer and len(card.answer) > 500:
            issues.append("Answer too long - high cognitive load")
            score -= 0.2

        # Context sufficiency: Question should have enough context
        if len(card.question.split()) < 5:
            issues.append(
                "Question lacks sufficient context for unambiguous recall")
            score -= 0.1

        # Cloze quality (if applicable)
        if card.card_type == "Missing":
            cloze_pattern = r"\{\{c\d+::[^}]+\}\}"
            clozes = re.findall(
                cloze_pattern, card.question + (card.answer or ""))
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

        # Check HTML structure
        if "<pre>" in card.question and "</pre>" not in card.question:
            issues.append("Unclosed <pre> tag in question")
            score -= 0.2

        if card.answer and "<pre>" in card.answer and "</pre>" not in card.answer:
            issues.append("Unclosed <pre> tag in answer")
            score -= 0.2

        # Check code block language specification
        code_blocks = re.findall(
            r'<pre><code[^>]*>(.*?)</code></pre>', card.question + (card.answer or ""), re.DOTALL)
        for block in code_blocks:
            if not re.search(r'class="language-[^"]*"', block):
                issues.append("Code block missing language class")
                score -= 0.1

        # Check for proper slug format
        if not re.match(r'^[a-z0-9-]+$', card.slug or ""):
            issues.append(
                "Invalid slug format (should be lowercase with hyphens only)")
            score -= 0.1

        # Check tag quality
        if not card.tags or len(card.tags) < 3:
            issues.append("Insufficient tags (minimum 3 required)")
            score -= 0.2

        # Check for duplicate tags
        if card.tags and len(card.tags) != len(set(card.tags)):
            issues.append("Duplicate tags found")
            score -= 0.1

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
        img_pattern = r'<img[^>]+>'
        images = re.findall(img_pattern, card.question + (card.answer or ""))
        for img in images:
            if 'alt=' not in img:
                issues.append("Image missing alt text for screen readers")
                score -= 0.2

        # Check for sufficient color contrast (basic check)
        if 'color:' in card.question.lower() or 'color:' in (card.answer or "").lower():
            # If colors are specified, they should meet contrast requirements
            # This is a basic check - full WCAG compliance would need more analysis
            issues.append(
                "Color usage detected - verify contrast ratios meet WCAG standards")
            score -= 0.1

        # Check for semantic HTML structure
        if card.question.count('<p>') > 10:
            issues.append(
                "Excessive paragraph breaks may indicate poor structure")
            score -= 0.1

        score = max(0.0, score)

        return QualityDimension(
            score=score,
            weight=0.1,
            issues=issues,
            strengths=self._identify_accessibility_strengths(card),
        )

    def _calculate_overall_score(self, dimensions: dict[str, QualityDimension]) -> float:
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
            if dimension.score < 0.8:  # Only suggest improvements for low-scoring dimensions
                suggestions.extend(self._get_dimension_improvements(
                    dim_name, dimension.issues))

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)

        return unique_suggestions

    def _get_dimension_improvements(self, dimension: str, issues: list[str]) -> list[str]:
        """Get improvement suggestions for a specific dimension."""
        suggestions = []

        for issue in issues:
            if "placeholder" in issue.lower():
                suggestions.append(
                    "Replace placeholder content with actual code examples")
            elif "incomplete" in issue.lower():
                suggestions.append(
                    "Complete partial code examples with runnable code")
            elif "too short" in issue.lower():
                suggestions.append(
                    "Add more context to make the question unambiguous")
            elif "punctuation" in issue.lower():
                suggestions.append("Add proper punctuation and capitalization")
            elif "too brief" in issue.lower():
                suggestions.append(
                    "Expand answer with more complete explanation")
            elif "doesn't match" in issue.lower():
                suggestions.append(
                    "Ensure content aligns with the specified domain/tags")
            elif "too long" in issue.lower():
                suggestions.append(
                    "Split complex questions into multiple atomic cards")
            elif "passive" in issue.lower():
                suggestions.append(
                    "Rewrite as active recall question (What/How/Why format)")
            elif "cognitive load" in issue.lower():
                suggestions.append(
                    "Break down complex answers into simpler components")
            elif "clozes" in issue.lower():
                suggestions.append(
                    "Review cloze deletions to ensure independent concepts")
            elif "unclosed" in issue.lower():
                suggestions.append("Fix HTML tag structure")
            elif "language class" in issue.lower():
                suggestions.append("Add language specification to code blocks")
            elif "slug format" in issue.lower():
                suggestions.append(
                    "Use lowercase letters, numbers, and hyphens only in slug")
            elif "insufficient tags" in issue.lower():
                suggestions.append("Add at least 3 relevant tags")
            elif "duplicate tags" in issue.lower():
                suggestions.append("Remove duplicate tags")
            elif "alt text" in issue.lower():
                suggestions.append("Add descriptive alt text for all images")
            elif "color" in issue.lower():
                suggestions.append(
                    "Verify color contrast meets WCAG AA standards")
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

        # Adjust for card complexity
        question_length = len(card.question)
        if question_length < 20:
            base_confidence *= 0.9  # Harder to assess very short questions
        elif question_length > 100:
            base_confidence *= 1.1  # Easier to assess detailed questions

        return min(1.0, max(0.5, base_confidence))

    def _identify_content_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify content-related strengths."""
        strengths = []

        if len(card.question) > 20:
            strengths.append("Good question length with sufficient context")

        if card.answer and len(card.answer) > 20:
            strengths.append("Comprehensive answer provided")

        tech_terms = ["function", "class", "method", "api", "algorithm"]
        if any(term in card.question.lower() for term in tech_terms):
            strengths.append("Uses appropriate technical terminology")

        return strengths

    def _identify_learning_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify learning science-related strengths."""
        strengths = []

        # Check for active recall indicators
        active_indicators = ["what", "how", "why", "explain", "describe"]
        if any(indicator in card.question.lower() for indicator in active_indicators):
            strengths.append("Uses active recall question format")

        # Check for appropriate length
        if 15 <= len(card.question.split()) <= 25:
            strengths.append("Optimal question length for focused recall")

        # Check cloze usage
        if card.card_type == "Missing":
            cloze_count = len(re.findall(
                r"\{\{c\d+::[^}]+\}\}", card.question))
            if 1 <= cloze_count <= 3:
                strengths.append("Appropriate number of independent clozes")

        return strengths

    def _identify_technical_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify technical quality strengths."""
        strengths = []

        if card.tags and len(card.tags) >= 3:
            strengths.append("Good tag coverage")

        if re.match(r'^[a-z0-9-]+$', card.slug or ""):
            strengths.append("Proper slug format")

        code_blocks = re.findall(
            r'<pre><code[^>]*>(.*?)</code></pre>', card.question + (card.answer or ""), re.DOTALL)
        if code_blocks and all('class="language-' in block for block in code_blocks):
            strengths.append(
                "Well-formatted code blocks with language specification")

        return strengths

    def _identify_accessibility_strengths(self, card: GeneratedCard) -> list[str]:
        """Identify accessibility-related strengths."""
        strengths = []

        img_pattern = r'<img[^>]+alt="[^"]*"[^>]*>'
        if re.search(img_pattern, card.question + (card.answer or "")):
            strengths.append("Images include alt text")

        if card.question.count('<p>') <= 5:
            strengths.append("Clean, well-structured HTML")

        return strengths
