"""ReAct Validator Agent for transparent validation.

Specialized ReAct agent for validation tasks with transparent reasoning chains.
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from ...models import GeneratedCard, PostValidationResult, PreValidationResult
from ...utils.logging import get_logger
from .base import LangChainAgentResult
from .react_agent import ReActAgent

logger = get_logger(__name__)


class ReActValidatorAgent:
    """ReAct Validator Agent for transparent content validation.

    Uses ReAct methodology to provide clear reasoning chains during validation,
    making it easier to understand why certain decisions were made.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        validator_type: str = "pre",  # "pre" or "post"
        max_iterations: int = 5,
    ):
        """Initialize ReAct validator agent.

        Args:
            model: Language model
            tools: Custom tools (will use defaults if None)
            validator_type: Type of validation ("pre" or "post")
            max_iterations: Maximum reasoning iterations
        """
        if tools is None:
            from .tools import get_tools_for_agent

            agent_type = f"{validator_type}_validator"
            tools = get_tools_for_agent(agent_type)

        self.validator_type = validator_type

        self.agent = ReActAgent(
            model=model,
            tools=tools,
            agent_type=f"validator_{validator_type}",
            temperature=0.0,  # Strict validation
            max_iterations=max_iterations,
        )

        logger.info(f"react_{validator_type}_validator_initialized")

    async def validate_pre(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
    ) -> PreValidationResult:
        """Run pre-validation with ReAct reasoning.

        Args:
            note_content: Note content to validate
            metadata: Note metadata
            qa_pairs: Q/A pairs

        Returns:
            Pre-validation result with reasoning
        """
        logger.info("react_pre_validation_start")

        input_data = {
            "task": "validate_content",
            "content": note_content,
            "metadata": metadata,
            "qa_pairs": qa_pairs,
            "validation_type": "structure and content quality",
        }

        result = await self.agent.run(input_data)
        return self._process_pre_validation_result(result)

    async def validate_post(
        self,
        cards: List[GeneratedCard],
        metadata: Dict[str, Any],
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Run post-validation with ReAct reasoning.

        Args:
            cards: Generated cards to validate
            metadata: Note metadata
            strict_mode: Use strict validation rules

        Returns:
            Post-validation result with reasoning
        """
        logger.info("react_post_validation_start", card_count=len(cards))

        # Convert cards to content for validation
        apf_content = self._cards_to_apf(cards)

        input_data = {
            "task": "validate_content",
            "content": apf_content,
            "metadata": metadata,
            "validation_type": f"APF card quality{' (strict mode)' if strict_mode else ''}",
            "strict_mode": strict_mode,
        }

        result = await self.agent.run(input_data)
        return self._process_post_validation_result(result, cards)

    def _cards_to_apf(self, cards: List[GeneratedCard]) -> str:
        """Convert cards to APF format for validation.

        Args:
            cards: List of generated cards

        Returns:
            APF formatted content
        """
        apf_lines = [
            "<!-- PROMPT_VERSION: apf-v2.1 -->",
            "<!-- BEGIN_CARDS -->",
            "",
        ]

        for i, card in enumerate(cards, 1):
            tags_str = " ".join(card.tags)
            apf_lines.extend(
                [
                    f"<!-- Card {i} | slug: {card.slug} | CardType: {card.card_type} | Tags: {tags_str} -->",
                    "",
                    "<!-- Title -->",
                    card.front,
                    "",
                    "<!-- Key point -->",
                    card.back,
                    "",
                    f'<!-- manifest: {{"slug":"{card.slug}","lang":"en","type":"{card.card_type}","tags":{card.tags}}} -->',
                    "",
                ]
            )

        apf_lines.extend(
            [
                "<!-- END_CARDS -->",
                "END_OF_CARDS",
            ]
        )

        return "\n".join(apf_lines)

    def _process_pre_validation_result(
        self, agent_result: LangChainAgentResult
    ) -> PreValidationResult:
        """Process ReAct result into PreValidationResult.

        Args:
            agent_result: Result from ReAct agent

        Returns:
            Processed PreValidationResult
        """
        if not agent_result.success:
            return PreValidationResult(
                is_valid=False,
                error_type="validation_failed",
                error_details=agent_result.reasoning,
                suggested_fixes=["Retry validation"],
                confidence=0.0,
            )

        output = agent_result.reasoning

        # Analyze reasoning chain for validation results
        reasoning_data = agent_result.data.get("reasoning_chain", [])
        validation_steps = len(reasoning_data)

        # Determine validity based on output
        is_valid = True
        if any(
            phrase in output.lower()
            for phrase in [
                "validation failed",
                "content invalid",
                "structure issues",
                "critical problems",
                "cannot proceed",
            ]
        ):
            is_valid = False

        # Extract error details
        error_details = ""
        if not is_valid:
            # Try to extract specific issues from reasoning
            error_lines = []
            for step in reasoning_data:
                observation = step.get("observation", "")
                if any(
                    word in observation.lower()
                    for word in ["error", "issue", "problem", "invalid"]
                ):
                    error_lines.append(observation)

            error_details = " ".join(error_lines) if error_lines else output

        # Extract suggested fixes
        suggested_fixes = []
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower()
                for word in ["suggest", "fix", "recommend", "should"]
            ):
                suggested_fixes.append(line.strip())

        # Adjust confidence based on reasoning depth
        confidence = agent_result.confidence
        if validation_steps > 3:  # Thorough analysis
            confidence = min(confidence + 0.1, 1.0)

        logger.info(
            "react_pre_validation_completed",
            is_valid=is_valid,
            reasoning_steps=validation_steps,
            confidence=confidence,
        )

        return PreValidationResult(
            is_valid=is_valid,
            error_type="content_issue" if not is_valid else "none",
            error_details=error_details,
            suggested_fixes=suggested_fixes,
            confidence=confidence,
        )

    def _process_post_validation_result(
        self, agent_result: LangChainAgentResult, original_cards: List[GeneratedCard]
    ) -> PostValidationResult:
        """Process ReAct result into PostValidationResult.

        Args:
            agent_result: Result from ReAct agent
            original_cards: Original cards being validated

        Returns:
            Processed PostValidationResult
        """
        if not agent_result.success:
            return PostValidationResult(
                is_valid=False,
                error_details=agent_result.reasoning,
                warnings=agent_result.warnings,
                auto_fix_suggestions=[],
                confidence=0.0,
            )

        output = agent_result.reasoning
        reasoning_data = agent_result.data.get("reasoning_chain", [])

        # Analyze validation thoroughness
        validation_depth = len(reasoning_data)

        # Determine validity
        is_valid = True
        if any(
            phrase in output.lower()
            for phrase in [
                "validation failed",
                "cards invalid",
                "critical issues",
                "cannot accept",
                "major problems",
            ]
        ):
            is_valid = False

        # Extract warnings and suggestions
        warnings = agent_result.warnings.copy()
        auto_fix_suggestions = []

        # Parse reasoning chain for specific issues
        for step in reasoning_data:
            observation = step.get("observation", "")
            if "warning" in observation.lower():
                warnings.append(observation)
            if any(
                word in observation.lower()
                for word in ["fix", "correct", "update", "change"]
            ):
                auto_fix_suggestions.append(observation)

        # Extract suggestions from final output
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower()
                for word in ["suggest", "fix", "recommend", "improve"]
            ):
                auto_fix_suggestions.append(line.strip())

        # Adjust confidence based on validation thoroughness
        confidence = agent_result.confidence
        if validation_depth >= 3:  # Comprehensive validation
            confidence = min(confidence + 0.15, 1.0)

        logger.info(
            "react_post_validation_completed",
            is_valid=is_valid,
            validation_depth=validation_depth,
            warnings_count=len(warnings),
            suggestions_count=len(auto_fix_suggestions),
            confidence=confidence,
        )

        return PostValidationResult(
            is_valid=is_valid,
            error_details="" if is_valid else output,
            warnings=warnings,
            auto_fix_suggestions=list(set(auto_fix_suggestions)),  # Remove duplicates
            confidence=confidence,
        )
