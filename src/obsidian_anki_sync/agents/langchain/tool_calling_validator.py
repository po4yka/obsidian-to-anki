"""Tool Calling Validator Agent for content validation.

Specialized tool calling agent for validating APF cards and content
with parallel validation tools.
"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from obsidian_anki_sync.models import (
    GeneratedCard,
    PostValidationResult,
    PreValidationResult,
)
from obsidian_anki_sync.utils.logging import get_logger

from .base import LangChainAgentResult
from .tool_calling_agent import ToolCallingAgent

logger = get_logger(__name__)


class ToolCallingValidatorAgent:
    """Tool Calling Validator Agent for content validation.

    Uses tool calling agent with specialized validation tools for
    comprehensive content checking with parallel execution.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
        enable_parallel_tools: bool = True,
        validator_type: str = "post",  # "pre" or "post"
    ):
        """Initialize tool calling validator agent.

        Args:
            model: Language model supporting tool calling
            tools: Custom tools (will use defaults if None)
            enable_parallel_tools: Enable parallel tool execution
            validator_type: Type of validation ("pre" or "post")
        """
        if tools is None:
            from .tools import get_tools_for_agent

            agent_type = f"{validator_type}_validator"
            tools = get_tools_for_agent(agent_type)

        self.validator_type = validator_type

        self.agent = ToolCallingAgent(
            model=model,
            tools=tools,
            agent_type=f"{validator_type}_validator",
            temperature=0.0,  # Strict validation
            enable_parallel_tool_calls=enable_parallel_tools,
        )

        logger.info(f"tool_calling_{validator_type}_validator_initialized")

    async def validate_pre(
        self,
        note_content: str,
        metadata: dict[str, Any],
        qa_pairs: list[dict[str, Any]],
    ) -> PreValidationResult:
        """Run pre-validation using tool calling agent.

        Args:
            note_content: Note content to validate
            metadata: Note metadata
            qa_pairs: Q/A pairs

        Returns:
            Pre-validation result
        """
        logger.info("tool_calling_pre_validation_start")

        input_data = {
            "task": "validate_content",
            "content": note_content,
            "metadata": metadata,
            "qa_pairs": qa_pairs,
            "validation_type": "pre_validation",
        }

        result = await self.agent.run(input_data)
        return self._process_pre_validation_result(result)

    async def validate_post(
        self,
        cards: list[GeneratedCard],
        metadata: dict[str, Any],
        strict_mode: bool = True,
    ) -> PostValidationResult:
        """Run post-validation using tool calling agent.

        Args:
            cards: Generated cards to validate
            metadata: Note metadata
            strict_mode: Use strict validation

        Returns:
            Post-validation result
        """
        logger.info("tool_calling_post_validation_start", card_count=len(cards))

        # Convert cards to APF format for validation
        apf_content = self._cards_to_apf(cards)

        input_data = {
            "task": "validate_content",
            "content": apf_content,
            "metadata": metadata,
            "validation_type": "post_validation",
            "strict_mode": strict_mode,
        }

        result = await self.agent.run(input_data)
        return self._process_post_validation_result(result, cards)

    def _cards_to_apf(self, cards: list[GeneratedCard]) -> str:
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
        """Process agent result into PreValidationResult.

        Args:
            agent_result: Result from tool calling agent

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

        # Parse validation results from agent output
        output = agent_result.reasoning

        is_valid = "validation passed" in output.lower() or "valid" in output.lower()
        if "failed" in output.lower() or "error" in output.lower():
            is_valid = False

        error_type = "none" if is_valid else "content_issue"
        error_details = output if not is_valid else ""

        # Extract suggested fixes
        suggested_fixes = []
        if "suggest" in output.lower():
            # Simple extraction - could be more sophisticated
            lines = output.split("\n")
            for line in lines:
                if any(
                    word in line.lower() for word in ["suggest", "fix", "recommend"]
                ):
                    suggested_fixes.append(line.strip())

        return PreValidationResult(
            is_valid=is_valid,
            error_type=error_type,
            error_details=error_details,
            suggested_fixes=suggested_fixes,
            confidence=agent_result.confidence,
        )

    def _process_post_validation_result(
        self, agent_result: LangChainAgentResult, original_cards: list[GeneratedCard]
    ) -> PostValidationResult:
        """Process agent result into PostValidationResult.

        Args:
            agent_result: Result from tool calling agent
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

        # Determine validity
        is_valid = True
        if any(
            word in output.lower()
            for word in ["failed", "error", "invalid", "incorrect"]
        ):
            is_valid = False

        # Extract warnings
        warnings = agent_result.warnings.copy()

        # Extract auto-fix suggestions
        auto_fix_suggestions = []
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower()
                for word in ["fix:", "correct:", "change:", "update:"]
            ):
                auto_fix_suggestions.append(line.strip())

        logger.info(
            "tool_calling_validation_completed",
            validator_type=self.validator_type,
            is_valid=is_valid,
            warnings_count=len(warnings),
            suggestions_count=len(auto_fix_suggestions),
        )

        return PostValidationResult(
            is_valid=is_valid,
            error_details="" if is_valid else output,
            warnings=warnings,
            auto_fix_suggestions=auto_fix_suggestions,
            confidence=agent_result.confidence,
        )
