"""Base classes and interfaces for LangChain-based agents.

This module defines the base interfaces and common functionality for LangChain agents
used in the card generation pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LangChainAgentResult:
    """Result from a LangChain agent execution."""

    success: bool
    reasoning: str
    data: Optional[Any] = None
    warnings: Optional[List[str]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseLangChainAgent(ABC):
    """Base class for LangChain-based agents.

    Provides common functionality and interface for all LangChain agents
    in the card generation pipeline.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        agent_type: str = "base",
    ):
        """Initialize LangChain agent.

        Args:
            model: LangChain language model
            tools: List of tools for the agent
            system_prompt: System prompt for the agent
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            agent_type: Type identifier for the agent
        """
        self.model = model
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.agent_type = agent_type

        # Initialize the underlying agent (to be implemented by subclasses)
        self.agent = self._create_agent()

        logger.info(
            "langchain_agent_initialized",
            agent_type=agent_type,
            model_type=type(model).__name__,
            tool_count=len(self.tools),
        )

    @abstractmethod
    def _create_agent(self) -> Any:
        """Create the underlying LangChain agent.

        Returns:
            LangChain agent instance
        """
        pass

    @abstractmethod
    async def run(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> LangChainAgentResult:
        """Run the agent with given input data.

        Args:
            input_data: Input data for the agent
            **kwargs: Additional arguments

        Returns:
            Agent execution result
        """
        pass

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent.

        Returns:
            Dictionary with agent information
        """
        return {
            "agent_type": self.agent_type,
            "model": str(self.model),
            "tools": [tool.name for tool in self.tools],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _validate_result(self, result: Any) -> LangChainAgentResult:
        """Validate and normalize agent result.

        Args:
            result: Raw result from agent execution

        Returns:
            Normalized LangChainAgentResult
        """
        try:
            if hasattr(result, "success") and hasattr(result, "reasoning"):
                # Already a LangChainAgentResult
                return result
            elif isinstance(result, dict):
                # Dictionary result
                return LangChainAgentResult(
                    success=result.get("success", True),
                    reasoning=result.get("reasoning", ""),
                    data=result.get("data"),
                    warnings=result.get("warnings", []),
                    confidence=result.get("confidence", 1.0),
                    metadata=result.get("metadata", {}),
                )
            else:
                # Fallback for other result types
                return LangChainAgentResult(
                    success=True,
                    reasoning=str(result),
                    data=result,
                )
        except Exception as e:
            logger.warning("result_validation_failed", error=str(e))
            return LangChainAgentResult(
                success=False,
                reasoning=f"Result validation failed: {e}",
                data=result,
                warnings=["Result validation error"],
            )

    def _extract_confidence(self, output: str) -> float:
        """Extract confidence score from agent output.

        Args:
            output: Agent output text

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Look for explicit confidence indicators
        confidence_indicators = {
            "high confidence": 0.9,
            "medium confidence": 0.7,
            "low confidence": 0.4,
            "very confident": 0.95,
            "somewhat confident": 0.6,
            "not confident": 0.3,
        }

        output_lower = output.lower()
        for indicator, score in confidence_indicators.items():
            if indicator in output_lower:
                return score

        # Look for percentage patterns
        import re
        percentage_match = re.search(r"(\d+)%", output)
        if percentage_match:
            percentage = int(percentage_match.group(1)) / 100.0
            return min(max(percentage, 0.0), 1.0)

        # Default confidence
        return 0.8

    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warnings from agent output.

        Args:
            output: Agent output text

        Returns:
            List of warning messages
        """
        warnings = []
        warning_patterns = [
            "warning:",
            "caution:",
            "note:",
            "important:",
            "be careful",
            "potential issue",
        ]

        lines = output.split("\n")
        for line in lines:
            line_lower = line.lower()
            for pattern in warning_patterns:
                if pattern in line_lower:
                    warnings.append(line.strip())
                    break

        return warnings
