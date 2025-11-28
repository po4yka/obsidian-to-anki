"""ReAct Repair Agent for content diagnosis and fixing.

Specialized ReAct agent for diagnosing issues and applying fixes
with transparent reasoning chains.
"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from ...utils.logging import get_logger
from .base import LangChainAgentResult
from .react_agent import ReActAgent

logger = get_logger(__name__)


class ReActRepairAgent:
    """ReAct Repair Agent for content diagnosis and fixing.

    Uses ReAct methodology to diagnose problems, understand root causes,
    and apply appropriate fixes with transparent reasoning.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 7,  # Allow more iterations for repair tasks
    ):
        """Initialize ReAct repair agent.

        Args:
            model: Language model
            tools: Custom tools (will use repair-focused tools)
            max_iterations: Maximum reasoning iterations
        """
        if tools is None:
            from .tools import get_tools_for_agent

            # Use validation tools for repair (APF validator, HTML formatter, etc.)
            tools = get_tools_for_agent("validator")

        self.agent = ReActAgent(
            model=model,
            tools=tools,
            agent_type="repair",
            temperature=0.0,  # Precise repairs
            max_iterations=max_iterations,
        )

        logger.info("react_repair_agent_initialized")

    async def diagnose_and_repair(
        self,
        content: str,
        error_context: dict[str, Any],
        repair_type: str = "content",  # "content", "structure", "format"
    ) -> dict[str, Any]:
        """Diagnose issues and suggest repairs using ReAct reasoning.

        Args:
            content: Content to diagnose and repair
            error_context: Context about the error/issue
            repair_type: Type of repair needed

        Returns:
            Repair result with diagnosis and suggestions
        """
        logger.info("react_repair_diagnosis_start", repair_type=repair_type)

        input_data = {
            "task": "diagnose_issue",
            "content": content,
            "error_message": error_context.get("error_message", ""),
            "context": error_context.get("context", ""),
            "repair_type": repair_type,
            "processing_stage": error_context.get("processing_stage", ""),
        }

        result = await self.agent.run(input_data)
        return self._process_repair_result(result, content)

    async def analyze_and_fix(
        self,
        content: str,
        analysis_type: str = "quality",  # "quality", "structure", "completeness"
    ) -> dict[str, Any]:
        """Analyze content and suggest improvements using ReAct reasoning.

        Args:
            content: Content to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis result with findings and suggestions
        """
        logger.info("react_repair_analysis_start", analysis_type=analysis_type)

        input_data = {
            "task": "analyze_pattern",
            "content": content,
            "pattern_type": analysis_type,
            "analysis_focus": f"Content {analysis_type} analysis and improvement suggestions",
        }

        result = await self.agent.run(input_data)
        return self._process_analysis_result(result, content)

    def _process_repair_result(
        self, agent_result: LangChainAgentResult, original_content: str
    ) -> dict[str, Any]:
        """Process ReAct result for repair tasks.

        Args:
            agent_result: Result from ReAct agent
            original_content: Original content being repaired

        Returns:
            Processed repair result
        """
        if not agent_result.success:
            return {
                "success": False,
                "diagnosis": "Failed to diagnose issue",
                "root_cause": agent_result.reasoning,
                "suggested_fixes": [],
                "confidence": 0.0,
                "reasoning_chain": [],
            }

        output = agent_result.reasoning
        reasoning_chain = agent_result.data.get("reasoning_chain", [])

        # Extract diagnosis components
        diagnosis = self._extract_diagnosis(output)
        root_cause = self._extract_root_cause(output, reasoning_chain)
        suggested_fixes = self._extract_fixes(output, reasoning_chain)

        # Determine if repair was successful
        repair_successful = len(suggested_fixes) > 0

        logger.info(
            "react_repair_completed",
            diagnosis=diagnosis,
            fixes_count=len(suggested_fixes),
            reasoning_steps=len(reasoning_chain),
            confidence=agent_result.confidence,
        )

        return {
            "success": repair_successful,
            "diagnosis": diagnosis,
            "root_cause": root_cause,
            "suggested_fixes": suggested_fixes,
            "confidence": agent_result.confidence,
            "reasoning_chain": reasoning_chain,
            "original_content_length": len(original_content),
        }

    def _process_analysis_result(
        self, agent_result: LangChainAgentResult, original_content: str
    ) -> dict[str, Any]:
        """Process ReAct result for analysis tasks.

        Args:
            agent_result: Result from ReAct agent
            original_content: Original content being analyzed

        Returns:
            Processed analysis result
        """
        if not agent_result.success:
            return {
                "success": False,
                "analysis": "Analysis failed",
                "findings": [],
                "recommendations": [],
                "confidence": 0.0,
                "reasoning_chain": [],
            }

        output = agent_result.reasoning
        reasoning_chain = agent_result.data.get("reasoning_chain", [])

        # Extract analysis components
        analysis = self._extract_analysis_summary(output)
        findings = self._extract_findings(output, reasoning_chain)
        recommendations = self._extract_recommendations(output, reasoning_chain)

        logger.info(
            "react_analysis_completed",
            findings_count=len(findings),
            recommendations_count=len(recommendations),
            reasoning_steps=len(reasoning_chain),
            confidence=agent_result.confidence,
        )

        return {
            "success": True,
            "analysis": analysis,
            "findings": findings,
            "recommendations": recommendations,
            "confidence": agent_result.confidence,
            "reasoning_chain": reasoning_chain,
            "original_content_length": len(original_content),
        }

    def _extract_diagnosis(self, output: str) -> str:
        """Extract diagnosis from agent output.

        Args:
            output: Agent output

        Returns:
            Diagnosis summary
        """
        # Look for diagnosis patterns
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower() for word in ["diagnosis:", "issue:", "problem:"]
            ):
                return line.strip()

        # Fallback: first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith("Thought:"):
                return line

        return "Diagnosis completed"

    def _extract_root_cause(self, output: str, reasoning_chain: list[dict]) -> str:
        """Extract root cause analysis.

        Args:
            output: Agent output
            reasoning_chain: Reasoning steps

        Returns:
            Root cause description
        """
        # Look for root cause in output
        lines = output.split("\n")
        for line in lines:
            if any(
                phrase in line.lower()
                for phrase in ["root cause:", "cause:", "reason:"]
            ):
                return line.strip()

        # Look in reasoning chain observations
        for step in reasoning_chain:
            observation = step.get("observation", "")
            if any(word in observation.lower() for word in ["cause", "root", "reason"]):
                return observation.strip()

        return "Root cause analysis completed"

    def _extract_fixes(self, output: str, reasoning_chain: list[dict]) -> list[str]:
        """Extract suggested fixes.

        Args:
            output: Agent output
            reasoning_chain: Reasoning steps

        Returns:
            List of suggested fixes
        """
        fixes = []

        # Extract from output
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower()
                for word in ["fix:", "repair:", "solution:", "suggest:"]
            ):
                fixes.append(line.strip())

        # Extract from reasoning observations
        for step in reasoning_chain:
            observation = step.get("observation", "")
            if any(
                word in observation.lower()
                for word in ["fix", "repair", "correct", "update"]
            ):
                fixes.append(observation.strip())

        return list(set(fixes))  # Remove duplicates

    def _extract_analysis_summary(self, output: str) -> str:
        """Extract analysis summary.

        Args:
            output: Agent output

        Returns:
            Analysis summary
        """
        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower() for word in ["analysis:", "summary:", "overview:"]
            ):
                return line.strip()

        # First substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 15 and not line.startswith("Thought:"):
                return line

        return "Analysis completed"

    def _extract_findings(self, output: str, reasoning_chain: list[dict]) -> list[str]:
        """Extract findings from analysis.

        Args:
            output: Agent output
            reasoning_chain: Reasoning steps

        Returns:
            List of findings
        """
        findings = []

        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower() for word in ["finding:", "found:", "identified:"]
            ):
                findings.append(line.strip())

        # From reasoning chain
        for step in reasoning_chain:
            observation = step.get("observation", "")
            if any(
                word in observation.lower()
                for word in ["found", "identified", "discovered"]
            ):
                findings.append(observation.strip())

        return list(set(findings))

    def _extract_recommendations(
        self, output: str, reasoning_chain: list[dict]
    ) -> list[str]:
        """Extract recommendations from analysis.

        Args:
            output: Agent output
            reasoning_chain: Reasoning steps

        Returns:
            List of recommendations
        """
        recommendations = []

        lines = output.split("\n")
        for line in lines:
            if any(
                word in line.lower()
                for word in ["recommend:", "suggest:", "improve:", "should:"]
            ):
                recommendations.append(line.strip())

        # From reasoning chain
        for step in reasoning_chain:
            observation = step.get("observation", "")
            if any(
                word in observation.lower()
                for word in ["recommend", "suggest", "improve"]
            ):
                recommendations.append(observation.strip())

        return list(set(recommendations))
