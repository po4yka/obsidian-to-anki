"""JSON Chat Agent implementation using LangChain.

This module provides a JSON Chat Agent optimized for structured data
processing and JSON-based operations.
"""

from typing import Any

from langchain.agents import create_json_chat_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseLangChainAgent, LangChainAgentResult

logger = get_logger(__name__)


class JSONChatAgent(BaseLangChainAgent):
    """JSON Chat Agent for structured data processing.

    Optimized for JSON processing, structured data extraction,
    and metadata operations with clear JSON-based communication.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool],
        agent_type: str = "json_chat",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        """Initialize JSON Chat Agent.

        Args:
            model: LangChain language model (preferably chat model)
            tools: List of tools for the agent
            agent_type: Type of agent
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        """
        # Set default system prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_type=agent_type,
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for JSON Chat Agent.

        Returns:
            System prompt string
        """
        return """You are a JSON processing specialist.

Your task is to handle structured data, JSON operations, and metadata processing.
You excel at parsing, validating, and manipulating structured data formats.

Capabilities:
- Parse and validate JSON structures
- Extract metadata from various formats
- Process structured data efficiently
- Generate JSON outputs when needed

Always work with structured data and provide clear, parseable results."""

    def _create_agent(self) -> Any:
        """Create the underlying LangChain JSON Chat Agent.

        Returns:
            LangChain JSONChatAgent instance
        """
        # Create JSON chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Create the agent
        agent = create_json_chat_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt,
        )

        return agent

    async def run(
        self, input_data: dict[str, Any], **kwargs: Any
    ) -> LangChainAgentResult:
        """Run the JSON Chat Agent.

        Args:
            input_data: Input data
            **kwargs: Additional arguments

        Returns:
            Agent execution result
        """
        try:
            from langchain.agents import AgentExecutor

            # Create agent executor
            executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=kwargs.get("verbose", False),
                max_iterations=kwargs.get("max_iterations", 3),
                max_execution_time=kwargs.get("max_execution_time"),
                handle_parsing_errors=kwargs.get("handle_parsing_errors", True),
                return_intermediate_steps=kwargs.get(
                    "return_intermediate_steps", False
                ),
            )

            # Prepare input
            input_text = input_data.get("input", "")
            if not input_text and "task" in input_data:
                input_text = self._format_json_input(input_data)

            # Run the agent
            logger.info(
                "json_chat_agent_executing",
                agent_type=self.agent_type,
                input_length=len(input_text),
            )

            result = await executor.ainvoke({"input": input_text})

            # Process result
            return self._process_result(result)

        except Exception as e:
            logger.error(
                "json_chat_agent_execution_failed",
                agent_type=self.agent_type,
                error=str(e),
                error_type=type(e).__name__,
            )

            return LangChainAgentResult(
                success=False,
                reasoning=f"JSON Chat agent execution failed: {e}",
                warnings=["JSON Chat agent error"],
                confidence=0.0,
            )

    def _format_json_input(self, input_data: dict[str, Any]) -> str:
        """Format input data for JSON processing.

        Args:
            input_data: Input data

        Returns:
            Formatted input text
        """
        task = input_data.get("task", "")

        if task == "parse_metadata":
            content = input_data.get("content", "")

            return f"""Parse metadata from the following content and return as structured JSON:

Content: {content[:1000]}{"..." if len(content) > 1000 else ""}

Extract all metadata fields and return as a valid JSON object."""

        elif task == "validate_structure":
            data = input_data.get("data", "")

            return f"""Validate the structure of the following data and return validation results as JSON:

Data: {data}

Return a JSON object with validation results, including any errors found."""

        else:
            return f"Process JSON task: {task}. Data: {input_data}"

    def _process_result(self, raw_result: dict[str, Any]) -> LangChainAgentResult:
        """Process raw agent result.

        Args:
            raw_result: Raw result from AgentExecutor

        Returns:
            Processed result
        """
        output = raw_result.get("output", "")
        intermediate_steps = raw_result.get("intermediate_steps", [])

        # Extract confidence and warnings
        confidence = self._extract_confidence(output)
        warnings = self._extract_warnings(output)

        # JSON agents typically produce structured output
        success = True
        try:
            # Try to parse output as JSON to validate
            import json

            json.loads(output)
            confidence = min(confidence + 0.2, 1.0)  # Bonus for valid JSON
        except json.JSONDecodeError:
            if "json" in output.lower() and (
                "error" in output.lower() or "invalid" in output.lower()
            ):
                success = False

        logger.info(
            "json_chat_agent_completed",
            success=success,
            output_length=len(output),
            confidence=confidence,
        )

        return LangChainAgentResult(
            success=success,
            reasoning=output,
            data={
                "raw_output": output,
                "intermediate_steps": intermediate_steps,
            },
            warnings=warnings,
            confidence=confidence,
            metadata={
                "agent_type": self.agent_type,
                "structured_output": True,
                "pattern": "json_chat",
            },
        )
