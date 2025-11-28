"""Structured Chat Agent implementation using LangChain.

This module provides a Structured Chat Agent optimized for multi-input
tool scenarios and complex structured operations.
"""

from typing import Any

from langchain.agents import create_structured_chat_agent
from langchain.agents.structured_chat.base import StructuredChatAgent as LangChainStructuredChatAgent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from ...utils.logging import get_logger
from .base import BaseLangChainAgent, LangChainAgentResult

logger = get_logger(__name__)


class StructuredChatAgent(BaseLangChainAgent):
    """Structured Chat Agent for multi-input tool scenarios.

    Optimized for complex operations requiring multiple structured inputs
    and tools. Provides clear input/output formatting for complex tasks.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool],
        agent_type: str = "structured_chat",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        """Initialize Structured Chat Agent.

        Args:
            model: LangChain language model (preferably chat model)
            tools: List of tools for the agent
            agent_type: Type of agent
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        """
        # Set default system prompts based on agent type
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt(agent_type)

        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            agent_type=agent_type,
        )

    def _get_default_system_prompt(self, agent_type: str) -> str:
        """Get default system prompt for agent type.

        Args:
            agent_type: Type of agent

        Returns:
            System prompt string
        """
        prompts = {
            "enrichment": """You are a content enrichment specialist.

Your task is to enhance and enrich content using multiple tools and inputs.
You have access to various enrichment tools and can process complex multi-part content.

Focus on:
- Adding valuable learning enhancements
- Improving content structure and clarity
- Incorporating additional context and examples
- Ensuring content completeness and effectiveness

Use your tools systematically to create enriched, comprehensive content.""",
            "complex_generator": """You are an advanced content generation specialist.

Your role is to generate complex content that requires multiple inputs and tools.
You can handle structured data, multiple content sources, and sophisticated generation tasks.

Capabilities:
- Process multiple content inputs simultaneously
- Use various generation and formatting tools
- Create structured, high-quality output
- Handle complex multi-step generation workflows

Generate content that meets the highest quality standards.""",
            "analyzer": """You are a comprehensive content analyzer.

Your task is to analyze complex content using multiple tools and perspectives.
You can process various types of content and provide detailed, multi-faceted analysis.

Analysis capabilities:
- Structural analysis across multiple dimensions
- Quality assessment using multiple criteria
- Comparative analysis of different content elements
- Pattern recognition and trend identification

Provide thorough, well-structured analysis results.""",
        }

        return prompts.get(
            agent_type,
            """You are a structured chat agent with access to multiple tools.

You excel at handling complex tasks that require:
- Multiple inputs and data sources
- Coordinated use of various tools
- Structured processing and output
- Systematic problem-solving approaches

Use your tools effectively to accomplish complex objectives.""",
        )

    def _create_agent(self) -> Any:
        """Create the underlying LangChain Structured Chat Agent.

        Returns:
            LangChain StructuredChatAgent instance
        """
        # Create structured chat prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the agent
        agent = create_structured_chat_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt,
        )

        return agent

    async def run(self, input_data: dict[str, Any], **kwargs: Any) -> LangChainAgentResult:
        """Run the Structured Chat Agent.

        Args:
            input_data: Input data with structured information
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
                max_iterations=kwargs.get("max_iterations", 5),
                max_execution_time=kwargs.get("max_execution_time"),
                handle_parsing_errors=kwargs.get(
                    "handle_parsing_errors", True),
                return_intermediate_steps=kwargs.get(
                    "return_intermediate_steps", False
                ),
            )

            # Prepare structured input
            input_text = input_data.get("input", "")
            if not input_text and "task" in input_data:
                input_text = self._format_structured_input(input_data)

            # Run the agent
            logger.info(
                "structured_chat_agent_executing",
                agent_type=self.agent_type,
                input_length=len(input_text),
                tool_count=len(self.tools),
            )

            result = await executor.ainvoke({"input": input_text})

            # Process result
            return self._process_result(result)

        except Exception as e:
            logger.error(
                "structured_chat_agent_execution_failed",
                agent_type=self.agent_type,
                error=str(e),
                error_type=type(e).__name__,
            )

            return LangChainAgentResult(
                success=False,
                reasoning=f"Structured Chat agent execution failed: {e}",
                warnings=["Structured Chat agent error"],
                confidence=0.0,
            )

    def _format_structured_input(self, input_data: dict[str, Any]) -> str:
        """Format structured input data for the agent.

        Args:
            input_data: Structured input data

        Returns:
            Formatted input text
        """
        task = input_data.get("task", "")

        if task == "enrich_content":
            content = input_data.get("content", "")
            metadata = input_data.get("metadata", {})
            enrichment_type = input_data.get("enrichment_type", "general")

            return f"""Please enrich the following content with {enrichment_type} enhancements:

CONTENT:
{content[:3000]}{"..." if len(content) > 3000 else ""}

METADATA:
{metadata}

ENRICHMENT REQUIREMENTS:
- Add relevant examples and context
- Improve clarity and structure
- Include additional learning aids
- Ensure completeness

Use your enrichment tools to create enhanced, comprehensive content."""

        elif task == "generate_complex":
            inputs = input_data.get("inputs", [])
            generation_type = input_data.get("generation_type", "content")
            requirements = input_data.get("requirements", {})

            input_sections = []
            for i, input_item in enumerate(inputs, 1):
                input_sections.append(f"INPUT {i}:\n{input_item}")

            return f"""Generate {generation_type} using multiple input sources:

{chr(10).join(input_sections)}

REQUIREMENTS:
{requirements}

Create high-quality {generation_type} that effectively combines and processes all input sources."""

        elif task == "analyze_multi_dimensional":
            content_items = input_data.get("content_items", [])
            analysis_dimensions = input_data.get("analysis_dimensions", [])
            analysis_focus = input_data.get("analysis_focus", "comprehensive")

            content_sections = []
            for i, item in enumerate(content_items, 1):
                content_sections.append(f"CONTENT ITEM {i}:\n{item}")

            return f"""Perform {analysis_focus} analysis across multiple dimensions:

{chr(10).join(content_sections)}

ANALYSIS DIMENSIONS:
{", ".join(analysis_dimensions)}

Provide comprehensive analysis covering all specified dimensions."""

        else:
            # Generic structured input
            structured_parts = []
            for key, value in input_data.items():
                if key != "task":
                    structured_parts.append(f"{key.upper()}:\n{value}")

            return f"""Process this structured task: {task}

{chr(10).join(structured_parts)}"""

    def _process_result(self, raw_result: dict[str, Any]) -> LangChainAgentResult:
        """Process raw agent result into LangChainAgentResult.

        Args:
            raw_result: Raw result from AgentExecutor

        Returns:
            Processed result
        """
        output = raw_result.get("output", "")
        intermediate_steps = raw_result.get("intermediate_steps", [])

        # Extract tool usage information
        tool_calls = []
        for step in intermediate_steps:
            action, observation = step
            tool_calls.append(
                {
                    "tool": action.tool,
                    "input": action.tool_input,
                    "output": observation,
                }
            )

        # Extract confidence and warnings
        confidence = self._extract_confidence(output)
        warnings = self._extract_warnings(output)

        # Determine success based on output characteristics
        success = True
        if any(
            phrase in output.lower()
            for phrase in [
                "failed",
                "error",
                "unable to",
                "cannot complete",
                "insufficient",
                "incomplete",
            ]
        ):
            success = False

        # Adjust confidence based on tool usage complexity
        if len(tool_calls) > 2:  # Complex multi-tool usage
            confidence = min(confidence + 0.1, 1.0)

        logger.info(
            "structured_chat_agent_completed",
            agent_type=self.agent_type,
            success=success,
            tool_calls=len(tool_calls),
            output_length=len(output),
            confidence=confidence,
        )

        return LangChainAgentResult(
            success=success,
            reasoning=output,
            data={
                "raw_output": output,
                "intermediate_steps": intermediate_steps,
                "tool_calls": tool_calls,
            },
            warnings=warnings,
            confidence=confidence,
            metadata={
                "agent_type": self.agent_type,
                "tool_calls_count": len(tool_calls),
                "structured_input": True,
                "pattern": "structured_chat",
            },
        )
