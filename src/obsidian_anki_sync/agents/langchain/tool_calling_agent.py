"""Tool Calling Agent implementation using LangChain.

This module provides a Tool Calling Agent that supports parallel function calling
and integrates with the card generation pipeline.
"""

from typing import Any

from langchain.agents import create_tool_calling_agent
from langchain.agents.tool_calling_agent.base import ToolCallingAgent as LangChainToolCallingAgent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from ...utils.logging import get_logger
from .base import BaseLangChainAgent, LangChainAgentResult

logger = get_logger(__name__)


class ToolCallingAgent(BaseLangChainAgent):
    """Tool Calling Agent for card generation and validation.

    Uses LangChain's tool calling agent which supports parallel function calling
    and is the recommended approach for modern chat models.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool],
        agent_type: str = "tool_calling",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        enable_parallel_tool_calls: bool = True,
    ):
        """Initialize Tool Calling Agent.

        Args:
            model: LangChain language model (must support tool calling)
            tools: List of tools for the agent
            agent_type: Type of agent (generator, validator, etc.)
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            enable_parallel_tool_calls: Whether to enable parallel tool calls
        """
        self.enable_parallel_tool_calls = enable_parallel_tool_calls

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
            "generator": """You are an expert card generation agent using APF (Active Prompt Format) for Anki flashcards.

Your task is to generate high-quality flashcards from Q&A pairs. You have access to various tools to help you:

- card_template: Generate APF card templates
- slug_generator: Create unique slugs for cards
- html_formatter: Format and validate HTML content
- apf_validator: Validate APF card structure

Use these tools effectively to create well-formatted, valid APF cards. Always ensure cards follow APF v2.1 specification.""",
            "validator": """You are a validation specialist for APF cards and content.

Your role is to validate and ensure quality of generated cards and content. You have access to:

- apf_validator: Check APF format compliance
- html_formatter: Validate HTML structure
- content_hash: Verify content integrity

Use these tools to thoroughly validate content and provide detailed feedback on any issues found.""",
            "pre_validator": """You are a pre-validation specialist for note content and structure.

Your task is to validate notes before card generation. You have access to:

- metadata_extractor: Extract and validate YAML frontmatter
- qa_extractor: Parse Q&A pairs from content
- content_hash: Check content integrity

Use these tools to ensure notes are properly structured before processing.""",
            "post_validator": """You are a post-validation specialist for generated cards.

Your role is to validate completed cards and suggest improvements. You have access to:

- apf_validator: Final APF compliance check
- html_formatter: HTML validation and formatting
- content_hash: Verify no corruption occurred

Provide comprehensive validation results and actionable improvement suggestions.""",
        }

        return prompts.get(
            agent_type, "You are a helpful AI assistant with access to various tools."
        )

    def _create_agent(self) -> Any:
        """Create the underlying LangChain tool calling agent.

        Returns:
            LangChain ToolCallingAgent instance
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the agent
        agent = create_tool_calling_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt,
        )

        return agent

    async def run(self, input_data: dict[str, Any], **kwargs: Any) -> LangChainAgentResult:
        """Run the tool calling agent.

        Args:
            input_data: Input data containing 'input' key with user query
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

            # Prepare input
            input_text = input_data.get("input", "")
            if not input_text and "task" in input_data:
                # Convert task-based input to natural language
                input_text = self._format_task_input(input_data)

            # Run the agent
            logger.info(
                "tool_calling_agent_executing",
                agent_type=self.agent_type,
                input_length=len(input_text),
                tool_count=len(self.tools),
            )

            result = await executor.ainvoke({"input": input_text})

            # Process result
            return self._process_result(result)

        except Exception as e:
            logger.error(
                "tool_calling_agent_execution_failed",
                agent_type=self.agent_type,
                error=str(e),
                error_type=type(e).__name__,
            )

            return LangChainAgentResult(
                success=False,
                reasoning=f"Agent execution failed: {e}",
                warnings=["Tool calling agent error"],
                confidence=0.0,
            )

    def _format_task_input(self, input_data: dict[str, Any]) -> str:
        """Format task-based input into natural language query.

        Args:
            input_data: Task input data

        Returns:
            Natural language query for the agent
        """
        task = input_data.get("task", "")

        if task == "generate_cards":
            note_content = input_data.get("note_content", "")
            qa_pairs = input_data.get("qa_pairs", [])
            slug_base = input_data.get("slug_base", "")

            return f"""Generate APF cards from the following note content:

Note Content:
{note_content[:1000]}{"..." if len(note_content) > 1000 else ""}

Q&A Pairs ({len(qa_pairs)}):
{chr(10).join(f"- Q: {qa.get('question_en', '')[:100]}... A: {qa.get('answer_en', '')[:100]}..." for qa in qa_pairs)}

Base Slug: {slug_base}

Please use the available tools to generate properly formatted APF cards."""

        elif task == "validate_content":
            content = input_data.get("content", "")
            validation_type = input_data.get("validation_type", "general")

            return f"""Validate the following content for {validation_type}:

Content:
{content[:2000]}{"..." if len(content) > 2000 else ""}

Please use the appropriate validation tools and provide detailed feedback."""

        elif task == "extract_metadata":
            note_content = input_data.get("note_content", "")

            return f"""Extract metadata from the following note:

Note Content:
{note_content[:1500]}{"..." if len(note_content) > 1500 else ""}

Please use the metadata extraction tools to parse YAML frontmatter and extract relevant information."""

        else:
            # Generic fallback
            return f"Process the following task: {task}. Input data: {input_data}"

    def _process_result(self, raw_result: dict[str, Any]) -> LangChainAgentResult:
        """Process raw agent result into LangChainAgentResult.

        Args:
            raw_result: Raw result from AgentExecutor

        Returns:
            Processed result
        """
        output = raw_result.get("output", "")
        intermediate_steps = raw_result.get("intermediate_steps", [])

        # Extract confidence and warnings from output
        confidence = self._extract_confidence(output)
        warnings = self._extract_warnings(output)

        # Log tool usage
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

        logger.info(
            "tool_calling_agent_completed",
            agent_type=self.agent_type,
            output_length=len(output),
            tool_calls=len(tool_calls),
            confidence=confidence,
        )

        return LangChainAgentResult(
            success=True,
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
                "parallel_tools_enabled": self.enable_parallel_tool_calls,
            },
        )
