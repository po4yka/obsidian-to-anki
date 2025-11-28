"""ReAct Agent implementation using LangChain.

This module provides a ReAct (Reasoning + Acting) Agent that alternates
between reasoning steps and tool usage for transparent decision-making.
"""

from typing import Any

from langchain.agents import create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from ...utils.logging import get_logger
from .base import BaseLangChainAgent, LangChainAgentResult

logger = get_logger(__name__)


class ReActAgent(BaseLangChainAgent):
    """ReAct Agent for reasoning and validation tasks.

    Uses the ReAct pattern: Reason → Act → Observe → Reason → ...
    Provides transparent reasoning chains and is good for validation,
    diagnosis, and repair tasks.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool],
        agent_type: str = "react",
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_iterations: int = 5,
    ):
        """Initialize ReAct Agent.

        Args:
            model: LangChain language model
            tools: List of tools for the agent
            agent_type: Type of agent (validator, repair, etc.)
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_iterations: Maximum reasoning iterations
        """
        self.max_iterations = max_iterations

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
            "validator": """You are a validation specialist using ReAct (Reasoning + Acting) methodology.

Your task is to validate content and provide detailed analysis. You have access to various validation tools.

For each validation task:
1. REASON about what needs to be checked
2. ACT by using appropriate tools
3. OBSERVE the tool results
4. REASON about the findings
5. Continue until you have a complete assessment

Be thorough and methodical in your validation process.""",
            "repair": """You are a content repair specialist using ReAct methodology.

Your role is to diagnose issues and suggest fixes. You have access to various tools for analysis and repair.

For each repair task:
1. REASON about what might be wrong
2. ACT by using diagnostic tools
3. OBSERVE the results
4. REASON about the root cause
5. ACT by suggesting or applying fixes
6. Continue until the issue is resolved

Always explain your reasoning clearly.""",
            "analyzer": """You are a content analyzer using ReAct methodology.

Your task is to analyze content and provide insights. You have access to various analysis tools.

For each analysis task:
1. REASON about what aspects to examine
2. ACT by using analysis tools
3. OBSERVE the results
4. REASON about patterns and insights
5. Continue until you have comprehensive analysis

Be systematic and thorough in your analysis.""",
        }

        return prompts.get(
            agent_type,
            """You are a helpful AI assistant using ReAct methodology.

For each task:
1. REASON about the problem
2. ACT by using available tools
3. OBSERVE the results
4. REASON about what you learned
5. Continue until the task is complete

Always explain your reasoning clearly.""",
        )

    def _create_agent(self) -> Any:
        """Create the underlying LangChain ReAct agent.

        Returns:
            LangChain ReActAgent instance
        """
        # Create ReAct prompt template
        template = """{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        # Create the agent
        agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt,
        )

        return agent

    async def run(
        self, input_data: dict[str, Any], **kwargs: Any
    ) -> LangChainAgentResult:
        """Run the ReAct agent.

        Args:
            input_data: Input data containing 'input' key with query
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
                max_iterations=self.max_iterations,
                max_execution_time=kwargs.get("max_execution_time"),
                handle_parsing_errors=kwargs.get("handle_parsing_errors", True),
                return_intermediate_steps=kwargs.get("return_intermediate_steps", True),
            )

            # Prepare input
            input_text = input_data.get("input", "")
            if not input_text and "task" in input_data:
                input_text = self._format_task_input(input_data)

            # Run the agent
            logger.info(
                "react_agent_executing",
                agent_type=self.agent_type,
                input_length=len(input_text),
                max_iterations=self.max_iterations,
            )

            result = await executor.ainvoke({"input": input_text})

            # Process result
            return self._process_result(result)

        except Exception as e:
            logger.error(
                "react_agent_execution_failed",
                agent_type=self.agent_type,
                error=str(e),
                error_type=type(e).__name__,
            )

            return LangChainAgentResult(
                success=False,
                reasoning=f"ReAct agent execution failed: {e}",
                warnings=["ReAct agent error"],
                confidence=0.0,
            )

    def _format_task_input(self, input_data: dict[str, Any]) -> str:
        """Format task-based input for ReAct agent.

        Args:
            input_data: Task input data

        Returns:
            Formatted input for ReAct reasoning
        """
        task = input_data.get("task", "")

        if task == "validate_content":
            content = input_data.get("content", "")
            validation_type = input_data.get("validation_type", "general")

            return f"""Please validate the following content for {validation_type}:

Content to validate:
{content[:2000]}{"..." if len(content) > 2000 else ""}

Use your tools to check for issues, format problems, and quality concerns.
Provide a detailed analysis of any problems found and suggestions for fixes."""

        elif task == "diagnose_issue":
            error_message = input_data.get("error_message", "")
            context = input_data.get("context", "")

            return f"""Please diagnose the following issue:

Error: {error_message}

Context: {context}

Use your tools to investigate the root cause and suggest solutions.
Think step by step about what could be causing this problem."""

        elif task == "analyze_pattern":
            content = input_data.get("content", "")
            pattern_type = input_data.get("pattern_type", "general")

            return f"""Please analyze the following content for {pattern_type} patterns:

Content:
{content[:2000]}{"..." if len(content) > 2000 else ""}

Use your tools to identify patterns, trends, and insights.
Provide a systematic analysis of what you find."""

        else:
            return f"Process this task: {task}. Details: {input_data}"

    def _process_result(self, raw_result: dict[str, Any]) -> LangChainAgentResult:
        """Process raw agent result into LangChainAgentResult.

        Args:
            raw_result: Raw result from AgentExecutor

        Returns:
            Processed result
        """
        output = raw_result.get("output", "")
        intermediate_steps = raw_result.get("intermediate_steps", [])

        # Extract reasoning chain
        reasoning_steps = []
        for step in intermediate_steps:
            action, observation = step
            reasoning_steps.append(
                {
                    "thought": getattr(action, "log", ""),
                    "action": action.tool,
                    "action_input": action.tool_input,
                    "observation": observation,
                }
            )

        # Extract final answer and confidence
        confidence = self._extract_confidence(output)
        warnings = self._extract_warnings(output)

        # Check if the task was completed successfully
        success = True
        if "failed" in output.lower() or "error" in output.lower():
            success = False
        if "could not" in output.lower() or "unable to" in output.lower():
            success = False

        logger.info(
            "react_agent_completed",
            agent_type=self.agent_type,
            success=success,
            reasoning_steps=len(reasoning_steps),
            output_length=len(output),
            confidence=confidence,
        )

        return LangChainAgentResult(
            success=success,
            reasoning=output,
            data={
                "raw_output": output,
                "reasoning_chain": reasoning_steps,
                "intermediate_steps": intermediate_steps,
            },
            warnings=warnings,
            confidence=confidence,
            metadata={
                "agent_type": self.agent_type,
                "reasoning_steps": len(reasoning_steps),
                "max_iterations": self.max_iterations,
                "pattern": "react",
            },
        )
