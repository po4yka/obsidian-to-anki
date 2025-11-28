"""Search Agent implementation using LangChain.

This module provides a Self Ask With Search Agent for research tasks
and knowledge base queries.
"""

from typing import Any, Dict, List, Optional

from langchain.agents import create_self_ask_with_search_agent
from langchain.agents.self_ask_with_search.base import SelfAskWithSearchAgent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from ...utils.logging import get_logger
from .base import BaseLangChainAgent, LangChainAgentResult

logger = get_logger(__name__)


class SearchAgent(BaseLangChainAgent):
    """Search Agent for research and knowledge base queries.

    Uses Self Ask With Search methodology to decompose questions
    and perform targeted searches for information retrieval.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        agent_type: str = "search",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """Initialize Search Agent.

        Args:
            model: LangChain language model
            tools: List of tools (should include search tools)
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
        """Get default system prompt for Search Agent.

        Returns:
            System prompt string
        """
        return """You are a research specialist using search-based methodologies.

Your task is to find information, research topics, and retrieve knowledge from various sources.
You excel at decomposing complex questions and performing targeted searches.

Research capabilities:
- Break down complex questions into searchable components
- Use search tools to find relevant information
- Synthesize findings from multiple sources
- Provide comprehensive research results

Always use your search tools effectively to gather accurate, relevant information."""

    def _create_agent(self) -> SelfAskWithSearchAgent:
        """Create the underlying LangChain Self Ask With Search Agent.

        Returns:
            LangChain SelfAskWithSearchAgent instance
        """
        # Create self ask with search prompt template
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
        agent = create_self_ask_with_search_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt,
        )

        return agent

    async def run(self, input_data: Dict[str, Any], **kwargs) -> LangChainAgentResult:
        """Run the Search Agent.

        Args:
            input_data: Input data with research query
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
                max_iterations=kwargs.get("max_iterations", 4),
                max_execution_time=kwargs.get("max_execution_time"),
                handle_parsing_errors=kwargs.get("handle_parsing_errors", True),
                return_intermediate_steps=kwargs.get("return_intermediate_steps", True),
            )

            # Prepare input
            input_text = input_data.get("input", "")
            if not input_text and "query" in input_data:
                input_text = self._format_search_query(input_data)

            # Run the agent
            logger.info(
                "search_agent_executing",
                query_length=len(input_text),
            )

            result = await executor.ainvoke({"input": input_text})

            # Process result
            return self._process_result(result)

        except Exception as e:
            logger.error(
                "search_agent_execution_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

            return LangChainAgentResult(
                success=False,
                reasoning=f"Search agent execution failed: {e}",
                warnings=["Search agent error"],
                confidence=0.0,
            )

    def _format_search_query(self, input_data: Dict[str, Any]) -> str:
        """Format research query for search agent.

        Args:
            input_data: Input data with query information

        Returns:
            Formatted search query
        """
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        search_type = input_data.get("search_type", "general")

        if context:
            return f"""Research the following topic with context: {query}

Context: {context}

Search Type: {search_type}

Find comprehensive information and provide well-researched answers."""
        else:
            return f"""Research the following topic: {query}

Search Type: {search_type}

Gather relevant information and provide detailed findings."""

    def _process_result(self, raw_result: Dict[str, Any]) -> LangChainAgentResult:
        """Process raw agent result.

        Args:
            raw_result: Raw result from AgentExecutor

        Returns:
            Processed result
        """
        output = raw_result.get("output", "")
        intermediate_steps = raw_result.get("intermediate_steps", [])

        # Extract search actions
        search_actions = []
        for step in intermediate_steps:
            action, observation = step
            if hasattr(action, "tool") and "search" in action.tool.lower():
                search_actions.append(
                    {
                        "query": action.tool_input,
                        "results": observation,
                    }
                )

        # Extract confidence based on search thoroughness
        confidence = self._extract_confidence(output)

        # Adjust confidence based on number of searches performed
        search_count = len(search_actions)
        if search_count > 0:
            confidence = min(confidence + (search_count * 0.1), 1.0)

        warnings = self._extract_warnings(output)

        # Determine success
        success = True
        if (
            "no information found" in output.lower()
            or "unable to find" in output.lower()
        ):
            success = False
            confidence = max(confidence - 0.3, 0.0)

        logger.info(
            "search_agent_completed",
            success=success,
            searches_performed=search_count,
            output_length=len(output),
            confidence=confidence,
        )

        return LangChainAgentResult(
            success=success,
            reasoning=output,
            data={
                "raw_output": output,
                "search_actions": search_actions,
                "intermediate_steps": intermediate_steps,
            },
            warnings=warnings,
            confidence=confidence,
            metadata={
                "agent_type": self.agent_type,
                "searches_performed": search_count,
                "pattern": "self_ask_with_search",
            },
        )
