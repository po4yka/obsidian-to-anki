"""LangChain agent factory for creating and configuring agents.

This module provides a factory pattern for creating LangChain agents
based on configuration and task requirements.
"""

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from ...config import Config
from ...utils.logging import get_logger
from .base import BaseLangChainAgent
from .tools import get_tools_for_agent

logger = get_logger(__name__)


class LangChainAgentFactory:
    """Factory for creating LangChain agents."""

    def __init__(self, config: Config):
        """Initialize factory with configuration.

        Args:
            config: Service configuration
        """
        self.config = config
        self._agent_cache: dict[str, BaseLangChainAgent] = {}

    def create_agent(
        self,
        agent_type: str,
        model: BaseLanguageModel | None = None,
        langchain_agent_type: str = "tool_calling",
        tools: list[BaseTool] | None = None,
        **kwargs,
    ) -> BaseLangChainAgent:
        """Create a LangChain agent.

        Args:
            agent_type: Type of agent (generator, validator, etc.)
            model: Language model to use (optional, will use config default)
            langchain_agent_type: LangChain agent type (tool_calling, react, etc.)
            tools: Tools for the agent (optional, will use defaults)
            **kwargs: Additional arguments

        Returns:
            Configured LangChain agent
        """
        # Create cache key
        cache_key = f"{agent_type}_{langchain_agent_type}_{id(model)}"

        # Check cache
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        # Get model if not provided
        if model is None:
            model = self._get_model_for_agent(agent_type)

        # Get tools if not provided
        if tools is None:
            tools = get_tools_for_agent(agent_type)

        # Get agent configuration
        agent_config = self._get_agent_config(agent_type, langchain_agent_type)

        # Create agent based on type
        agent = self._create_agent_by_type(
            agent_type=agent_type,
            langchain_agent_type=langchain_agent_type,
            model=model,
            tools=tools,
            config=agent_config,
            **kwargs,
        )

        # Cache the agent
        self._agent_cache[cache_key] = agent

        logger.info(
            "langchain_agent_created",
            agent_type=agent_type,
            langchain_agent_type=langchain_agent_type,
            model=str(model),
            tool_count=len(tools),
        )

        return agent

    def _get_model_for_agent(self, agent_type: str) -> BaseLanguageModel:
        """Get the appropriate model for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Configured language model
        """
        # This would need to be implemented based on the config
        # For now, return a placeholder
        # In practice, this would convert config model names to LangChain models
        raise NotImplementedError("Model factory integration needed")

    def _get_agent_config(
        self, agent_type: str, langchain_agent_type: str
    ) -> dict[str, Any]:
        """Get configuration for a specific agent.

        Args:
            agent_type: Type of agent (generator, validator, etc.)
            langchain_agent_type: LangChain agent type

        Returns:
            Agent configuration dictionary
        """
        # Default configurations
        default_configs = {
            "tool_calling": {
                "temperature": 0.0,
                "max_tokens": None,
                "enable_parallel_tool_calls": True,
            },
            "react": {
                "temperature": 0.0,
                "max_tokens": 2000,
                "max_iterations": 5,
            },
            "structured_chat": {
                "temperature": 0.0,
                "max_tokens": None,
            },
            "json_chat": {
                "temperature": 0.0,
                "max_tokens": None,
            },
        }

        base_config = default_configs.get(langchain_agent_type, {})

        # Agent-specific overrides
        agent_overrides = {
            "generator": {
                "temperature": 0.3,  # Allow some creativity for generation
                "max_tokens": 4000,
            },
            "validator": {
                "temperature": 0.0,  # Strict validation
                "max_tokens": 1000,
            },
            "pre_validator": {
                "temperature": 0.0,
                "max_tokens": 1500,
            },
            "post_validator": {
                "temperature": 0.0,
                "max_tokens": 2000,
            },
        }

        # Merge configurations
        config = base_config.copy()
        if agent_type in agent_overrides:
            config.update(agent_overrides[agent_type])

        return config  # type: ignore[no-any-return]

    def _create_agent_by_type(
        self,
        agent_type: str,
        langchain_agent_type: str,
        model: BaseLanguageModel,
        tools: list[BaseTool],
        config: dict[str, Any],
        **kwargs,
    ) -> BaseLangChainAgent:
        """Create agent based on LangChain agent type.

        Args:
            agent_type: Application agent type
            langchain_agent_type: LangChain agent type
            model: Language model
            tools: Tools for the agent
            config: Agent configuration
            **kwargs: Additional arguments

        Returns:
            Created agent instance
        """
        if langchain_agent_type == "tool_calling":
            from .tool_calling_agent import ToolCallingAgent

            return ToolCallingAgent(
                model=model, tools=tools, agent_type=agent_type, **config, **kwargs
            )

        elif langchain_agent_type == "react":
            from .react_agent import ReActAgent

            return ReActAgent(
                model=model, tools=tools, agent_type=agent_type, **config, **kwargs
            )

        elif langchain_agent_type == "structured_chat":
            from .structured_chat_agent import StructuredChatAgent

            return StructuredChatAgent(
                model=model, tools=tools, agent_type=agent_type, **config, **kwargs
            )

        elif langchain_agent_type == "json_chat":
            from .json_chat_agent import JSONChatAgent

            return JSONChatAgent(
                model=model, tools=tools, agent_type=agent_type, **config, **kwargs
            )

        else:
            raise ValueError(f"Unknown LangChain agent type: {langchain_agent_type}")

    def get_available_agent_types(self) -> list[str]:
        """Get list of available LangChain agent types.

        Returns:
            List of agent type names
        """
        return ["tool_calling", "react", "structured_chat", "json_chat"]

    def clear_cache(self):
        """Clear the agent cache."""
        self._agent_cache.clear()
        logger.info("langchain_agent_cache_cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached agents.

        Returns:
            Cache information
        """
        return {
            "cached_agents": len(self._agent_cache),
            "cache_keys": list(self._agent_cache.keys()),
        }
