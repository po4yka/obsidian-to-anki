"""LangChain-based agent implementations for card generation pipeline.

This package provides LangChain agent types (Tool Calling, ReAct, Structured Chat, etc.)
as alternatives and complements to PydanticAI agents.
"""

from .base import BaseLangChainAgent, LangChainAgentResult
from .factory import LangChainAgentFactory
from .tools import (
    APFValidatorTool,
    CardTemplateTool,
    ContentHashTool,
    HTMLFormatterTool,
    MetadataExtractorTool,
    SlugGeneratorTool,
)

__all__ = [
    # Tools
    "APFValidatorTool",
    # Base classes
    "BaseLangChainAgent",
    "CardTemplateTool",
    "ContentHashTool",
    "HTMLFormatterTool",
    # Factory
    "LangChainAgentFactory",
    "LangChainAgentResult",
    "MetadataExtractorTool",
    "SlugGeneratorTool",
]
