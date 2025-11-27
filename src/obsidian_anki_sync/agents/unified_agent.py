"""Unified Agent Interface for seamless switching between agent frameworks.

This module provides a unified interface that can work with both PydanticAI
and LangChain agents, enabling seamless switching based on configuration
or task requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ..config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UnifiedAgentResult:
    """Unified result from any agent type."""

    success: bool
    reasoning: str
    data: Optional[Any] = None
    warnings: Optional[List[str]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    agent_framework: str = "unknown"  # "pydantic_ai" or "langchain"
    agent_type: str = "unknown"

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class UnifiedAgentInterface(ABC):
    """Abstract base class for unified agent interface."""

    @abstractmethod
    async def generate_cards(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards from Q/A pairs."""
        pass

    @abstractmethod
    async def validate_pre(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
    ) -> UnifiedAgentResult:
        """Run pre-validation on note content."""
        pass

    @abstractmethod
    async def validate_post(
        self,
        cards: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        strict_mode: bool = True,
    ) -> UnifiedAgentResult:
        """Run post-validation on generated cards."""
        pass

    @abstractmethod
    async def enrich_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        enrichment_type: str = "general",
    ) -> UnifiedAgentResult:
        """Enrich content with additional context."""
        pass

    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the underlying agent."""
        pass


class PydanticAIUnifiedAgent(UnifiedAgentInterface):
    """Unified interface wrapper for PydanticAI agents."""

    def __init__(self, config: Config):
        """Initialize PydanticAI unified agent.

        Args:
            config: Service configuration
        """
        self.config = config
        self.agent_framework = "pydantic_ai"
        self._agents = {}

    def _get_agent(self, agent_type: str):
        """Get or create a PydanticAI agent.

        Args:
            agent_type: Type of agent to get

        Returns:
            Agent instance
        """
        if agent_type not in self._agents:
            try:
                if agent_type == "generator":
                    from .pydantic.generator import GeneratorAgentAI
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    model = get_model_for_agent(self.config, "generator")
                    self._agents[agent_type] = GeneratorAgentAI(model)
                elif agent_type == "pre_validator":
                    from .pydantic.pre_validator import PreValidatorAgentAI
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    model = get_model_for_agent(self.config, "pre_validator")
                    self._agents[agent_type] = PreValidatorAgentAI(model)
                elif agent_type == "post_validator":
                    from .pydantic.post_validator import PostValidatorAgentAI
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    model = get_model_for_agent(self.config, "post_validator")
                    self._agents[agent_type] = PostValidatorAgentAI(model)
                elif agent_type == "context_enrichment":
                    from .pydantic.context_enrichment import ContextEnrichmentAgentAI
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    model = get_model_for_agent(
                        self.config, "context_enrichment")
                    self._agents[agent_type] = ContextEnrichmentAgentAI(model)
                else:
                    raise ValueError(
                        f"Unknown PydanticAI agent type: {agent_type}")
            except Exception as e:
                logger.warning(
                    "pydantic_ai_agent_creation_failed",
                    agent_type=agent_type,
                    error=str(e),
                )
                raise

        return self._agents[agent_type]

    async def generate_cards(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards using PydanticAI."""
        try:
            from ...models import NoteMetadata, QAPair

            # Convert to PydanticAI expected types
            note_metadata = NoteMetadata(**metadata)
            qa_list = [QAPair(**qa) for qa in qa_pairs]

            agent = self._get_agent("generator")
            result = await agent.generate_cards(note_content, note_metadata, qa_list, slug_base)

            return UnifiedAgentResult(
                success=result.success,
                reasoning="Cards generated successfully" if result.success else result.errors[
                    0] if result.errors else "Generation failed",
                data=result,
                warnings=result.warnings,
                confidence=0.9,  # PydanticAI generally reliable
                agent_framework=self.agent_framework,
                agent_type="generator",
            )

        except Exception as e:
            logger.error("pydantic_ai_generation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"PydanticAI generation failed: {e}",
                warnings=["PydanticAI agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="generator",
            )

    async def validate_pre(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
    ) -> UnifiedAgentResult:
        """Run pre-validation using PydanticAI."""
        try:
            from ...models import NoteMetadata, QAPair

            note_metadata = NoteMetadata(**metadata)
            qa_list = [QAPair(**qa) for qa in qa_pairs]

            agent = self._get_agent("pre_validator")
            result = await agent.validate(note_content, note_metadata, qa_list)

            return UnifiedAgentResult(
                success=result.is_valid,
                reasoning=result.error_details if not result.is_valid else "Pre-validation passed",
                data=result,
                warnings=[fix for fix in result.suggested_fixes],
                confidence=result.confidence,
                agent_framework=self.agent_framework,
                agent_type="pre_validator",
            )

        except Exception as e:
            logger.error("pydantic_ai_pre_validation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"PydanticAI pre-validation failed: {e}",
                warnings=["PydanticAI agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="pre_validator",
            )

    async def validate_post(
        self,
        cards: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        strict_mode: bool = True,
    ) -> UnifiedAgentResult:
        """Run post-validation using PydanticAI."""
        try:
            from ...models import GeneratedCard

            # Convert to PydanticAI expected types
            card_list = [GeneratedCard(**card) for card in cards]

            agent = self._get_agent("post_validator")
            result = await agent.validate(card_list, strict_mode)

            return UnifiedAgentResult(
                success=result.is_valid,
                reasoning=result.error_details if not result.is_valid else "Post-validation passed",
                data=result,
                warnings=result.warnings,
                confidence=result.confidence,
                agent_framework=self.agent_framework,
                agent_type="post_validator",
            )

        except Exception as e:
            logger.error("pydantic_ai_post_validation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"PydanticAI post-validation failed: {e}",
                warnings=["PydanticAI agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="post_validator",
            )

    async def enrich_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        enrichment_type: str = "general",
    ) -> UnifiedAgentResult:
        """Enrich content using PydanticAI."""
        try:
            agent = self._get_agent("context_enrichment")

            # PydanticAI enrichment expects specific format
            enrichment_input = {
                "content": content,
                "metadata": metadata,
                "enrichment_type": enrichment_type,
            }

            result = await agent.enrich(enrichment_input)

            return UnifiedAgentResult(
                success=result.success,
                reasoning="Content enriched successfully" if result.success else "Enrichment failed",
                data=result,
                warnings=result.warnings if hasattr(
                    result, "warnings") else [],
                confidence=0.8,  # Enrichment is subjective
                agent_framework=self.agent_framework,
                agent_type="context_enrichment",
            )

        except Exception as e:
            logger.error("pydantic_ai_enrichment_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"PydanticAI enrichment failed: {e}",
                warnings=["PydanticAI agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="context_enrichment",
            )

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about PydanticAI agents."""
        return {
            "framework": self.agent_framework,
            "available_agents": list(self._agents.keys()),
            "description": "PydanticAI-based agents with type-safe structured outputs",
        }


class LangChainUnifiedAgent(UnifiedAgentInterface):
    """Unified interface wrapper for LangChain agents."""

    def __init__(self, config: Config):
        """Initialize LangChain unified agent.

        Args:
            config: Service configuration
        """
        self.config = config
        self.agent_framework = "langchain"
        self._factory = None

    def _get_factory(self):
        """Get or create LangChain agent factory."""
        if self._factory is None:
            from .langchain.factory import LangChainAgentFactory
            self._factory = LangChainAgentFactory(self.config)
        return self._factory

    def _get_agent_for_task(self, task_type: str):
        """Get appropriate LangChain agent for task type.

        Args:
            task_type: Type of task (generator, validator, etc.)

        Returns:
            LangChain agent instance
        """
        factory = self._get_factory()

        # Map task types to agent configurations
        agent_configs = {
            "generator": {
                "agent_type": "generator",
                "langchain_agent_type": getattr(self.config, "langchain_generator_type", "tool_calling"),
            },
            "pre_validator": {
                "agent_type": "pre_validator",
                "langchain_agent_type": getattr(self.config, "langchain_pre_validator_type", "react"),
            },
            "post_validator": {
                "agent_type": "post_validator",
                "langchain_agent_type": getattr(self.config, "langchain_post_validator_type", "tool_calling"),
            },
            "context_enrichment": {
                "agent_type": "enrichment",
                "langchain_agent_type": getattr(self.config, "langchain_enrichment_type", "structured_chat"),
            },
        }

        config = agent_configs.get(task_type, agent_configs["generator"])
        return factory.create_agent(**config)

    async def generate_cards(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards using LangChain agents."""
        try:
            from .langchain.tool_calling_generator import ToolCallingGeneratorAgent

            # Use specialized generator agent
            agent = ToolCallingGeneratorAgent()
            result = await agent.generate_cards(note_content, metadata, qa_pairs, slug_base)

            return UnifiedAgentResult(
                success=result.success,
                reasoning="Cards generated successfully" if result.success else "Generation failed",
                data=result,
                warnings=result.warnings,
                confidence=0.85,  # LangChain agents generally reliable
                agent_framework=self.agent_framework,
                agent_type="tool_calling_generator",
            )

        except Exception as e:
            logger.error("langchain_generation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"LangChain generation failed: {e}",
                warnings=["LangChain agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="generator",
            )

    async def validate_pre(
        self,
        note_content: str,
        metadata: Dict[str, Any],
        qa_pairs: List[Dict[str, Any]],
    ) -> UnifiedAgentResult:
        """Run pre-validation using LangChain agents."""
        try:
            from .langchain.react_validator import ReActValidatorAgent

            # Use ReAct validator for transparent reasoning
            agent = ReActValidatorAgent(validator_type="pre")
            result = await agent.validate_pre(note_content, metadata, qa_pairs)

            return UnifiedAgentResult(
                success=result.is_valid,
                reasoning=result.error_details if not result.is_valid else "Pre-validation passed",
                data=result,
                warnings=result.warnings,
                confidence=result.confidence,
                agent_framework=self.agent_framework,
                agent_type="react_pre_validator",
            )

        except Exception as e:
            logger.error("langchain_pre_validation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"LangChain pre-validation failed: {e}",
                warnings=["LangChain agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="pre_validator",
            )

    async def validate_post(
        self,
        cards: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        strict_mode: bool = True,
    ) -> UnifiedAgentResult:
        """Run post-validation using LangChain agents."""
        try:
            from .langchain.tool_calling_validator import ToolCallingValidatorAgent
            from ...models import GeneratedCard

            # Convert to expected types
            card_list = [GeneratedCard(**card) for card in cards]

            # Use tool calling validator for parallel validation
            agent = ToolCallingValidatorAgent(validator_type="post")
            result = await agent.validate_post(card_list, metadata, strict_mode)

            return UnifiedAgentResult(
                success=result.is_valid,
                reasoning=result.error_details if not result.is_valid else "Post-validation passed",
                data=result,
                warnings=result.warnings,
                confidence=result.confidence,
                agent_framework=self.agent_framework,
                agent_type="tool_calling_post_validator",
            )

        except Exception as e:
            logger.error("langchain_post_validation_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"LangChain post-validation failed: {e}",
                warnings=["LangChain agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="post_validator",
            )

    async def enrich_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        enrichment_type: str = "general",
    ) -> UnifiedAgentResult:
        """Enrich content using LangChain agents."""
        try:
            agent = self._get_agent_for_task("context_enrichment")

            input_data = {
                "task": "enrich_content",
                "content": content,
                "metadata": metadata,
                "enrichment_type": enrichment_type,
            }

            result = await agent.run(input_data)

            return UnifiedAgentResult(
                success=result.success,
                reasoning=result.reasoning,
                data=result.data,
                warnings=result.warnings,
                confidence=result.confidence,
                agent_framework=self.agent_framework,
                agent_type="enrichment",
            )

        except Exception as e:
            logger.error("langchain_enrichment_failed", error=str(e))
            return UnifiedAgentResult(
                success=False,
                reasoning=f"LangChain enrichment failed: {e}",
                warnings=["LangChain agent error"],
                confidence=0.0,
                agent_framework=self.agent_framework,
                agent_type="context_enrichment",
            )

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about LangChain agents."""
        factory = self._get_factory()
        return {
            "framework": self.agent_framework,
            "factory_info": factory.get_cache_info(),
            "available_types": factory.get_available_agent_types(),
            "description": "LangChain-based agents with various reasoning patterns",
        }


class UnifiedAgentSelector:
    """Selector for choosing between agent frameworks."""

    def __init__(self, config: Config):
        """Initialize agent selector.

        Args:
            config: Service configuration
        """
        self.config = config
        self._agents = {}

        # Initialize memory-enhanced generator if available
        self.memory_enhanced_available = False
        try:
            from .memory_enhanced_generator import MemoryEnhancedGenerator
            self.memory_enhanced_generator = MemoryEnhancedGenerator(config)
            self.memory_enhanced_available = True
            logger.info("memory_enhanced_generator_available")
        except Exception as e:
            logger.warning(f"Memory-enhanced generator not available: {e}")

    def get_agent(
        self,
        framework: Optional[str] = None,
        task_type: str = "generator"
    ) -> UnifiedAgentInterface:
        """Get unified agent for specified framework and task.

        Args:
            framework: Agent framework ("pydantic_ai", "langchain", or "memory_enhanced").
                       If None, uses config default.
            task_type: Type of task (generator, validator, etc.)

        Returns:
            Unified agent interface
        """
        # Determine framework
        if framework is None:
            framework = getattr(self.config, "agent_framework", "pydantic_ai")

        # Special handling for memory-enhanced framework
        if framework == "memory_enhanced":
            if self.memory_enhanced_available and task_type == "generator":
                return self.memory_enhanced_generator
            else:
                # Fallback to pydantic_ai if memory not available
                logger.warning(
                    "memory_enhanced_not_available_fallback_to_pydantic_ai")
                framework = "pydantic_ai"

        cache_key = f"{framework}_{task_type}"

        if cache_key not in self._agents:
            if framework == "pydantic_ai":
                self._agents[cache_key] = PydanticAIUnifiedAgent(self.config)
            elif framework == "langchain":
                self._agents[cache_key] = LangChainUnifiedAgent(self.config)
            else:
                raise ValueError(f"Unknown agent framework: {framework}")

        return self._agents[cache_key]

    def get_agent_with_fallback(
        self,
        primary_framework: str,
        fallback_framework: str,
        task_type: str = "generator"
    ) -> UnifiedAgentInterface:
        """Get agent with fallback support.

        Args:
            primary_framework: Primary framework to try
            fallback_framework: Fallback framework if primary fails
            task_type: Type of task

        Returns:
            Unified agent interface
        """
        try:
            return self.get_agent(primary_framework, task_type)
        except Exception as e:
            logger.warning(
                "primary_agent_framework_failed",
                primary=primary_framework,
                fallback=fallback_framework,
                task_type=task_type,
                error=str(e),
            )
            return self.get_agent(fallback_framework, task_type)
