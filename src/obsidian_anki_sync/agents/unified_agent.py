"""Unified Agent Interface for seamless switching between agent frameworks.

This module provides a unified interface that can work with both PydanticAI
and LangChain agents, enabling seamless switching based on configuration
or task requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UnifiedAgentResult:
    """Unified result from any agent type."""

    success: bool
    reasoning: str
    data: Any | None = None
    warnings: list[str] | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None
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
        metadata: dict[str, Any],
        qa_pairs: list[dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards from Q/A pairs."""

    @abstractmethod
    async def validate_pre(
        self,
        note_content: str,
        metadata: dict[str, Any],
        qa_pairs: list[dict[str, Any]],
    ) -> UnifiedAgentResult:
        """Run pre-validation on note content."""

    @abstractmethod
    async def validate_post(
        self,
        cards: list[dict[str, Any]],
        metadata: dict[str, Any],
        strict_mode: bool = True,
    ) -> UnifiedAgentResult:
        """Run post-validation on generated cards."""

    @abstractmethod
    async def enrich_content(
        self,
        content: str,
        metadata: dict[str, Any],
        enrichment_type: str = "general",
    ) -> UnifiedAgentResult:
        """Enrich content with additional context."""

    @abstractmethod
    def get_agent_info(self) -> dict[str, Any]:
        """Get information about the underlying agent."""


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
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    from .pydantic.generator import GeneratorAgentAI

                    model = get_model_for_agent(self.config, "generator")
                    self._agents[agent_type] = GeneratorAgentAI(model)
                elif agent_type == "pre_validator":
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    from .pydantic.pre_validator import PreValidatorAgentAI

                    model = get_model_for_agent(self.config, "pre_validator")
                    self._agents[agent_type] = PreValidatorAgentAI(model)
                elif agent_type == "post_validator":
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    from .pydantic.post_validator import PostValidatorAgentAI

                    model = get_model_for_agent(self.config, "post_validator")
                    self._agents[agent_type] = PostValidatorAgentAI(model)
                elif agent_type == "context_enrichment":
                    from ...domain.interfaces.llm_config import get_model_for_agent
                    from .pydantic.context_enrichment import ContextEnrichmentAgentAI

                    model = get_model_for_agent(self.config, "context_enrichment")
                    self._agents[agent_type] = ContextEnrichmentAgentAI(model)
                else:
                    msg = f"Unknown PydanticAI agent type: {agent_type}"
                    raise ValueError(msg)
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
        metadata: dict[str, Any],
        qa_pairs: list[dict[str, Any]],
        slug_base: str,
    ) -> UnifiedAgentResult:
        """Generate APF cards using PydanticAI."""
        try:
            from ...models import NoteMetadata, QAPair

            # Convert to PydanticAI expected types
            note_metadata = NoteMetadata(**metadata)
            qa_list = [QAPair(**qa) for qa in qa_pairs]

            agent = self._get_agent("generator")
            result = await agent.generate_cards(
                note_content, note_metadata, qa_list, slug_base
            )

            return UnifiedAgentResult(
                success=result.success,
                reasoning=(
                    "Cards generated successfully"
                    if result.success
                    else result.errors[0]
                    if result.errors
                    else "Generation failed"
                ),
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
        metadata: dict[str, Any],
        qa_pairs: list[dict[str, Any]],
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
                reasoning=(
                    result.error_details
                    if not result.is_valid
                    else "Pre-validation passed"
                ),
                data=result,
                warnings=list(result.suggested_fixes),
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
        cards: list[dict[str, Any]],
        metadata: dict[str, Any],
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
                reasoning=(
                    result.error_details
                    if not result.is_valid
                    else "Post-validation passed"
                ),
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
        metadata: dict[str, Any],
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
                reasoning=(
                    "Content enriched successfully"
                    if result.success
                    else "Enrichment failed"
                ),
                data=result,
                warnings=result.warnings if hasattr(result, "warnings") else [],
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

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about PydanticAI agents."""
        return {
            "framework": self.agent_framework,
            "available_agents": list(self._agents.keys()),
            "description": "PydanticAI-based agents with type-safe structured outputs",
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
        self, framework: str | None = None, task_type: str = "generator"
    ) -> UnifiedAgentInterface:
        """Get unified agent for specified framework and task.

        Args:
            framework: Agent framework ("pydantic_ai" or "memory_enhanced").
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
                logger.warning("memory_enhanced_not_available_fallback_to_pydantic_ai")
                framework = "pydantic_ai"

        cache_key = f"{framework}_{task_type}"

        if cache_key not in self._agents:
            if framework == "pydantic_ai":
                self._agents[cache_key] = PydanticAIUnifiedAgent(self.config)
            else:
                msg = f"Unknown agent framework: {framework}. Only 'pydantic_ai' and 'memory_enhanced' are supported."
                raise ValueError(msg)

        return self._agents[cache_key]  # type: ignore[no-any-return]

    def get_agent_with_fallback(
        self,
        primary_framework: str,
        fallback_framework: str,
        task_type: str = "generator",
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
