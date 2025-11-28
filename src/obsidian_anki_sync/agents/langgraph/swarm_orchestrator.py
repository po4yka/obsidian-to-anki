"""LangGraph Swarm Orchestrator for dynamic agent collaboration.

This module implements the Swarm pattern where agents can dynamically hand off
control to other agents based on expertise, improving efficiency and specialization.

NEW in 2025: Dynamic agent handoffs, expertise-based routing, and collaborative workflows.
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langgraph_swarm import Swarm

from ...config import Config
from ...utils.logging import get_logger
from .state import PipelineState

logger = get_logger(__name__)


@dataclass
class AgentExpertise:
    """Defines an agent's expertise areas and handoff capabilities."""

    name: str
    expertise: list[str]
    handoff_agents: list[str]
    model: Any
    description: str


@dataclass
class SwarmResult:
    """Result from swarm execution."""

    final_agent: str
    handoffs: list[dict[str, Any]]
    total_time: float
    success: bool
    final_state: PipelineState


class LangGraphSwarmOrchestrator:
    """Swarm orchestrator for dynamic multi-agent collaboration.

    Uses LangGraph Swarm pattern to enable agents to hand off control
    based on expertise, improving efficiency and specialization.
    """

    def __init__(
        self,
        config: Config,
        enable_swarm: bool = True,
        max_handoffs: int = 3,
        handoff_strategy: Literal["expertise", "load", "random"] = "expertise",
    ):
        """Initialize swarm orchestrator.

        Args:
            config: Service configuration
            enable_swarm: Whether to use swarm pattern
            max_handoffs: Maximum number of agent handoffs
            handoff_strategy: Strategy for agent handoffs
        """
        self.config = config
        self.enable_swarm = enable_swarm
        self.max_handoffs = max_handoffs
        self.handoff_strategy = handoff_strategy

        # Initialize store for agent memory
        self.store = InMemoryStore()

        # Define agent expertise profiles
        self.agent_expertise = self._define_agent_expertise()

        # Build swarm workflow
        self.swarm = self._build_swarm() if enable_swarm else None

        logger.info(
            "swarm_orchestrator_initialized",
            enable_swarm=enable_swarm,
            max_handoffs=max_handoffs,
            strategy=handoff_strategy,
        )

    def _define_agent_expertise(self) -> dict[str, AgentExpertise]:
        """Define expertise profiles for each agent."""
        from ...providers.pydantic_ai_models import PydanticAIModelFactory

        # Create models once for efficiency
        try:
            pre_val_model = PydanticAIModelFactory.create_from_config(
                self.config, model_name=self.config.get_model_for_agent("pre_validator")
            )
            gen_model = PydanticAIModelFactory.create_from_config(
                self.config, model_name=self.config.get_model_for_agent("generator")
            )
            post_val_model = PydanticAIModelFactory.create_from_config(
                self.config,
                model_name=self.config.get_model_for_agent("post_validator"),
            )
            context_model = PydanticAIModelFactory.create_from_config(
                self.config,
                model_name=self.config.get_model_for_agent("context_enrichment"),
            )
            quality_model = PydanticAIModelFactory.create_from_config(
                self.config,
                model_name=self.config.get_model_for_agent("memorization_quality"),
            )
        except Exception as e:
            logger.warning("failed_to_create_swarm_models", error=str(e))
            return {}

        return {
            "pre_validator": AgentExpertise(
                name="pre_validator",
                expertise=["structure validation", "format checking", "note parsing"],
                handoff_agents=["generator"],
                model=pre_val_model,
                description="Validates note structure and format before processing",
            ),
            "generator": AgentExpertise(
                name="generator",
                expertise=["card generation", "APF format", "content creation"],
                handoff_agents=["post_validator", "context_enrichment"],
                model=gen_model,
                description="Creates Anki cards from note content",
            ),
            "post_validator": AgentExpertise(
                name="post_validator",
                expertise=["quality validation", "error detection", "correction"],
                handoff_agents=["generator", "context_enrichment"],
                model=post_val_model,
                description="Validates card quality and suggests improvements",
            ),
            "context_enrichment": AgentExpertise(
                name="context_enrichment",
                expertise=["examples", "mnemonics", "explanations", "enhancement"],
                handoff_agents=["memorization_quality"],
                model=context_model,
                description="Enhances cards with examples and learning aids",
            ),
            "memorization_quality": AgentExpertise(
                name="memorization_quality",
                expertise=["SRS effectiveness", "learning theory", "cognitive load"],
                handoff_agents=[],  # Terminal agent
                model=quality_model,
                description="Evaluates cards for spaced repetition effectiveness",
            ),
        }

    def _build_swarm(self) -> Swarm:
        """Build the swarm with agent handoff capabilities."""
        if not self.enable_swarm:
            return None

        agents = []
        for expertise in self.agent_expertise.values():
            # Create agent with handoff tools
            handoff_tools = []
            for handoff_agent in expertise.handoff_agents:
                handoff_tools.append(self._create_handoff_tool(handoff_agent))

            # Create agent using modern langchain.agents API
            agent = create_agent(
                expertise.model,
                tools=handoff_tools,
                system_prompt=self._create_agent_prompt(expertise),
            )
            agents.append(agent)

        return Swarm(
            agents=agents,
            max_handoffs=self.max_handoffs,
            handoff_strategy=self.handoff_strategy,
        )

    def _create_handoff_tool(self, target_agent: str):
        """Create a tool for handing off to another agent."""

        def handoff_tool(state: PipelineState, reason: str) -> Command:
            """Hand off control to another agent."""
            return Command(
                goto=target_agent,
                update={
                    "current_agent": target_agent,
                    "handoff_reason": reason,
                    "handoff_count": state.get("handoff_count", 0) + 1,
                },
            )

        handoff_tool.__name__ = f"handoff_to_{target_agent}"
        return handoff_tool

    def _create_agent_prompt(self, expertise: AgentExpertise) -> str:
        """Create specialized prompt for agent based on expertise."""
        base_prompt = f"""You are a {expertise.name} agent specializing in: {', '.join(expertise.expertise)}.

Description: {expertise.description}

Your role is to excel in your areas of expertise. If you encounter a task that would be better
handled by another agent, use the appropriate handoff tool.

Available handoffs: {', '.join(expertise.handoff_agents) if expertise.handoff_agents else 'None (terminal agent)'}

Always focus on your core competencies and hand off when appropriate.
"""

        # Add expertise-specific instructions
        if "validation" in expertise.name:
            base_prompt += "\nFocus on detecting issues and providing clear feedback."
        elif "generator" in expertise.name:
            base_prompt += "\nFocus on creating high-quality, APF-compliant cards."
        elif "enrichment" in expertise.name:
            base_prompt += "\nFocus on adding valuable learning enhancements."
        elif "quality" in expertise.name:
            base_prompt += "\nFocus on SRS effectiveness and learning outcomes."

        return base_prompt

    async def process_note_with_swarm(
        self,
        note_content: str,
        metadata: dict,
        qa_pairs: list[dict],
        file_path: str | None = None,
    ) -> SwarmResult:
        """Process a note through the swarm orchestration.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: Q/A pairs to process
            file_path: Optional file path

        Returns:
            SwarmResult with execution details
        """
        start_time = time.time()

        if not self.enable_swarm or not self.swarm:
            # Fallback to traditional orchestration
            logger.info("swarm_disabled_fallback_to_traditional")
            return await self._fallback_processing(
                note_content, metadata, qa_pairs, file_path, start_time
            )

        # Initialize swarm state
        initial_state = self._create_swarm_state(
            note_content, metadata, qa_pairs, file_path
        )

        try:
            # Execute swarm workflow
            final_state = await self.swarm.ainvoke(initial_state)

            # Track handoffs
            handoffs = final_state.get("handoffs", [])
            final_agent = final_state.get("current_agent", "unknown")

            success = final_state.get("current_stage") == "complete"
            total_time = time.time() - start_time

            result = SwarmResult(
                final_agent=final_agent,
                handoffs=handoffs,
                total_time=total_time,
                success=success,
                final_state=final_state,
            )

            logger.info(
                "swarm_processing_complete",
                final_agent=final_agent,
                handoffs=len(handoffs),
                success=success,
                total_time=total_time,
            )

            return result

        except Exception as e:
            logger.error("swarm_processing_failed", error=str(e))
            # Fallback on error
            return await self._fallback_processing(
                note_content, metadata, qa_pairs, file_path, start_time
            )

    def _create_swarm_state(
        self,
        note_content: str,
        metadata: dict,
        qa_pairs: list[dict],
        file_path: str | None,
    ) -> PipelineState:
        """Create initial state for swarm processing."""
        return {
            # Input data
            "note_content": note_content,
            "metadata_dict": metadata,
            "qa_pairs_dicts": qa_pairs,
            "file_path": file_path,
            # Swarm control
            "current_agent": "pre_validator",  # Start with validation
            "handoff_count": 0,
            "max_handoffs": self.max_handoffs,
            "handoffs": [],
            # Pipeline state
            "current_stage": "pre_validation",
            "step_count": 0,
            "errors": [],
            "messages": [],
            "start_time": time.time(),
        }

    async def _fallback_processing(
        self,
        note_content: str,
        metadata: dict,
        qa_pairs: list[dict],
        file_path: str | None,
        start_time: float,
    ) -> SwarmResult:
        """Fallback to traditional processing when swarm is disabled or fails."""
        from .orchestrator import LangGraphOrchestrator

        # Create traditional orchestrator
        traditional_orchestrator = LangGraphOrchestrator(self.config)

        # Convert to proper types for traditional processing
        from pathlib import Path

        from ...models import NoteMetadata, QAPair

        note_metadata = NoteMetadata(**metadata)
        qa_objects = [QAPair(**qa) for qa in qa_pairs]
        path_obj = Path(file_path) if file_path else None

        # Process traditionally
        result = await traditional_orchestrator.process_note(
            note_content=note_content,
            metadata=note_metadata,
            qa_pairs=qa_objects,
            file_path=path_obj,
        )

        # Convert to swarm result format
        return SwarmResult(
            final_agent="traditional_orchestrator",
            handoffs=[],
            total_time=time.time() - start_time,
            success=result.success,
            final_state={},  # Not available in traditional result
        )

    def get_swarm_stats(self) -> dict[str, Any]:
        """Get statistics about swarm performance."""
        if not self.swarm:
            return {"enabled": False}

        return {
            "enabled": True,
            "max_handoffs": self.max_handoffs,
            "handoff_strategy": self.handoff_strategy,
            "agents": list(self.agent_expertise.keys()),
            "expertise_areas": {
                name: agent.expertise for name, agent in self.agent_expertise.items()
            },
        }
