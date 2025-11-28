"""Tool Calling Generator Agent for APF card generation.

Specialized tool calling agent for generating APF cards with parallel tool support.
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from ...models import GeneratedCard, GenerationResult, NoteMetadata, QAPair
from ...utils.logging import get_logger
from .base import LangChainAgentResult
from .tool_calling_agent import ToolCallingAgent

logger = get_logger(__name__)


class ToolCallingGeneratorAgent:
    """Tool Calling Generator Agent for APF card generation.

    Uses tool calling agent with specialized tools for card generation,
    supporting parallel tool execution for efficient card creation.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: list[BaseTool] | None = None,
        enable_parallel_tools: bool = True,
        temperature: float = 0.3,
    ):
        """Initialize tool calling generator agent.

        Args:
            model: Language model supporting tool calling
            tools: Custom tools (will use defaults if None)
            enable_parallel_tools: Enable parallel tool execution
            temperature: Sampling temperature for creativity
        """
        if tools is None:
            from .tools import get_tools_for_agent

            tools = get_tools_for_agent("generator")

        self.agent = ToolCallingAgent(
            model=model,
            tools=tools,
            agent_type="generator",
            temperature=temperature,
            enable_parallel_tool_calls=enable_parallel_tools,
        )

        logger.info("tool_calling_generator_initialized")

    async def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> GenerationResult:
        """Generate APF cards using tool calling agent.

        Args:
            note_content: Full note content
            metadata: Note metadata
            qa_pairs: Q/A pairs to convert
            slug_base: Base slug for cards

        Returns:
            GenerationResult with cards
        """
        logger.info(
            "tool_calling_generation_start",
            title=metadata.title,
            qa_count=len(qa_pairs),
        )

        # Prepare input for the agent
        input_data = {
            "task": "generate_cards",
            "note_content": note_content,
            "metadata": metadata.model_dump(),
            "qa_pairs": [qa.model_dump() for qa in qa_pairs],
            "slug_base": slug_base,
        }

        # Run the agent
        result = await self.agent.run(input_data)

        # Process the result into GenerationResult
        return self._process_generation_result(result, metadata)

    def _process_generation_result(
        self,
        agent_result: LangChainAgentResult,
        metadata: NoteMetadata,
    ) -> GenerationResult:
        """Process agent result into GenerationResult.

        Args:
            agent_result: Result from tool calling agent
            metadata: Original note metadata

        Returns:
            Processed GenerationResult
        """
        if not agent_result.success:
            return GenerationResult(
                cards=[],
                total_cards=0,
                errors=[agent_result.reasoning],
                warnings=agent_result.warnings,
            )

        # Extract generated cards from agent output
        # This would parse the APF content from the agent's response
        cards = self._extract_cards_from_output(agent_result.reasoning)

        logger.info(
            "tool_calling_generation_completed",
            total_cards=len(cards),
            confidence=agent_result.confidence,
        )

        return GenerationResult(
            cards=cards,
            total_cards=len(cards),
            errors=[],
            warnings=agent_result.warnings,
        )

    def _extract_cards_from_output(self, output: str) -> list[GeneratedCard]:
        """Extract GeneratedCard objects from agent output.

        Args:
            output: Agent output containing APF content

        Returns:
            List of GeneratedCard objects
        """
        cards = []

        # Parse APF format from output
        # This is a simplified implementation - in practice, you'd want
        # more robust APF parsing
        import re

        # Find card sections
        card_pattern = r"<!-- Card (\d+) \| slug: ([^|]+) \| CardType: ([^|]+) \| Tags: ([^-->]+) -->"
        matches = re.findall(card_pattern, output, re.MULTILINE)

        for match in matches:
            card_num, slug, card_type, tags_str = match

            # Extract title and content (simplified)
            # In practice, this would need more sophisticated parsing
            card_content = self._extract_card_content(output, int(card_num))

            if card_content:
                card = GeneratedCard(
                    front=card_content.get("front", ""),
                    back=card_content.get("back", ""),
                    card_type=card_type,
                    tags=tags_str.strip().split(),
                    slug=slug.strip(),
                    metadata={},
                )
                cards.append(card)

        return cards

    def _extract_card_content(
        self, output: str, card_number: int
    ) -> dict[str, str] | None:
        """Extract content for a specific card number.

        Args:
            output: Full agent output
            card_number: Card number to extract

        Returns:
            Dictionary with front/back content
        """
        # This is a simplified implementation
        # In practice, you'd need proper APF parsing logic
        lines = output.split("\n")
        card_start = None

        for i, line in enumerate(lines):
            if f"<!-- Card {card_number} |" in line:
                card_start = i
                break

        if card_start is None:
            return None

        # Extract content until next card or end
        content_lines = []
        for line in lines[card_start + 1 :]:
            if line.startswith("<!-- Card ") and "slug:" in line:
                break
            content_lines.append(line)

        content = "\n".join(content_lines).strip()

        # Simple front/back extraction (very simplified)
        if "<!-- Title -->" in content:
            parts = content.split("<!-- Title -->", 1)
            if len(parts) > 1:
                title_part = parts[1].split("<!--", 1)[0].strip()
                back_part = ""
                if "<!-- Key point -->" in content:
                    back_parts = content.split("<!-- Key point -->", 1)
                    if len(back_parts) > 1:
                        back_part = back_parts[1].split("<!--", 1)[0].strip()

                return {
                    "front": title_part,
                    "back": back_part,
                }

        return None
