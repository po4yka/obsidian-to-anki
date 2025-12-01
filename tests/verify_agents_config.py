
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from obsidian_anki_sync.agents.pydantic_ai_agents import (
    PreValidatorAgentAI,
    GeneratorAgentAI,
    PostValidatorAgentAI,
    MemorizationQualityAgentAI,
    CardSplittingAgentAI
)
from obsidian_anki_sync.agents.pydantic.split_validator import SplitValidatorAgentAI
from obsidian_anki_sync.agents.pydantic.highlight import HighlightAgentAI

# Set dummy API key
os.environ["OPENAI_API_KEY"] = "sk-test"

async def verify_agents():
    print("Verifying agent configurations...")

    # Mock model
    model = OpenAIChatModel("gpt-4o-mini")
    # Mock the underlying client to avoid network calls
    model._client = AsyncMock()

    agents_to_test = [
        ("PreValidatorAgentAI", PreValidatorAgentAI),
        ("GeneratorAgentAI", GeneratorAgentAI),
        ("PostValidatorAgentAI", PostValidatorAgentAI),
        ("MemorizationQualityAgentAI", MemorizationQualityAgentAI),
        ("CardSplittingAgentAI", CardSplittingAgentAI),
        ("SplitValidatorAgentAI", SplitValidatorAgentAI),
        ("HighlightAgentAI", HighlightAgentAI),
    ]

    for name, agent_cls in agents_to_test:
        print(f"Testing {name}...")
        try:
            agent_instance = agent_cls(model=model)
            print(f"  Initialized {name} successfully.")

            # Verify internal agent is created
            if not hasattr(agent_instance, 'agent'):
                print(f"  ERROR: {name} has no 'agent' attribute.")
                continue

            if not isinstance(agent_instance.agent, Agent):
                print(f"  ERROR: {name}.agent is not an instance of pydantic_ai.Agent")
                continue

            print(f"  {name} verification PASSED.")

        except Exception as e:
            print(f"  ERROR initializing {name}: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(verify_agents())
