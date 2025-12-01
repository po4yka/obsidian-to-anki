
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel

# Set dummy API key to avoid validation error if it checks env
os.environ["OPENAI_API_KEY"] = "sk-test"

class Deps(BaseModel):
    pass

class Output(BaseModel):
    foo: str

async def main():
    try:
        # Mock model - try without api_key arg
        model = OpenAIChatModel("gpt-4o-mini")

        print("Initializing Agent with output_retries=5...")
        agent = Agent(
            model=model,
            output_type=Output,
            system_prompt="test",
            output_retries=5
        )
        print("Agent initialized successfully.")

        print("Running agent...")
        # We need to mock the run execution because we don't want real network calls
        # But we want to see if arguments are passed correctly.
        # If we can't mock easily, we might just fail on network, which is fine
        # as long as we don't get the 'unexpected keyword argument' error BEFORE network.

        try:
            await agent.run("hello")
        except Exception as e:
            print(f"Run failed with: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"Caught exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
