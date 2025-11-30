import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx

# Add src to path
sys.path.append("src")

from obsidian_anki_sync.agents.parser_repair import ParserRepairAgent
from obsidian_anki_sync.providers.anthropic import AnthropicProvider
from obsidian_anki_sync.providers.openai import OpenAIProvider


async def test_openai_async():
    print("Testing OpenAIProvider.generate_async...")
    provider = OpenAIProvider(api_key="dummy")

    # Mock async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Async response"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "gpt-4",
    }
    mock_response.raise_for_status = MagicMock()

    provider.async_client.post = AsyncMock(return_value=mock_response)

    result = await provider.generate_async(model="gpt-4", prompt="test")
    print(f"Result: {result}")
    assert result["response"] == "Async response"
    print("OpenAIProvider.generate_async passed.")

    print("Testing OpenAIProvider.generate_json_async...")
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": '{"key": "value"}'}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "gpt-4",
    }
    result_json = await provider.generate_json_async(model="gpt-4", prompt="test json")
    print(f"Result JSON: {result_json}")
    assert result_json["key"] == "value"
    print("OpenAIProvider.generate_json_async passed.")


async def test_anthropic_async():
    print("Testing AnthropicProvider.generate_async...")
    provider = AnthropicProvider(api_key="dummy")

    # Mock async_client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "Async Claude response"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "model": "claude-3-opus",
        "stop_reason": "end_turn",
    }
    mock_response.raise_for_status = MagicMock()

    provider.async_client.post = AsyncMock(return_value=mock_response)

    result = await provider.generate_async(model="claude-3-opus", prompt="test")
    print(f"Result: {result}")
    assert result["response"] == "Async Claude response"
    print("AnthropicProvider.generate_async passed.")


async def test_parser_repair_async():
    print("Testing ParserRepairAgent.analyze_and_correct_proactively_async...")

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.generate_json_async = AsyncMock(
        return_value={
            "quality_score": 0.9,
            "issues_found": [],
            "needs_correction": False,
            "confidence": 1.0,
        }
    )

    agent = ParserRepairAgent(ollama_client=mock_provider)

    result = await agent.analyze_and_correct_proactively_async(content="test content")
    print(f"Result: {result}")
    assert result.quality_score == 0.9
    assert not result.needs_correction
    print("ParserRepairAgent.analyze_and_correct_proactively_async passed.")


async def main():
    await test_openai_async()
    await test_anthropic_async()
    await test_parser_repair_async()
    print("\nAll async tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
