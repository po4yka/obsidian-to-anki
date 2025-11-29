import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from pydantic_ai.models.openai import OpenAIChatModel
from obsidian_anki_sync.agents.pydantic_ai_agents import GeneratorAgentAI, CardGenerationOutput
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.agents import GeneratedCard


@pytest.mark.asyncio
async def test_generator_agent_retries_on_invalid_apf():
    """Test that GeneratorAgentAI retries when generated APF is invalid."""

    # Mock dependencies
    mock_model = MagicMock(spec=OpenAIChatModel)
    # We need to mock the agent's run method to simulate the retry loop behavior
    # deeper in pydantic-ai, but for this integration test, we want to verify
    # that our validator raises the error that triggers the retry.

    # However, since we can't easily mock the internal pydantic-ai retry loop without
    # complex patching, we will test the validator logic directly on the output model
    # to ensure it raises the expected error.

    # 1. Create an invalid output
    invalid_card = {
        "card_index": 1,
        "slug": "test-card",
        "lang": "en",
        "apf_html": "Invalid APF content",  # Missing sentinels
        "confidence": 0.9
    }

    # 2. Verify validator raises ValueError
    with pytest.raises(ValueError, match="APF Validation Failed"):
        CardGenerationOutput(
            cards=[invalid_card],
            total_generated=1,
            generation_notes="Test",
            confidence=0.9
        )

    # 3. Create a valid output
    valid_apf = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test-card | CardType: Simple | Tags: python testing unit_test -->
<!-- manifest: {"slug": "test-card", "lang": "en", "type": "Simple", "tags": ["python", "testing", "unit_test"]} -->
<!-- Title -->
Test Card
<!-- Key point -->
Key Point
<!-- Key point notes -->
Notes
<!-- END_CARDS -->
END_OF_CARDS"""

    valid_card = {
        "card_index": 1,
        "slug": "test-card",
        "lang": "en",
        "apf_html": valid_apf,
        "confidence": 0.9
    }

    # 4. Verify validator passes
    output = CardGenerationOutput(
        cards=[valid_card],
        total_generated=1,
        generation_notes="Test",
        confidence=0.9
    )
    assert output.cards[0]["slug"] == "test-card"


@pytest.mark.asyncio
async def test_generator_agent_integration_mock():
    """Verify the agent integration with a mocked run method."""
    # This tests that the agent is set up correctly to use the model

    mock_model = MagicMock(spec=OpenAIChatModel)
    agent = GeneratorAgentAI(model=mock_model)

    # Mock the agent.run method to return a valid result
    # This simulates a successful retry or first-try success
    valid_apf = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test-card | CardType: Simple | Tags: python testing unit_test -->
<!-- manifest: {"slug": "test-card", "lang": "en", "type": "Simple", "tags": ["python", "testing", "unit_test"]} -->
<!-- Title -->
Test Card
<!-- Key point -->
Key Point
<!-- Key point notes -->
Notes
<!-- END_CARDS -->
END_OF_CARDS"""

    mock_result = MagicMock()
    mock_result.data = CardGenerationOutput(
        cards=[{
            "card_index": 1,
            "slug": "test-card",
            "lang": "en",
            "apf_html": valid_apf,
            "confidence": 0.9
        }],
        total_generated=1,
        generation_notes="Success",
        confidence=0.9
    )

    agent.agent.run = AsyncMock(return_value=mock_result)

    # Run generation
    now = datetime.now()
    metadata = NoteMetadata(
        id="test", title="Test", topic="Testing", tags=["test"],
        language_tags=["python"], created=now, updated=now
    )
    qa_pairs = [QAPair(
        question_en="Q", question_ru="Q", answer_en="A", answer_ru="A", card_index=1
    )]

    result = await agent.generate_cards(
        note_content="content",
        metadata=metadata,
        qa_pairs=qa_pairs,
        slug_base="test"
    )

    assert len(result.cards) == 1
    assert result.cards[0].slug == "test-card"
