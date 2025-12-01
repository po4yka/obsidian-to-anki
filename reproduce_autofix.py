from unittest.mock import MagicMock

import pytest

from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.agents.post_validation.validator import PostValidatorAgent


@pytest.fixture()
def mock_ollama_client():
    client = MagicMock()
    # Mock response for _llm_based_fix
    client.generate_json.return_value = {
        "corrected_cards": [
            {
                "card_index": 1,
                "slug": "test-slug",
                "lang": "en",
                "apf_html": "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->\n<!-- Card 1 | slug: test-slug | CardType: Simple | Tags: tag -->\n<!-- Title -->\n<p>Corrected Title</p>\n<!-- END_CARDS -->\nEND_OF_CARDS",
                "confidence": 0.9,
            }
        ]
    }
    return client


@pytest.fixture()
def validator(mock_ollama_client):
    return PostValidatorAgent(ollama_client=mock_ollama_client)


def test_classify_errors_html_syntax(validator):
    error_details = "HTML validation failed: Tag 'div' is not closed"
    classification = validator._classify_errors(error_details)
    # Currently expects 'unknown' or maybe 'template_content' depending on implementation
    # We want it to be 'html_syntax' eventually
    print(f"HTML Syntax Classification: {classification}")


def test_classify_errors_apf_sentinel(validator):
    error_details = "Missing '<!-- END_CARDS -->'"
    classification = validator._classify_errors(error_details)
    print(f"APF Sentinel Classification: {classification}")


def test_classify_errors_section_header(validator):
    error_details = "Missing '<!-- Title -->'"
    classification = validator._classify_errors(error_details)
    print(f"Section Header Classification: {classification}")


def test_attempt_auto_fix_html_syntax(validator):
    cards = [
        GeneratedCard(
            card_index=1,
            slug="test-slug",
            lang="en",
            apf_html="<div>Unclosed div",
            confidence=0.8,
        )
    ]
    error_details = "HTML validation failed: Tag 'div' is not closed"

    # This should trigger LLM fix if classified correctly or falls back to LLM
    fixed_cards = validator.attempt_auto_fix(cards, error_details)

    if fixed_cards:
        print("HTML Syntax Fix: Success")
    else:
        print("HTML Syntax Fix: Failed")


if __name__ == "__main__":
    # Manually run if executed as script
    m_client = MagicMock()
    m_client.generate_json.return_value = {
        "corrected_cards": [
            {
                "card_index": 1,
                "slug": "test-slug",
                "lang": "en",
                "apf_html": "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->\n<!-- Card 1 | slug: test-slug | CardType: Simple | Tags: tag -->\n<!-- Title -->\n<p>Corrected Title</p>\n<!-- END_CARDS -->\nEND_OF_CARDS",
                "confidence": 0.9,
            }
        ]
    }
    v = PostValidatorAgent(ollama_client=m_client)

    print("--- Running Classify Tests ---")
    test_classify_errors_html_syntax(v)
    test_classify_errors_apf_sentinel(v)
    test_classify_errors_section_header(v)

    print("\n--- Running Auto Fix Tests ---")
    test_attempt_auto_fix_html_syntax(v)
