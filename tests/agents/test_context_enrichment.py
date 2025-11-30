"""Normalization and prompt regression tests for context enrichment."""

import pytest

from obsidian_anki_sync.agents.context_enrichment_prompts import (
    CONTEXT_ENRICHMENT_PROMPT,
)
from obsidian_anki_sync.agents.models import (
    EnrichmentAddition,
    EnrichmentType,
    normalize_enrichment_label,
)


def test_enrichment_alias_is_normalized() -> None:
    """math_science alias should normalize to the related tag."""
    addition = EnrichmentAddition(
        enrichment_type="math_science",
        content="Newton's law with SI units.",
        rationale="Provide concrete units",
    )
    assert addition.enrichment_type is EnrichmentType.RELATED


def test_prompt_limits_vocabularies() -> None:
    """Prompt must enumerate the allowed enrichment types only."""
    assert "example/mnemonic/visual/related/practical" in CONTEXT_ENRICHMENT_PROMPT
    assert "math_science" not in CONTEXT_ENRICHMENT_PROMPT


def test_strict_normalization_rejects_unknown_alias() -> None:
    """Helper should raise when alias mapping is disabled for coverage."""
    with pytest.raises(ValueError):
        normalize_enrichment_label("math_science", apply_aliases=False)
