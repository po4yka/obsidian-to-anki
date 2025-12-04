#!/usr/bin/env python
"""Test card generation for a single note to see detailed errors."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.obsidian.parser import parse_note


async def test_note(note_path: str) -> None:
    """Test card generation for a single note."""


"""Test card generation for a single note to see detailed errors."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obsidian_anki_sync.agents.langgraph import LangGraphOrchestrator
from obsidian_anki_sync.config import load_config
from obsidian_anki_sync.obsidian.parser import parse_note


async def test_note(note_path: str) -> None:
    """Test card generation for a single note."""
    print(f"Testing: {note_path}")

    config = load_config()
    orchestrator = LangGraphOrchestrator(config)

    # Parse the note
    note_file = Path(note_path)
    metadata, qa_pairs = parse_note(note_file)

    if not qa_pairs:
        print("No QA pairs found in note")
        return

    print(f"Found {len(qa_pairs)} QA pairs")
    print(f"Metadata: {metadata}")

    # Run the pipeline
    pipeline_result = await orchestrator.process_note(
        note_content=note_file.read_text(),
        metadata=metadata,
        qa_pairs=qa_pairs,
        file_path=note_file,
    )

    print(f"\n=== Pipeline Result ===")
    print(f"Success: {pipeline_result.success}")

    if pipeline_result.pre_validation:
        print(f"\nPre-validation:")
        print(f"  Valid: {pipeline_result.pre_validation.is_valid}")
        print(f"  Error type: {pipeline_result.pre_validation.error_type}")
        print(f"  Details: {pipeline_result.pre_validation.error_details}")

    if pipeline_result.generation:
        print(f"\nGeneration:")
        print(f"  Cards: {pipeline_result.generation.total_cards}")

    if pipeline_result.post_validation:
        print(f"\nPost-validation:")
        print(f"  Valid: {pipeline_result.post_validation.is_valid}")
        print(f"  Error type: {pipeline_result.post_validation.error_type}")
        print(f"  Details: {pipeline_result.post_validation.error_details}")
        if pipeline_result.post_validation.corrected_cards:
            print(
                f"  Corrected cards: {len(pipeline_result.post_validation.corrected_cards)}"
            )
        if pipeline_result.post_validation.applied_changes:
            print(
                f"  Applied changes: {pipeline_result.post_validation.applied_changes}"
            )

    print(f"\nStage times: {pipeline_result.stage_times}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_note.py <path_to_note.md>")
        sys.exit(1)

    asyncio.run(test_note(sys.argv[1]))
    print(f"Success: {pipeline_result.success}")

    if pipeline_result.pre_validation:
        print(f"\nPre-validation:")
        print(f"  Valid: {pipeline_result.pre_validation.is_valid}")
        print(f"  Error type: {pipeline_result.pre_validation.error_type}")
        print(f"  Details: {pipeline_result.pre_validation.error_details}")

    if pipeline_result.generation:
        print(f"\nGeneration:")
        print(f"  Cards: {pipeline_result.generation.total_cards}")

    if pipeline_result.post_validation:
        print(f"\nPost-validation:")
        print(f"  Valid: {pipeline_result.post_validation.is_valid}")
        print(f"  Error type: {pipeline_result.post_validation.error_type}")
        print(f"  Details: {pipeline_result.post_validation.error_details}")
        if pipeline_result.post_validation.corrected_cards:
            print(
                f"  Corrected cards: {len(pipeline_result.post_validation.corrected_cards)}"
            )
        if pipeline_result.post_validation.applied_changes:
            print(
                f"  Applied changes: {pipeline_result.post_validation.applied_changes}"
            )

    print(f"\nStage times: {pipeline_result.stage_times}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_note.py <path_to_note.md>")
        sys.exit(1)

    asyncio.run(test_note(sys.argv[1]))
