import pytest
from obsidian_anki_sync.agents.langgraph.state_models import (
    validate_metadata_dict,
    validate_qa_pairs_dicts,
    validate_autofix_dict,
)

def test_validate_metadata_dict_valid():
    data = {
        "id": "123",
        "title": "Test Note",
        "topic": "Test Topic",
        "tags": ["tag1"],
        "language": "en"
    }
    assert validate_metadata_dict(data) is True

def test_validate_metadata_dict_invalid():
    data = {
        "id": "123",
        # Missing title and topic
    }
    assert validate_metadata_dict(data) is False

def test_validate_qa_pairs_dicts_valid():
    data = [
        {
            "id": "1",
            "question": "Q1",
            "answer": "A1",
            "index": 1
        }
    ]
    assert validate_qa_pairs_dicts(data) is True

def test_validate_qa_pairs_dicts_invalid():
    data = [
        {
            "id": "1",
            "question": "Q1",
            # Missing answer
            "index": 1
        }
    ]
    assert validate_qa_pairs_dicts(data) is False

def test_validate_autofix_dict_valid():
    data = {
        "fixed": True,
        "changes": ["fix1"],
        "error": None
    }
    assert validate_autofix_dict(data) is True

def test_validate_autofix_dict_invalid():
    data = {
        "fixed": "not_boolean", # Invalid type
        "changes": ["fix1"],
        "error": None
    }
    assert validate_autofix_dict(data) is False
