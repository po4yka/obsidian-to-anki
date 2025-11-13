"""Tests for deterministic GUID generation."""

from obsidian_anki_sync.utils.guid import deterministic_guid


def test_guid_deterministic() -> None:
    guid1 = deterministic_guid(["note", "path", "1", "en"])
    guid2 = deterministic_guid(["note", "path", "1", "en"])
    assert guid1 == guid2


def test_guid_changes_with_input() -> None:
    guid1 = deterministic_guid(["note", "path", "1", "en"])
    guid2 = deterministic_guid(["note", "path", "1", "ru"])
    assert guid1 != guid2


def test_guid_length() -> None:
    guid = deterministic_guid(["note"], length=20)
    assert len(guid) == 20
