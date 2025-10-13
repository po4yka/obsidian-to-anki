"""Tests for content hash computation."""

from dataclasses import replace

from obsidian_anki_sync.apf.generator import compute_content_hash


class TestContentHash:
    """Validate that content hash captures all relevant sections."""

    def test_hash_changes_with_followups(self, sample_qa_pair, sample_metadata):
        """Modifying follow-ups should change the hash."""
        hash_original = compute_content_hash(sample_qa_pair, sample_metadata, "en")

        modified_pair = replace(sample_qa_pair, followups="New follow-up prompt")
        hash_modified = compute_content_hash(modified_pair, sample_metadata, "en")

        assert hash_original != hash_modified

    def test_hash_changes_with_references(self, sample_qa_pair, sample_metadata):
        """References contribute to hash."""
        hash_original = compute_content_hash(sample_qa_pair, sample_metadata, "en")

        modified_pair = replace(sample_qa_pair, references="https://example.com")
        hash_modified = compute_content_hash(modified_pair, sample_metadata, "en")

        assert hash_original != hash_modified

    def test_hash_differs_by_language(self, sample_qa_pair, sample_metadata):
        """Different language surfaces should yield distinct hashes."""
        hash_en = compute_content_hash(sample_qa_pair, sample_metadata, "en")
        hash_ru = compute_content_hash(sample_qa_pair, sample_metadata, "ru")

        assert hash_en != hash_ru
