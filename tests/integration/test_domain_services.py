"""Integration tests for domain services."""

from pathlib import Path

from obsidian_anki_sync.domain.services.content_hash_service import ContentHashService
from obsidian_anki_sync.domain.services.slug_service import SlugService


class TestDomainServicesIntegration:
    """Test domain services working together."""

    def test_slug_service_and_hash_service_integration(self):
        """Test that slug service and hash service work together properly."""
        # Test data
        file_path = Path("/vault/notes/test-note.md")
        source_dir = "/vault"
        card_index = 1

        # Generate slug
        slug_base = SlugService.generate_slug_base(file_path, source_dir, card_index)
        full_slug = SlugService.generate_full_slug(slug_base, "en")

        # Generate content hash
        content = f"Test content for {full_slug}"
        content_hash = ContentHashService.compute_hash(content)

        # Verify slug format
        assert SlugService.validate_slug(full_slug)
        assert full_slug.endswith("-en")

        # Verify hash is consistent
        same_hash = ContentHashService.compute_hash(content)
        assert content_hash == same_hash

        # Verify hash changes with different content
        different_hash = ContentHashService.compute_hash(content + "modified")
        assert content_hash != different_hash

    def test_content_hash_different_algorithms(self):
        """Test content hashing with different algorithms."""
        content = "Test content for hashing"

        sha256_hash = ContentHashService.compute_hash(content, "sha256", 64)
        md5_hash = ContentHashService.compute_hash(content, "md5", 32)

        # Different algorithms should produce different results
        assert sha256_hash != md5_hash

        # Same algorithm should be consistent
        sha256_hash2 = ContentHashService.compute_hash(content, "sha256", 64)
        assert sha256_hash == sha256_hash2

    def test_slug_collision_resolution(self):
        """Test slug collision resolution."""
        base_slug = "test-note"

        # First slug should be clean
        slug1 = SlugService.generate_thread_safe_slug(base_slug, "en")
        assert slug1 == f"{base_slug}-en"

        # Second slug should get collision counter
        slug2 = SlugService.generate_thread_safe_slug(base_slug, "en")
        assert slug2 == f"{base_slug}-en-1"

        # Third slug should increment counter
        slug3 = SlugService.generate_thread_safe_slug(base_slug, "en")
        assert slug3 == f"{base_slug}-en-2"

        # Different language should not collide
        slug_ru = SlugService.generate_thread_safe_slug(base_slug, "ru")
        assert slug_ru == f"{base_slug}-ru"

        # Reset for clean state
        SlugService._counters.clear()

    def test_content_hash_normalization(self):
        """Test that content normalization produces consistent hashes."""
        # Different whitespace should produce same hash
        content1 = "Line 1\nLine 2\nLine 3"
        content2 = "Line 1\nLine 2\nLine 3\n"  # Extra newline
        content3 = "  Line 1\n  Line 2\n  Line 3  "  # Extra spaces

        hash1 = ContentHashService.compute_hash(content1)
        hash2 = ContentHashService.compute_hash(content2)
        hash3 = ContentHashService.compute_hash(content3)

        # Should be the same after normalization
        assert hash1 == hash2 == hash3

    def test_structured_data_hashing(self):
        """Test hashing of structured data (like metadata)."""
        metadata1 = {
            "topic": "Testing",
            "languages": ["en", "ru"],
            "difficulty": "medium",
        }

        metadata2 = {
            "topic": "Testing",
            "languages": ["en", "ru"],
            "difficulty": "medium",
        }

        metadata3 = {
            "topic": "Testing",
            "languages": ["ru", "en"],  # Different order
            "difficulty": "medium",
        }

        hash1 = ContentHashService.compute_structured_hash(metadata1)
        hash2 = ContentHashService.compute_structured_hash(metadata2)
        hash3 = ContentHashService.compute_structured_hash(metadata3)

        # Same data should hash the same
        assert hash1 == hash2

        # List order should not matter (normalized)
        assert hash1 == hash3

    def test_slug_language_extraction(self):
        """Test extracting language from slug."""
        # Test valid slugs
        assert SlugService.extract_language_from_slug("test-note-en") == "en"
        assert SlugService.extract_language_from_slug("complex-slug-ru-1") == "ru"
        assert SlugService.extract_language_from_slug("no-language") is None
        assert SlugService.extract_language_from_slug("invalid-lang-xyz") is None

    def test_slug_base_extraction(self):
        """Test extracting base slug without language."""
        assert SlugService.get_base_slug("test-note-en") == "test-note"
        assert SlugService.get_base_slug("complex-slug-ru-1") == "complex-slug"
        assert SlugService.get_base_slug("no-suffix") == "no-suffix"

    def test_cache_functionality(self):
        """Test that hash caching works correctly."""
        content = "Cache test content"

        # First call should compute hash
        hash1 = ContentHashService.compute_hash(content)
        initial_cache_size = ContentHashService.get_cache_size()

        # Second call should use cache
        hash2 = ContentHashService.compute_hash(content)
        cached_size = ContentHashService.get_cache_size()

        # Results should be identical
        assert hash1 == hash2

        # Cache size should be >= initial (may have grown)
        assert cached_size >= initial_cache_size

        # Clear cache
        ContentHashService.clear_cache()
        assert ContentHashService.get_cache_size() == 0
