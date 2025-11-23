"""Tests for slug generator (UNIT-slug-01)."""

from obsidian_anki_sync.sync.slug_generator import create_manifest, generate_slug


class TestSlugGeneration:
    """Test slug generation (UNIT-slug-01)."""

    def test_basic_slug_generation(self) -> None:
        """Test basic slug generation."""
        slug, slug_base, hash6 = generate_slug(
            source_path="70-Kotlin/q-coroutines-basics.md",
            card_index=1,
            lang="en",
            existing_slugs=set(),
        )

        assert slug.endswith("-en")
        assert slug_base in slug
        assert "p01" in slug_base
        assert hash6 is None  # No collision

    def test_slug_sanitization(self) -> None:
        """Test slug sanitization of special characters."""
        slug, slug_base, hash6 = generate_slug(
            source_path="Test/Q&A File!@#.md",
            card_index=1,
            lang="en",
            existing_slugs=set(),
        )

        # Should sanitize special chars to dashes
        assert "@" not in slug
        assert "!" not in slug
        assert "#" not in slug
        # Should collapse multiple dashes
        assert "--" not in slug

    def test_directory_included_in_slug(self) -> None:
        """Slug should reflect directory segments to avoid collisions."""
        slug, slug_base, _ = generate_slug(
            source_path="40-Android/performance/q-startup.md",
            card_index=1,
            lang="en",
            existing_slugs=set(),
        )

        assert "40-android" in slug_base

    def test_slug_collision_resolution(self) -> None:
        """Test collision resolution with hash6."""
        existing = {"test-slug-p01-en"}

        slug, slug_base, hash6 = generate_slug(
            source_path="test-slug.md", card_index=1, lang="en", existing_slugs=existing
        )

        # Should add hash6 to resolve collision
        assert hash6 is not None
        assert len(hash6) == 6
        assert hash6 in slug_base
        assert hash6  # sanity check
        assert slug not in existing

    def test_slug_collision_suffix_is_stable(self) -> None:
        """Hash suffix should be stable across calls regardless of content changes."""
        existing = {"duplicate-note-p01-en"}
        slug1, _, hash1 = generate_slug(
            source_path="duplicate-note.md",
            card_index=1,
            lang="en",
            existing_slugs=existing,
        )
        # Repeat with the same inputs (hash should be identical)
        slug2, _, hash2 = generate_slug(
            source_path="duplicate-note.md",
            card_index=1,
            lang="en",
            existing_slugs=existing,
        )

        assert slug1 == slug2
        assert hash1 == hash2

    def test_language_suffix(self) -> None:
        """Test language suffix added to slug."""
        slug_en, _, _ = generate_slug("test.md", 1, "en", set())
        slug_ru, _, _ = generate_slug("test.md", 1, "ru", set())

        assert slug_en.endswith("-en")
        assert slug_ru.endswith("-ru")
        assert slug_en != slug_ru

    def test_card_index_formatting(self) -> None:
        """Test card index zero-padding."""
        slug1, base1, _ = generate_slug("test.md", 1, "en", set())
        slug9, base9, _ = generate_slug("test.md", 9, "en", set())
        slug10, base10, _ = generate_slug("test.md", 10, "en", set())

        assert "p01" in base1
        assert "p09" in base9
        assert "p10" in base10

    def test_slug_determinism(self) -> None:
        """Test slug generation is deterministic."""
        slug1, base1, hash1 = generate_slug("test.md", 1, "en", set())
        slug2, base2, hash2 = generate_slug("test.md", 1, "en", set())

        assert slug1 == slug2
        assert base1 == base2
        assert hash1 == hash2

    def test_slug_prefers_note_tail(self) -> None:
        """Long slugs should preserve the note-specific tail segment."""
        slug, slug_base, _ = generate_slug(
            source_path="30-System-Design/q-microservices-vs-monolith-architecture.md",
            card_index=1,
            lang="en",
            existing_slugs=set(),
        )

        assert "architecture" in slug_base
        assert slug.endswith("-en")


class TestManifestCreation:
    """Test manifest generation."""

    def test_create_basic_manifest(self, sample_metadata) -> None:
        """Test creating a manifest."""
        manifest = create_manifest(
            slug="test-p01-en",
            slug_base="test-p01",
            lang="en",
            source_path="test.md",
            card_index=1,
            metadata=sample_metadata,
            guid="test-guid-123",
            hash6=None,
        )

        assert manifest.slug == "test-p01-en"
        assert manifest.slug_base == "test-p01"
        assert manifest.lang == "en"
        assert manifest.source_path == "test.md"
        assert manifest.source_anchor == "p01"
        assert manifest.note_id == sample_metadata.id
        assert manifest.card_index == 1

    def test_manifest_with_hash(self, sample_metadata) -> None:
        """Test manifest with collision hash."""
        manifest = create_manifest(
            slug="test-abc123-p01-en",
            slug_base="test-abc123-p01",
            lang="en",
            source_path="test.md",
            card_index=1,
            guid="test-guid-456",
            metadata=sample_metadata,
            hash6="abc123",
        )

        assert manifest.hash6 == "abc123"
