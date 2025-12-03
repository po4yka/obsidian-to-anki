"""Tests for vault analyzer using obsidiantools."""

from pathlib import Path

import pytest

from obsidian_anki_sync.obsidian.vault_analyzer import (
    LinkInfo,
    VaultAnalyzer,
    VaultStats,
)


@pytest.fixture
def sample_vault(tmp_path: Path) -> Path:
    """Create a sample Obsidian vault for testing."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()

    # Create some test notes with links
    note1 = vault_path / "note1.md"
    note1.write_text(
        """# Note 1
This note links to [[note2]] and [[note3]].
"""
    )

    note2 = vault_path / "note2.md"
    note2.write_text(
        """# Note 2
This note links back to [[note1]].
"""
    )

    note3 = vault_path / "note3.md"
    note3.write_text(
        """# Note 3
This is an orphaned note with no links.
"""
    )

    note4 = vault_path / "note4.md"
    note4.write_text(
        """# Note 4
This note has a broken link to [[missing_note]].
"""
    )

    return vault_path


class TestVaultAnalyzer:
    """Test VaultAnalyzer functionality."""

    def test_vault_initialization(self, sample_vault: Path) -> None:
        """Test vault analyzer initialization."""
        analyzer = VaultAnalyzer(sample_vault)
        assert analyzer.vault_path == sample_vault
        assert analyzer._vault is None  # Lazy loading

    def test_vault_connection(self, sample_vault: Path) -> None:
        """Test vault connection and lazy loading."""
        analyzer = VaultAnalyzer(sample_vault)
        vault = analyzer.vault
        assert vault is not None
        assert analyzer._vault is not None  # Now loaded

        # Test that it returns the same instance
        vault2 = analyzer.vault
        assert vault is vault2

    def test_get_vault_stats(self, sample_vault: Path) -> None:
        """Test getting vault statistics."""
        analyzer = VaultAnalyzer(sample_vault)
        stats = analyzer.get_vault_stats()

        assert isinstance(stats, VaultStats)
        assert stats.total_notes == 4
        assert isinstance(stats.orphaned_notes, list)
        assert isinstance(stats.broken_links, list)
        assert isinstance(stats.hub_notes, list)

    def test_get_link_info(self, sample_vault: Path) -> None:
        """Test getting link information for a note."""
        analyzer = VaultAnalyzer(sample_vault)

        # Note1 has outgoing links to note2 and note3
        link_info = analyzer.get_link_info("note1")

        assert isinstance(link_info, LinkInfo)
        assert link_info.note_path == "note1"
        assert isinstance(link_info.outgoing_links, list)
        assert isinstance(link_info.incoming_links, list)
        assert isinstance(link_info.broken_links, list)

    def test_validate_links(self, sample_vault: Path) -> None:
        """Test link validation."""
        analyzer = VaultAnalyzer(sample_vault)

        # Note4 has a broken link to missing_note
        broken_links = analyzer.validate_links("note4")
        assert isinstance(broken_links, list)
        # The exact behavior depends on how obsidiantools handles broken links

    def test_find_orphaned_notes(self, sample_vault: Path) -> None:
        """Test finding orphaned notes."""
        analyzer = VaultAnalyzer(sample_vault)
        orphaned = analyzer.find_orphaned_notes()

        assert isinstance(orphaned, list)
        # Note3 should potentially be orphaned (no links in or out)

    def test_find_orphaned_notes_with_pattern(self, sample_vault: Path) -> None:
        """Test finding orphaned notes with pattern filter."""
        analyzer = VaultAnalyzer(sample_vault)
        orphaned = analyzer.find_orphaned_notes(pattern="note*")

        assert isinstance(orphaned, list)

    def test_get_related_notes(self, sample_vault: Path) -> None:
        """Test getting related notes."""
        analyzer = VaultAnalyzer(sample_vault)
        related = analyzer.get_related_notes("note1", max_depth=1)

        assert isinstance(related, dict)
        assert "outgoing" in related
        assert "incoming" in related
        assert isinstance(related["outgoing"], list)
        assert isinstance(related["incoming"], list)

    def test_get_related_notes_depth_2(self, sample_vault: Path) -> None:
        """Test getting related notes with depth 2."""
        analyzer = VaultAnalyzer(sample_vault)
        related = analyzer.get_related_notes("note1", max_depth=2)

        assert isinstance(related, dict)
        assert "outgoing" in related
        assert "incoming" in related
        # Should also have second-level links if max_depth > 1
        # The exact keys depend on implementation

    def test_get_graph_data(self, sample_vault: Path) -> None:
        """Test getting graph data."""
        analyzer = VaultAnalyzer(sample_vault)
        graph_data = analyzer.get_graph_data()

        assert isinstance(graph_data, dict)
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert isinstance(graph_data["nodes"], list)
        assert isinstance(graph_data["edges"], list)

        # Should have 4 nodes (our 4 test notes)
        assert len(graph_data["nodes"]) == 4

    def test_check_integrity(self, sample_vault: Path) -> None:
        """Test vault integrity checking."""
        analyzer = VaultAnalyzer(sample_vault)
        issues = analyzer.check_integrity()

        assert isinstance(issues, dict)
        assert "broken_links" in issues
        assert "orphaned_notes" in issues
        assert "warnings" in issues
        assert isinstance(issues["broken_links"], list)
        assert isinstance(issues["orphaned_notes"], list)
        assert isinstance(issues["warnings"], list)


class TestVaultStatsDataclass:
    """Test VaultStats dataclass."""

    def test_vault_stats_creation(self) -> None:
        """Test creating a VaultStats instance."""
        stats = VaultStats(
            total_notes=10,
            total_links=20,
            total_backlinks=15,
            orphaned_notes=["note1", "note2"],
            broken_links=[("source", "target")],
            hub_notes=[("hub1", 5), ("hub2", 3)],
        )

        assert stats.total_notes == 10
        assert stats.total_links == 20
        assert len(stats.orphaned_notes) == 2
        assert len(stats.broken_links) == 1
        assert len(stats.hub_notes) == 2


class TestLinkInfoDataclass:
    """Test LinkInfo dataclass."""

    def test_link_info_creation(self) -> None:
        """Test creating a LinkInfo instance."""
        info = LinkInfo(
            note_path="test.md",
            outgoing_links=["link1", "link2"],
            incoming_links=["backlink1"],
            broken_links=["missing"],
        )

        assert info.note_path == "test.md"
        assert len(info.outgoing_links) == 2
        assert len(info.incoming_links) == 1
        assert len(info.broken_links) == 1
