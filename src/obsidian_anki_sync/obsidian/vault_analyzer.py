"""Vault analysis utilities using obsidiantools.

Provides high-level access to Obsidian vault structure, link graphs,
and metadata extraction for analytics, validation, and integrity checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from obsidiantools.api import Vault

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VaultStats:
    """Statistics about the vault."""

    total_notes: int
    total_links: int
    total_backlinks: int
    orphaned_notes: list[str]
    broken_links: list[tuple[str, str]]  # (source_note, broken_link)
    hub_notes: list[tuple[str, int]]  # (note_name, incoming_link_count)


@dataclass
class LinkInfo:
    """Information about a note's links."""

    note_path: str
    outgoing_links: list[str]
    incoming_links: list[str]
    broken_links: list[str]


class VaultAnalyzer:
    """Analyzer for Obsidian vaults using obsidiantools.

    Provides high-level access to vault structure, link graphs, and metadata
    for analytics, validation, and integrity checks.
    """

    def __init__(self, vault_path: Path) -> None:
        """Initialize the vault analyzer.

        Args:
            vault_path: Path to the Obsidian vault directory
        """
        self.vault_path = vault_path
        self._vault: Vault | None = None
        logger.debug("vault_analyzer_initialized", vault_path=str(vault_path))

    @property
    def vault(self) -> Vault:
        """Lazy-load the vault object."""
        if self._vault is None:
            self._vault = Vault(self.vault_path).connect()
            logger.info(
                "vault_connected",
                vault_path=str(self.vault_path),
                note_count=len(self._vault.md_file_index),
            )
        return self._vault

    def get_vault_stats(self) -> VaultStats:
        """Get comprehensive vault statistics.

        Returns:
            VaultStats object with vault metrics
        """
        logger.debug("calculating_vault_stats")

        # Get all notes
        all_notes = list(self.vault.md_file_index.keys())

        # Count total wikilinks and backlinks
        total_links = sum(len(links) for links in self.vault.wikilinks_index.values())
        total_backlinks = sum(
            len(backlinks) for backlinks in self.vault.backlinks_index.values()
        )

        # Find orphaned notes using obsidiantools built-in method
        orphaned = self.vault.isolated_notes

        # Find broken links using obsidiantools built-in method
        broken_notes = self.vault.nonexistent_notes
        broken = []
        for source_note, links in self.vault.wikilinks_index.items():
            for link in links:
                if link in broken_notes:
                    broken.append((source_note, link))

        # Find hub notes (most incoming links)
        # Count backlinks for each note
        backlink_counts = {}
        for note in all_notes:
            count = len(self.vault.get_backlinks(note))
            if count > 0:
                backlink_counts[note] = count
        hub_notes = sorted(backlink_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        stats = VaultStats(
            total_notes=len(all_notes),
            total_links=total_links,
            total_backlinks=total_backlinks,
            orphaned_notes=list(orphaned),
            broken_links=broken,
            hub_notes=list(hub_notes),
        )

        logger.info(
            "vault_stats_calculated",
            total_notes=stats.total_notes,
            orphaned_count=len(stats.orphaned_notes),
            broken_links_count=len(stats.broken_links),
        )

        return stats

    def get_link_info(self, note_name: str) -> LinkInfo:
        """Get link information for a specific note.

        Args:
            note_name: Name of the note (without .md extension)

        Returns:
            LinkInfo object with link details
        """
        # Get outgoing wikilinks using obsidiantools method
        outgoing = list(self.vault.get_wikilinks(note_name))

        # Get incoming links (backlinks) using obsidiantools method
        incoming = list(self.vault.get_backlinks(note_name))

        # Check for broken outgoing links
        all_notes = list(self.vault.md_file_index.keys())
        broken = [link for link in outgoing if link not in all_notes]

        return LinkInfo(
            note_path=note_name,
            outgoing_links=outgoing,
            incoming_links=incoming,
            broken_links=broken,
        )

    def validate_links(self, note_name: str) -> list[str]:
        """Validate that all links from a note exist.

        Args:
            note_name: Name of the note to validate

        Returns:
            List of broken link names
        """
        link_info = self.get_link_info(note_name)
        return link_info.broken_links

    def find_orphaned_notes(self, pattern: str | None = None) -> list[str]:
        """Find notes with no incoming or outgoing links.

        Args:
            pattern: Optional glob pattern to filter notes (e.g., 'q-*')

        Returns:
            List of orphaned note names
        """
        stats = self.get_vault_stats()
        orphaned = stats.orphaned_notes

        if pattern:
            import fnmatch

            orphaned = [note for note in orphaned if fnmatch.fnmatch(note, pattern)]

        return orphaned

    def get_related_notes(
        self, note_name: str, max_depth: int = 1
    ) -> dict[str, list[str]]:
        """Get related notes via links (outgoing and incoming).

        Args:
            note_name: Name of the note
            max_depth: Maximum depth for traversal (1 = direct links only)

        Returns:
            Dictionary with 'outgoing' and 'incoming' lists
        """
        link_info = self.get_link_info(note_name)

        related = {
            "outgoing": link_info.outgoing_links,
            "incoming": link_info.incoming_links,
        }

        # If max_depth > 1, recursively get related notes
        # (This is a simple implementation; more sophisticated graph traversal
        # could be added later)
        if max_depth > 1:
            # Get second-level related notes
            second_level_out = []
            for linked_note in link_info.outgoing_links:
                try:
                    second_info = self.get_link_info(linked_note)
                    second_level_out.extend(second_info.outgoing_links)
                except Exception:
                    pass  # Skip if note doesn't exist

            second_level_in = []
            for linking_note in link_info.incoming_links:
                try:
                    second_info = self.get_link_info(linking_note)
                    second_level_in.extend(second_info.incoming_links)
                except Exception:
                    pass

            related["outgoing_2nd_level"] = list(set(second_level_out))
            related["incoming_2nd_level"] = list(set(second_level_in))

        return related

    def get_graph_data(self) -> dict[str, Any]:
        """Get graph data for visualization.

        Returns:
            Dictionary with nodes and edges suitable for graph visualization
        """
        nodes = list(self.vault.md_file_index.keys())
        edges = []

        for source, targets in self.vault.wikilinks_index.items():
            for target in targets:
                edges.append({"source": source, "target": target})

        return {"nodes": nodes, "edges": edges}

    def check_integrity(self) -> dict[str, Any]:
        """Run comprehensive vault integrity checks.

        Returns:
            Dictionary with integrity check results
        """
        logger.info("running_vault_integrity_checks")

        stats = self.get_vault_stats()

        issues = {
            "broken_links": stats.broken_links,
            "orphaned_notes": stats.orphaned_notes,
            "warnings": [],
        }

        # Check for high orphan rate
        orphan_rate = len(stats.orphaned_notes) / stats.total_notes
        if orphan_rate > 0.1:  # More than 10% orphaned
            issues["warnings"].append(
                f"High orphan rate: {orphan_rate:.1%} of notes are orphaned"
            )

        # Check for broken links
        if stats.broken_links:
            issues["warnings"].append(f"Found {len(stats.broken_links)} broken links")

        logger.info(
            "vault_integrity_check_complete",
            broken_links=len(issues["broken_links"]),
            orphaned_notes=len(issues["orphaned_notes"]),
            warnings=len(issues["warnings"]),
        )

        return issues
