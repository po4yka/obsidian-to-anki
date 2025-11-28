"""Link and wikilink validator for Q&A notes."""

import re
from pathlib import Path
from typing import Any

from thefuzz import fuzz, process

from .base import BaseValidator, Severity


class LinkValidator(BaseValidator):
    """Validates internal links and wikilinks in Q&A notes.

    Checks that:
    - All wikilinks resolve to existing files
    - At least one concept link exists
    - Related field links are valid
    - No self-referential links exist

    Uses fuzzy matching to suggest corrections for broken links.
    """

    def __init__(
        self,
        content: str,
        frontmatter: dict[str, Any],
        filepath: str,
        vault_root: Path,
    ) -> None:
        """Initialize link validator.

        Args:
            content: Full note content
            frontmatter: Parsed YAML frontmatter
            filepath: Path to the note file
            vault_root: Root path of the vault for resolving links
        """
        super().__init__(content, frontmatter, filepath)
        self.vault_root = vault_root
        self.all_note_files = self._index_vault_files()

    def _index_vault_files(self) -> set[str]:
        """Index all markdown files in the vault.

        Returns:
            Set of note names (without .md extension)
        """
        files: set[str] = set()
        for md_file in self.vault_root.rglob("*.md"):
            # Store filename without extension for wikilink matching
            files.add(md_file.stem)
        return files

    def validate(self) -> list:
        """Perform link validation."""
        self._check_concept_links()
        self._check_wikilink_resolution()
        self._check_related_field_links()
        self._check_self_links()

        return self.issues

    def _check_concept_links(self) -> None:
        """Check for at least one concept link [[c-*]]."""
        concept_pattern = r"\[\[c-[a-z0-9-]+\]\]"

        if not re.search(concept_pattern, self.content):
            self.add_issue(
                Severity.WARNING,
                "No concept links found. Add at least one [[c-concept-name]] link "
                "in content body.",
            )
        else:
            # Count concept links
            matches = re.findall(concept_pattern, self.content)
            self.add_passed(f"Concept links present ({len(matches)} found)")

    def _check_wikilink_resolution(self) -> None:
        """Check all wikilinks resolve to existing files with fuzzy suggestions."""
        # Find all wikilinks [[note-name]]
        wikilink_pattern = r"\[\[([^\]]+)\]\]"
        wikilinks = re.findall(wikilink_pattern, self.content)

        if not wikilinks:
            self.add_passed("No wikilinks to validate")
            return

        broken_links: list[str] = []
        all_notes_list = list(self.all_note_files)

        for link in wikilinks:
            # Handle links with display text [[note|display]]
            note_name = link.split("|")[0].strip()

            # Check if note exists
            if note_name not in self.all_note_files:
                broken_links.append(note_name)

                # Find similar note names using fuzzy matching
                matches = process.extract(
                    note_name,
                    all_notes_list,
                    scorer=fuzz.ratio,
                    limit=3,
                )

                # If reasonable match found (70%+ similarity)
                if matches and matches[0][1] >= 70:
                    suggested_fix = matches[0][0]
                    similarity = matches[0][1]

                    self.add_issue(
                        Severity.ERROR,
                        f"Broken link '{note_name}'. Did you mean '{suggested_fix}'? "
                        f"({similarity}% match)",
                    )

                    # Add auto-fix for matches 75%+ (covers topic migration cases)
                    self.add_fix(
                        f"Replace broken link '{note_name}' with '{suggested_fix}'",
                        lambda content, fm, nl=note_name, sf=suggested_fix: (
                            self._fix_broken_wikilink(content, fm, nl, sf)
                        ),
                        Severity.ERROR,
                        safe=False,  # User should review before applying
                    )
                elif matches:
                    # Show multiple suggestions for lower confidence
                    suggestions = [f"{m[0]} ({m[1]}%)" for m in matches[:3]]
                    self.add_issue(
                        Severity.ERROR,
                        f"Broken link '{note_name}'. "
                        f"Similar notes: {', '.join(suggestions)}",
                    )

        if broken_links and not any(self.fixes):  # No auto-fixes suggested
            self.add_issue(
                Severity.ERROR,
                f"Broken wikilinks found: {', '.join(broken_links[:5])}"
                f"{'...' if len(broken_links) > 5 else ''}",
            )
        elif not broken_links:
            self.add_passed(f"All wikilinks resolve ({len(wikilinks)} checked)")

    def _fix_broken_wikilink(
        self,
        content: str,
        frontmatter: dict[str, Any],
        broken_link: str,
        suggested_link: str,
    ) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Replace broken wikilink with suggested link."""
        new_content = content

        # Replace [[broken-link]] with [[suggested-link]]
        new_content = new_content.replace(f"[[{broken_link}]]", f"[[{suggested_link}]]")

        # Also replace with display text: [[broken-link|text]] with [[suggested-link|text]]
        new_content = re.sub(
            rf"\[\[{re.escape(broken_link)}\|([^\]]+)\]\]",
            rf"[[{suggested_link}|\1]]",
            new_content,
        )

        return new_content, frontmatter

    def _check_related_field_links(self) -> None:
        """Check YAML related field links exist."""
        if "related" not in self.frontmatter:
            return

        related = self.frontmatter["related"]

        if not isinstance(related, list):
            return

        broken_related: list[str] = []
        all_notes_list = list(self.all_note_files)

        for link in related:
            if isinstance(link, str) and link not in self.all_note_files:
                broken_related.append(link)

                # Find similar note names using fuzzy matching
                matches = process.extract(
                    link,
                    all_notes_list,
                    scorer=fuzz.ratio,
                    limit=3,
                )

                # Add auto-fix for reasonable matches (70%+)
                if matches and matches[0][1] >= 70:
                    suggested_fix = matches[0][0]
                    self.add_fix(
                        f"Replace related '{link}' with '{suggested_fix}'",
                        lambda content, fm, bl=link, sf=suggested_fix: (
                            self._fix_related_field_link(content, fm, bl, sf)
                        ),
                        Severity.ERROR,
                        safe=False,
                    )

        if broken_related:
            self.add_issue(
                Severity.ERROR,
                f"YAML related field contains non-existent notes: "
                f"{', '.join(broken_related)}",
            )
        else:
            self.add_passed(f"All related field links valid ({len(related)} checked)")

    def _fix_related_field_link(
        self,
        content: str,
        frontmatter: dict[str, Any],
        broken_link: str,
        suggested_link: str,
    ) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Replace broken link in YAML related field."""
        new_frontmatter = dict(frontmatter)
        if "related" in new_frontmatter and isinstance(
            new_frontmatter["related"], list
        ):
            new_frontmatter["related"] = [
                suggested_link if item == broken_link else item
                for item in new_frontmatter["related"]
            ]
        return content, new_frontmatter

    def _check_self_links(self) -> None:
        """Check for self-referential links."""
        # Get current note name
        current_note = Path(self.filepath).stem

        # Find all wikilinks
        wikilink_pattern = r"\[\[([^\]|]+)"
        wikilinks = re.findall(wikilink_pattern, self.content)

        self_links = [link for link in wikilinks if link.strip() == current_note]

        if self_links:
            self.add_issue(
                Severity.WARNING,
                f"Self-referential link found: [[{current_note}]]. Remove self-links.",
            )
            # Add auto-fix for self-referential links
            self.add_fix(
                f"Remove self-referential link: [[{current_note}]]",
                lambda content, fm, cn=current_note: self._fix_remove_self_links(
                    content, fm, cn
                ),
                Severity.WARNING,
                safe=True,
            )
        else:
            self.add_passed("No self-referential links")

    def _fix_remove_self_links(
        self, content: str, frontmatter: dict[str, Any], current_note: str
    ) -> tuple[str, dict[str, Any]]:
        """Auto-fix: Remove self-referential links."""
        # Remove [[current-note]] and [[current-note|display]]
        new_content = content
        new_content = re.sub(rf"\[\[{re.escape(current_note)}\]\]", "", new_content)
        new_content = re.sub(
            rf"\[\[{re.escape(current_note)}\|[^\]]+\]\]", "", new_content
        )

        # Clean up empty bullets and extra spaces
        new_content = re.sub(r"- \s*\n", "", new_content)
        new_content = re.sub(r"\n\n\n+", "\n\n", new_content)

        return new_content, frontmatter
