"""APF Renderer - Convert JSON CardSpec to APF v2.1 HTML.

This module provides deterministic conversion from structured JSON
card specifications to APF HTML format, eliminating the risk of
truncated or malformed output from LLMs.
"""

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.pydantic.card_schema import CardSection, CardSpec


class APFRenderer:
    """Convert JSON CardSpec to APF v2.1 HTML.

    This renderer produces deterministic, well-formed APF output
    from structured card specifications. All required sentinels
    and markers are guaranteed to be present.
    """

    PROMPT_VERSION = "apf-v2.1"

    def render(self, spec: "CardSpec") -> str:
        """Render a CardSpec to APF v2.1 HTML.

        Args:
            spec: CardSpec with all card data

        Returns:
            Complete APF HTML string with all required markers
        """
        parts = [
            f"<!-- PROMPT_VERSION: {self.PROMPT_VERSION} -->",
            "<!-- BEGIN_CARDS -->",
            "",
            self._render_card_header(spec),
            "",
            self._render_title(spec.front.title),
            "",
            self._render_key_point(spec.front),
            "",
            self._render_key_point_notes(spec.front.key_point_notes),
            "",
            self._render_other_notes(spec.front.other_notes),
            "",
            self._render_extra(spec.front.extra),
            "",
            self._render_manifest(spec),
            "<!-- END_CARDS -->",
            "END_OF_CARDS",
        ]
        return "\n".join(parts)

    def render_batch(self, specs: list["CardSpec"]) -> str:
        """Render multiple cards separated by card markers.

        Args:
            specs: List of CardSpec objects

        Returns:
            APF HTML with all cards
        """
        if not specs:
            return ""
        if len(specs) == 1:
            return self.render(specs[0])

        # For batch, each card is a complete APF block
        rendered_cards = [self.render(spec) for spec in specs]
        return "\n\n<!-- CARD_SEPARATOR -->\n\n".join(rendered_cards)

    def _render_card_header(self, spec: "CardSpec") -> str:
        """Render the card header comment."""
        tags_str = " ".join(spec.tags) if spec.tags else ""
        return (
            f"<!-- Card {spec.card_index} | slug: {spec.slug} | "
            f"CardType: {spec.card_type} | Tags: {tags_str} -->"
        )

    def _render_title(self, title: str) -> str:
        """Render the title section."""
        # Escape HTML in title but preserve intentional HTML tags
        return f"<!-- Title -->\n{title}"

    def _render_key_point(self, section: "CardSection") -> str:
        """Render the key point (code block) section."""
        header = "<!-- Key point (code block / image) -->"

        if not section.key_point_code:
            return header

        lang = section.key_point_code_lang or "plaintext"
        # Escape code content for HTML
        escaped_code = html.escape(section.key_point_code)

        return (
            f'{header}\n<pre><code class="language-{lang}">{escaped_code}</code></pre>'
        )

    def _render_key_point_notes(self, notes: list[str]) -> str:
        """Render the key point notes section."""
        header = "<!-- Key point notes -->"

        if not notes:
            return f"{header}\n<ul></ul>"

        items = []
        for note in notes:
            # Escape HTML but preserve intentional formatting
            items.append(f"<li>{note}</li>")

        items_str = "\n".join(items)
        return f"{header}\n<ul>\n{items_str}\n</ul>"

    def _render_other_notes(self, other_notes: str) -> str:
        """Render the other notes section."""
        header = "<!-- Other notes -->"

        if not other_notes:
            return header

        return f"{header}\n{other_notes}"

    def _render_extra(self, extra: str) -> str:
        """Render the extra section."""
        header = "<!-- Extra -->"

        if not extra:
            return header

        return f"{header}\n{extra}"

    def _render_manifest(self, spec: "CardSpec") -> str:
        """Render the manifest comment."""
        manifest = {
            "slug": spec.slug,
            "slug_base": spec.slug_base or spec.slug.rsplit("-", 2)[0],
            "lang": spec.lang,
            "type": spec.card_type,
            "tags": spec.tags,
            "guid": spec.guid,
        }

        # Add source info if available
        if spec.source_path:
            manifest["source_path"] = spec.source_path
        if spec.source_anchor:
            manifest["source_anchor"] = spec.source_anchor

        manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        return f"<!-- manifest: {manifest_json} -->"


class APFSentinelValidator:
    """Validate APF structure has all required sentinels.

    Used to verify that APF output (whether from renderer or LLM)
    contains all required structural markers.
    """

    REQUIRED_SENTINELS = [
        "<!-- PROMPT_VERSION:",
        "<!-- BEGIN_CARDS -->",
        "<!-- Card ",
        "<!-- Title -->",
        "<!-- Key point",
        "<!-- manifest:",
        "<!-- END_CARDS -->",
    ]

    def validate(self, apf_html: str) -> list[str]:
        """Check for missing sentinels in APF HTML.

        Args:
            apf_html: APF HTML content to validate

        Returns:
            List of missing sentinel markers (empty if all present)
        """
        if not apf_html:
            return self.REQUIRED_SENTINELS.copy()

        missing = []
        for sentinel in self.REQUIRED_SENTINELS:
            if sentinel not in apf_html:
                missing.append(sentinel)
        return missing

    def is_valid(self, apf_html: str) -> bool:
        """Check if APF HTML has all required sentinels.

        Args:
            apf_html: APF HTML content to validate

        Returns:
            True if all sentinels present, False otherwise
        """
        return len(self.validate(apf_html)) == 0
