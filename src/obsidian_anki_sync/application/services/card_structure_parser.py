"""Application service for parsing card structures.

This module provides application-level services for parsing
card structures, following Clean Architecture principles.
"""

import re

from obsidian_anki_sync.domain.interfaces.card_generation import (
    ICardStructureParser,
    ParsedCardStructure,
)
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class CardStructureParser(ICardStructureParser):
    """Application service for parsing card structures from APF HTML."""

    def parse_card_structure(self, apf_html: str) -> ParsedCardStructure:
        """Parse the structure of an English APF card for translation.

        Extracts the key components that define the card's logical structure:
        - Title (translated)
        - Key point code block (preserved as-is)
        - Key point notes (translated)
        - Other notes (translated)

        Args:
            apf_html: The complete APF HTML content

        Returns:
            ParsedCardStructure with the extracted components
        """
        # Extract title
        title_match = re.search(r"<!-- Title -->\s*\n(.*?)\n\s*\n", apf_html, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Extract key point code block (preserve exactly)
        key_point_code = None
        key_point_match = re.search(
            r"<!-- Key point \(code block / image\) -->\s*\n(.*?)\n\s*\n<!-- Key point notes -->",
            apf_html,
            re.DOTALL,
        )
        if key_point_match:
            code_block = key_point_match.group(1).strip()
            # Only extract if it's actually a code block
            if code_block.startswith("<pre><code") and code_block.endswith(
                "</code></pre>"
            ):
                key_point_code = code_block

        # Extract key point notes
        notes_match = re.search(
            r"<!-- Key point notes -->\s*\n<ul>\s*\n(.*?)\n\s*</ul>",
            apf_html,
            re.DOTALL,
        )
        key_point_notes = []
        if notes_match:
            ul_content = notes_match.group(1)
            # Extract individual list items
            li_matches = re.findall(r"<li>(.*?)</li>", ul_content, re.DOTALL)
            key_point_notes = [li.strip() for li in li_matches]

        # Extract other notes (optional)
        other_notes = None
        other_match = re.search(
            r"<!-- Other notes.*? -->\s*\n<ul>\s*\n(.*?)\n\s*</ul>", apf_html, re.DOTALL
        )
        if other_match:
            ul_content = other_match.group(1)
            li_matches = re.findall(r"<li>(.*?)</li>", ul_content, re.DOTALL)
            other_notes = [li.strip() for li in li_matches]

        return ParsedCardStructure(
            title=title,
            key_point_code=key_point_code,
            key_point_notes=key_point_notes,
            other_notes=other_notes,
        )
