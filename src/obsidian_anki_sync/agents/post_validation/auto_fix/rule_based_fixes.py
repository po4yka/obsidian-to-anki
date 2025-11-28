"""Rule-based fixes for APF card headers."""

import re

from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class RuleBasedHeaderFixer:
    """Attempts to fix card headers using rule-based transformations."""

    @staticmethod
    def apply_fixes(cards: list[GeneratedCard]) -> list[GeneratedCard] | None:
        """Attempt to fix card headers using rule-based transformations.

        Args:
            cards: Cards with potential header format issues

        Returns:
            Fixed cards if successful, None otherwise
        """
        fixed_cards = []
        any_fixes = False

        for card in cards:
            lines = card.apf_html.split("\n")
            if not lines:
                fixed_cards.append(card)
                continue

            # Find the card header line (should be after BEGIN_CARDS)
            header_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("<!-- Card "):
                    header_line_idx = i
                    break

            if header_line_idx is None:
                logger.warning(
                    "auto_fix_no_header_found",
                    slug=card.slug,
                )
                fixed_cards.append(card)
                continue

            original_header = lines[header_line_idx].strip()
            fixed_header = RuleBasedHeaderFixer._fix_header(original_header)

            if fixed_header != original_header:
                logger.info(
                    "auto_fix_header_corrected",
                    slug=card.slug,
                    original=original_header[:100],
                    fixed=fixed_header[:100],
                )
                lines[header_line_idx] = fixed_header
                fixed_html = "\n".join(lines)
                any_fixes = True

                fixed_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=fixed_html,
                    confidence=card.confidence,
                    content_hash=card.content_hash,
                )
                fixed_cards.append(fixed_card)
            else:
                # No fix needed or fix didn't help
                fixed_cards.append(card)

        if any_fixes:
            return fixed_cards
        return None

    @staticmethod
    def _fix_header(header: str) -> str:
        """Apply all header fixes to a single header line.

        Args:
            header: Original header string

        Returns:
            Fixed header string
        """
        fixed_header = header

        # Common fixes:
        # 1. Normalize spacing around pipes: "Card 1|slug:" -> "Card 1 | slug:"
        fixed_header = re.sub(r"\s*\|\s*", " | ", fixed_header)

        # 2. Fix CardType spacing and capitalization
        # Match variations like "CardType:Simple", "cardtype: Simple", "CardType :Simple"
        fixed_header = re.sub(
            r"[Cc]ard[Tt]ype\s*:\s*([Ss]imple|[Mm]issing|[Dd]raw)",
            lambda m: f"CardType: {str(m.group(1)).capitalize()}",
            fixed_header,
        )

        # 3. Normalize Tags format: ensure it's "Tags: " not "tags:" or "Tags :"
        fixed_header = re.sub(r"[Tt]ags\s*:", "Tags:", fixed_header)

        # 4. Fix slug format: convert underscores to hyphens, lowercase
        slug_match = re.search(r"slug:\s*([^\s|]+)", fixed_header)
        if slug_match:
            original_slug = slug_match.group(1)
            fixed_slug = original_slug.lower().replace("_", "-")
            # Remove any invalid characters
            fixed_slug = re.sub(r"[^a-z0-9-]", "-", fixed_slug)
            # Remove multiple consecutive hyphens
            fixed_slug = re.sub(r"-+", "-", fixed_slug)
            # Remove leading/trailing hyphens
            fixed_slug = fixed_slug.strip("-")
            fixed_header = fixed_header.replace(
                f"slug: {original_slug}", f"slug: {fixed_slug}"
            )

        # 5. Ensure tags are space-separated, not comma-separated
        tags_match = re.search(r"Tags:\s*([^>]+)", fixed_header)
        if tags_match:
            tags_str = tags_match.group(1).strip()
            # Replace commas with spaces
            fixed_tags = re.sub(r"\s*,\s*", " ", tags_str)
            # Normalize multiple spaces
            fixed_tags = re.sub(r"\s+", " ", fixed_tags)
            fixed_header = fixed_header.replace(
                f"Tags: {tags_str}", f"Tags: {fixed_tags}"
            )

        # 6. Ensure proper comment ending
        if not fixed_header.endswith("-->"):
            if fixed_header.endswith("->"):
                fixed_header += ">"
            elif fixed_header.endswith("-"):
                fixed_header += "->"
            else:
                fixed_header += " -->"

        # 7. Ensure proper comment beginning
        if not fixed_header.startswith("<!--"):
            fixed_header = "<!-- " + fixed_header.lstrip("<!")

        return fixed_header
