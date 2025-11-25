"""Aggressive deterministic fixes as last resort."""

import json
import re

from ....utils.logging import get_logger
from ...models import GeneratedCard

logger = get_logger(__name__)


class AggressiveFixer:
    """Applies aggressive deterministic fixes as last resort.

    This class attempts to fix cards even when error details are unclear,
    by applying all known fixes regardless of error message.
    """

    @staticmethod
    def apply_fixes(
        cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Apply aggressive deterministic fixes to cards.

        Args:
            cards: Cards to fix
            error_details: Error description

        Returns:
            Fixed cards if any fixes applied, None otherwise
        """
        fixed_cards = []
        any_fixes = False

        for card in cards:
            fixed_html = card.apf_html
            card_fixed = False

            # Aggressively ensure all sentinels are present
            if "<!-- PROMPT_VERSION:" not in fixed_html:
                fixed_html = "<!-- PROMPT_VERSION: apf-v2.1 -->\n" + fixed_html
                card_fixed = True

            if "<!-- BEGIN_CARDS -->" not in fixed_html:
                if "<!-- PROMPT_VERSION:" in fixed_html:
                    fixed_html = fixed_html.replace(
                        "<!-- PROMPT_VERSION: apf-v2.1 -->",
                        "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->",
                    )
                else:
                    fixed_html = "<!-- BEGIN_CARDS -->\n" + fixed_html
                card_fixed = True

            if "<!-- END_CARDS -->" not in fixed_html:
                if "END_OF_CARDS" in fixed_html:
                    fixed_html = fixed_html.replace(
                        "END_OF_CARDS", "<!-- END_CARDS -->\nEND_OF_CARDS"
                    )
                else:
                    fixed_html += "\n<!-- END_CARDS -->\nEND_OF_CARDS"
                card_fixed = True

            if not fixed_html.rstrip().endswith("END_OF_CARDS"):
                fixed_html = fixed_html.rstrip() + "\nEND_OF_CARDS"
                card_fixed = True

            # Aggressively fix all standalone <code> tags
            code_pattern = r"<code(?:\s[^>]*)?>.*?</code>"
            matches = list(
                re.finditer(code_pattern, fixed_html, re.DOTALL | re.IGNORECASE)
            )
            for match in reversed(matches):
                code_tag = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                context_before = fixed_html[max(0, start_pos - 500) : start_pos]
                pre_matches = list(
                    re.finditer(r"<pre(?:\s[^>]*)?>", context_before, re.IGNORECASE)
                )
                if pre_matches:
                    last_pre = pre_matches[-1]
                    pre_start = last_pre.start() + (start_pos - 500)
                    between = fixed_html[pre_start:start_pos]
                    if "</pre>" not in between and "</PRE>" not in between:
                        continue
                wrapped = f"<pre>{code_tag}</pre>"
                fixed_html = fixed_html[:start_pos] + wrapped + fixed_html[end_pos:]
                card_fixed = True

            # Ensure manifest exists and is valid
            if "<!-- manifest:" not in fixed_html:
                header_match = re.search(
                    r"<!--\s*Card\s+(\d+)\s*\|\s*slug:\s*([a-z0-9-]+)\s*\|\s*CardType:\s*(\w+)\s*\|\s*Tags:\s*([^>]+)\s*-->",
                    fixed_html,
                )
                if header_match:
                    card_num, slug, card_type, tags_str = header_match.groups()
                    tags = tags_str.strip().split()[:6]  # Limit to 6 tags
                    manifest_data = {
                        "slug": slug,
                        "lang": card.lang,
                        "type": card_type.capitalize(),
                        "tags": tags,
                    }
                    manifest_json = json.dumps(manifest_data, separators=(",", ":"))
                    manifest_comment = f"<!-- manifest: {manifest_json} -->"
                    if "<!-- END_CARDS -->" in fixed_html:
                        fixed_html = fixed_html.replace(
                            "<!-- END_CARDS -->",
                            f"{manifest_comment}\n<!-- END_CARDS -->",
                        )
                    else:
                        fixed_html += f"\n{manifest_comment}"
                    card_fixed = True

            if card_fixed:
                fixed_card = GeneratedCard(
                    card_index=card.card_index,
                    slug=card.slug,
                    lang=card.lang,
                    apf_html=fixed_html,
                    confidence=card.confidence,
                    content_hash=card.content_hash,
                )
                fixed_cards.append(fixed_card)
                any_fixes = True
            else:
                fixed_cards.append(card)

        return fixed_cards if any_fixes else None
