"""Deterministic fixes for APF card validation errors."""

import json
import re

from ....utils.logging import get_logger
from ...models import GeneratedCard

logger = get_logger(__name__)


class DeterministicFixer:
    """Applies deterministic fixes without LLM for common issues.

    This class handles a wide range of fixable errors including:
    - Missing sentinels (PROMPT_VERSION, BEGIN_CARDS, END_CARDS, END_OF_CARDS)
    - Invalid card headers
    - Missing manifests
    - HTML structure issues
    - Tag format issues
    - Manifest slug mismatches
    """

    @staticmethod
    def apply_fixes(
        cards: list[GeneratedCard], error_details: str
    ) -> list[GeneratedCard] | None:
        """Apply deterministic fixes to cards.

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

            # Apply all fix methods
            fixed_html, sentinel_fixed = DeterministicFixer._fix_missing_sentinels(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or sentinel_fixed

            fixed_html, header_fixed = DeterministicFixer._fix_invalid_card_header(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or header_fixed

            fixed_html, manifest_fixed = DeterministicFixer._fix_missing_manifest(
                fixed_html, error_details, card.slug, card.lang
            )
            card_fixed = card_fixed or manifest_fixed

            fixed_html, code_fixed = DeterministicFixer._fix_inline_code(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or code_fixed

            fixed_html, tags_fixed = DeterministicFixer._fix_tag_format(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or tags_fixed

            fixed_html, slug_fixed = DeterministicFixer._fix_manifest_slug_mismatch(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or slug_fixed

            fixed_html, field_fixed = DeterministicFixer._fix_missing_field_headers(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or field_fixed

            fixed_html, json_fixed = DeterministicFixer._fix_invalid_manifest_json(
                fixed_html, error_details, card.slug, card.lang
            )
            card_fixed = card_fixed or json_fixed

            fixed_html, dup_fixed = DeterministicFixer._fix_duplicate_end_markers(
                fixed_html, error_details, card.slug
            )
            card_fixed = card_fixed or dup_fixed

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

    @staticmethod
    def _fix_missing_sentinels(
        html: str, error_details: str, slug: str
    ) -> tuple[str, bool]:
        """Fix missing sentinels (PROMPT_VERSION, BEGIN_CARDS, END_CARDS, END_OF_CARDS).

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if "Missing" not in error_details and "sentinel" not in error_details.lower():
            return html, False

        fixed = False

        # Add PROMPT_VERSION if missing
        if "<!-- PROMPT_VERSION:" not in html:
            html = "<!-- PROMPT_VERSION: apf-v2.1 -->\n" + html
            fixed = True
            logger.debug("deterministic_fix_added_prompt_version", slug=slug)

        # Add BEGIN_CARDS if missing
        if "<!-- BEGIN_CARDS -->" not in html:
            if "<!-- PROMPT_VERSION:" in html:
                html = html.replace(
                    "<!-- PROMPT_VERSION: apf-v2.1 -->",
                    "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->",
                )
            else:
                html = "<!-- BEGIN_CARDS -->\n" + html
            fixed = True
            logger.debug("deterministic_fix_added_begin_cards", slug=slug)

        # Add END_CARDS if missing
        if "<!-- END_CARDS -->" not in html:
            # Remove any existing END_OF_CARDS first to avoid duplicates
            html = html.replace("END_OF_CARDS", "").rstrip()
            html += "\n<!-- END_CARDS -->\nEND_OF_CARDS"
            fixed = True
            logger.debug("deterministic_fix_added_end_cards", slug=slug)

        # Add END_OF_CARDS if missing (must be last line)
        if not html.rstrip().endswith("END_OF_CARDS"):
            html = html.rstrip() + "\nEND_OF_CARDS"
            fixed = True
            logger.debug("deterministic_fix_added_end_of_cards", slug=slug)

        return html, fixed

    @staticmethod
    def _fix_invalid_card_header(
        html: str, error_details: str, slug: str
    ) -> tuple[str, bool]:
        """Fix invalid card header format.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if (
            "Invalid card header" not in error_details
            and "card header format" not in error_details.lower()
        ):
            return html, False

        # Try to extract and fix card header - match various formats
        header_match = re.search(
            r"<!--\s*Card\s+(\d+)\s*\|\s*slug:\s*([a-z0-9-]+)\s*\|\s*(?:CardType|type):\s*(Simple|Missing|Draw)\s*\|\s*Tags:\s*([^>]+?)\s*-->", html, re.IGNORECASE)
        if header_match:
            # Full header match - extract components
            card_num, extracted_slug, card_type, tags_str = header_match.groups()
            card_type = card_type.capitalize()
        else:
            # Fallback: try to extract individual components
            num_match = re.search(r"<!--\s*Card\s+(\d+)", html)
            if not num_match:
                return html, False
            card_num = num_match.group(1)

            # Extract slug from card if available
            slug_match = re.search(r"slug:\s*([a-z0-9-]+)", html)
            extracted_slug = slug_match.group(1) if slug_match else slug

            # Extract card type
            card_type_match = re.search(
                r"(?:CardType|type):\s*(Simple|Missing|Draw)", html, re.IGNORECASE
            )
            card_type = (
                card_type_match.group(1).capitalize()
                if card_type_match else "Simple"
            )

            # Extract tags
            tags_match = re.search(r"Tags:\s*([^>]+)", html)
            if tags_match:
                tags_str = tags_match.group(1).strip()
            else:
                # Generate default tags from slug
                tags_str = " ".join(extracted_slug.split("-")[:3])

        # For single cards, always use card number 1
        card_num = "1"

        # Build correct header
        correct_header = f"<!-- Card {card_num} | slug: {extracted_slug} | CardType: {card_type} | Tags: {tags_str} -->"

        # Find and replace the existing header
        # Match headers that may be missing the closing -- before >
        header_pattern = r"<!--\s*Card\s+\d+.*?(?:-->|->|>)"
        if re.search(header_pattern, html):
            html = re.sub(header_pattern, correct_header, html, count=1)
            logger.debug(
                "deterministic_fix_card_header",
                slug=slug,
                new_header=correct_header,
            )
            return html, True
        else:
            # If no header found to replace, add one at the beginning
            html = correct_header + "\n" + html
            logger.debug(
                "deterministic_fix_card_header_added",
                slug=slug,
                header=correct_header,
            )
            return html, True

    @staticmethod
    def _fix_missing_manifest(
        html: str, error_details: str, slug: str, lang: str
    ) -> tuple[str, bool]:
        """Fix missing manifest.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug
            lang: Card language

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if "Missing manifest" not in error_details and not (
            "manifest" in error_details.lower() and "missing" in error_details.lower()
        ):
            return html, False

        # Extract card info from header
        header_match = re.search(
            r"<!--\s*Card\s+(\d+)\s*\|\s*slug:\s*([a-z0-9-]+)\s*\|\s*CardType:\s*(\w+)\s*\|\s*Tags:\s*([^>]+)\s*-->",
            html,
        )
        if not header_match:
            return html, False

        card_num, extracted_slug, card_type, tags_str = header_match.groups()
        tags = tags_str.strip().split()

        # Create manifest JSON
        manifest_data = {
            "slug": extracted_slug,
            "lang": lang,
            "type": card_type,
            "tags": tags,
        }
        manifest_json = json.dumps(manifest_data, separators=(",", ":"))
        manifest_comment = f"<!-- manifest: {manifest_json} -->"

        # Insert manifest before END_CARDS or at end
        if "<!-- END_CARDS -->" in html:
            html = html.replace(
                "<!-- END_CARDS -->",
                f"{manifest_comment}\n<!-- END_CARDS -->",
            )
        else:
            html += f"\n{manifest_comment}"

        logger.debug("deterministic_fix_added_manifest", slug=slug)
        return html, True

    @staticmethod
    def _fix_inline_code(html: str, error_details: str, slug: str) -> tuple[str, bool]:
        """Fix inline <code> without <pre> wrapper.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if "HTML" not in error_details or (
            "code" not in error_details.lower()
            and "inline" not in error_details.lower()
        ):
            return html, False

        # Find standalone <code> tags not inside <pre>
        code_pattern = r"<code(?:\s[^>]*)?>.*?</code>"
        matches = list(re.finditer(
            code_pattern, html, re.DOTALL | re.IGNORECASE))

        if not matches:
            return html, False

        fixed = False
        # Process in reverse to avoid offset issues
        for match in reversed(matches):
            code_tag = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Check if already inside <pre>
            context_before = html[max(0, start_pos - 500): start_pos]
            pre_matches = list(
                re.finditer(r"<pre(?:\s[^>]*)?>",
                            context_before, re.IGNORECASE)
            )
            if pre_matches:
                last_pre = pre_matches[-1]
                pre_start = last_pre.start() + (start_pos - 500)
                between = html[pre_start:start_pos]
                if "</pre>" not in between and "</PRE>" not in between:
                    continue  # Already wrapped

            # Wrap standalone code tag
            wrapped = f"<pre>{code_tag}</pre>"
            html = html[:start_pos] + wrapped + html[end_pos:]
            fixed = True
            logger.debug("deterministic_fix_wrapped_code", slug=slug)

        return html, fixed

    @staticmethod
    def _fix_tag_format(html: str, error_details: str, slug: str) -> tuple[str, bool]:
        """Fix tag format issues (snake_case, count).

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if "tag" not in error_details.lower() or not (
            "format" in error_details.lower()
            or "snake_case" in error_details.lower()
            or "Must have" in error_details
        ):
            return html, False

        # Extract tags from header (stop before -->)
        tags_match = re.search(r"Tags:\s*(.+?)\s*-->", html)
        if not tags_match:
            return html, False

        tags_str = tags_match.group(1).strip()
        tags = tags_str.split()

        # Fix tag format: convert to snake_case, remove invalid chars
        fixed_tags = []
        for tag in tags:
            # Convert to lowercase, replace spaces/hyphens with underscores
            fixed_tag = tag.lower().replace("-", "_").replace(" ", "_")
            # Remove any remaining invalid chars
            fixed_tag = re.sub(r"[^a-z0-9_]", "", fixed_tag)
            fixed_tag = re.sub(r"_+", "_", fixed_tag).strip("_")
            if fixed_tag:
                fixed_tags.append(fixed_tag)

        # Ensure minimum 3 tags
        if len(fixed_tags) < 3:
            # Add default tags from slug
            slug_parts = slug.split("-")
            for part in slug_parts:
                if part and part not in fixed_tags and len(fixed_tags) < 6:
                    fixed_tags.append(part)

        # Limit to 6 tags max
        fixed_tags = fixed_tags[:6]

        if fixed_tags == tags:
            return html, False

        # Replace tags in header (preserve -->)
        new_tags_str = " ".join(fixed_tags)
        html = re.sub(r"(Tags:\s*)(.+?)(\s*-->)", rf"\1{new_tags_str}\3", html)
        logger.debug(
            "deterministic_fix_tags",
            slug=slug,
            old=tags,
            new=fixed_tags,
        )

        return html, True

    @staticmethod
    def _fix_manifest_slug_mismatch(
        html: str, error_details: str, slug: str
    ) -> tuple[str, bool]:
        """Fix manifest slug mismatch.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if not (
            "manifest" in error_details.lower()
            and "slug" in error_details.lower()
            and "mismatch" in error_details.lower()
        ):
            return html, False

        # Extract slug from card header
        header_match = re.search(
            r"<!--\s*Card\s+\d+\s*\|\s*slug:\s*([^\s|]+)", html)
        if not header_match:
            return html, False

        header_slug = header_match.group(1)

        # Check manifest slug
        manifest_match = re.search(
            r'<!--\s*manifest:.*?"slug"\s*:\s*"([^"]+)"',
            html,
            re.DOTALL,
        )
        if not manifest_match:
            return html, False

        manifest_slug = manifest_match.group(1)
        if header_slug == manifest_slug:
            return html, False

        # Fix manifest to match header
        html = re.sub(
            r'(<!--\s*manifest:.*?"slug"\s*:\s*")[^"]+(")',
            rf"\1{header_slug}\2",
            html,
            flags=re.DOTALL,
        )
        logger.debug(
            "deterministic_fix_manifest_slug",
            slug=slug,
            fixed_slug=header_slug,
        )

        return html, True

    @staticmethod
    def _fix_missing_field_headers(
        html: str, error_details: str, slug: str
    ) -> tuple[str, bool]:
        """Fix missing field headers (add minimal required headers).

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if (
            "Missing required field header" not in error_details
            and "field header" not in error_details.lower()
        ):
            return html, False

        # Check for required headers
        required_headers = {
            "<!-- Title -->": "<!-- Title -->\n<p>Untitled</p>",
            "<!-- Key point": "<!-- Key point (code block) -->\n<p>Key point content</p>",
            "<!-- Key point notes -->": "<!-- Key point notes -->\n<ul>\n<li>Note</li>\n</ul>",
        }

        fixed = False
        for header_check, header_with_content in required_headers.items():
            if header_check not in html:
                # Find insertion point (after card header, before manifest)
                if "<!-- manifest:" in html:
                    html = html.replace(
                        "<!-- manifest:",
                        f"{header_with_content}\n\n<!-- manifest:",
                    )
                elif "<!-- END_CARDS -->" in html:
                    html = html.replace(
                        "<!-- END_CARDS -->",
                        f"{header_with_content}\n\n<!-- END_CARDS -->",
                    )
                else:
                    # Add at end before END_OF_CARDS
                    html = html.rstrip() + f"\n\n{header_with_content}"
                fixed = True
                logger.debug(
                    "deterministic_fix_added_field_header",
                    slug=slug,
                    header=header_check,
                )

        return html, fixed

    @staticmethod
    def _fix_invalid_manifest_json(
        html: str, error_details: str, slug: str, lang: str
    ) -> tuple[str, bool]:
        """Fix invalid manifest JSON.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug
            lang: Card language

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        if (
            "Invalid manifest JSON" not in error_details
            and "manifest JSON" not in error_details.lower()
        ):
            return html, False

        # Try to fix malformed JSON in manifest
        manifest_match = re.search(
            r"<!--\s*manifest:\s*({.*?})\s*-->", html, re.DOTALL)
        if not manifest_match:
            return html, False

        try:
            # Try to parse existing JSON
            json.loads(manifest_match.group(1))
            return html, False
        except json.JSONDecodeError:
            pass

        # JSON is invalid, try to extract and rebuild
        header_match = re.search(
            r"<!--\s*Card\s+\d+\s*\|\s*slug:\s*([^\s|]+)\s*\|\s*CardType:\s*(\w+)\s*\|\s*Tags:\s*([^>]+)\s*-->",
            html,
        )
        if not header_match:
            return html, False

        extracted_slug = header_match.group(1)
        card_type = header_match.group(2)
        tags = header_match.group(3).strip().split()

        # Create valid manifest
        manifest_data = {
            "slug": extracted_slug,
            "lang": lang,
            "type": card_type,
            "tags": tags,
        }
        manifest_json = json.dumps(manifest_data, separators=(",", ":"))
        new_manifest = f"<!-- manifest: {manifest_json} -->"

        # Replace old manifest
        html = re.sub(
            r"<!--\s*manifest:.*?-->",
            new_manifest,
            html,
            flags=re.DOTALL,
        )
        logger.debug("deterministic_fix_manifest_json", slug=slug)

        return html, True

    @staticmethod
    def _fix_duplicate_end_markers(
        html: str, error_details: str, slug: str
    ) -> tuple[str, bool]:
        """Fix duplicate END_OF_CARDS markers and malformed end structure.

        Args:
            html: Card HTML
            error_details: Error description
            slug: Card slug

        Returns:
            Tuple of (fixed_html, was_fixed)
        """
        # Check if this is relevant (triggered by various end marker issues)
        if not (
            "END_OF_CARDS" in error_details
            or "END_CARDS" in error_details
            or "end marker" in error_details.lower()
            or "extra" in error_details.lower()
        ):
            return html, False

        # Check for the common case: <!-- END_CARDS --> followed by multiple END_OF_CARDS
        if "<!-- END_CARDS -->" in html:
            # Split at END_CARDS marker to isolate the ending section
            parts = html.split("<!-- END_CARDS -->")
            if len(parts) >= 2:
                before_end = parts[0]
                after_end = "".join(parts[1:])

                # Count END_OF_CARDS occurrences in the after_end section
                end_count = after_end.count("END_OF_CARDS")
                if end_count > 1:
                    # Remove all END_OF_CARDS and add just one
                    after_end_clean = after_end.replace(
                        "END_OF_CARDS", "").strip()
                    html = before_end + "<!-- END_CARDS -->\n" + after_end_clean
                    if after_end_clean:
                        html += "\n"
                    html += "END_OF_CARDS"
                    logger.debug(
                        "deterministic_fix_duplicate_end_markers",
                        slug=slug,
                        removed_count=end_count - 1,
                    )
                    return html, True

        return html, False
