"""Infrastructure service for APF HTML post-processing.

This module handles the post-processing of APF HTML content,
following Clean Architecture principles.
"""

import json
import re

from obsidian_anki_sync.apf.html_generator import HTMLTemplateGenerator
from obsidian_anki_sync.apf.html_validator import validate_card_html
from obsidian_anki_sync.domain.interfaces.tag_generation import ITagGenerator
from obsidian_anki_sync.models import Manifest, NoteMetadata
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class APFHTMLPostProcessor:
    """Infrastructure service for APF HTML post-processing and validation."""

    def __init__(self, tag_generator: ITagGenerator | None = None):
        """Initialize HTML post processor.

        Args:
            tag_generator: Tag generation service
        """
        self.tag_generator = tag_generator

    def post_process_apf(
        self, apf_html: str, metadata: NoteMetadata, manifest: Manifest
    ) -> str:
        """Post-process APF HTML to ensure correctness.

        This method:
        1. Strips markdown code fences
        2. Ensures APF v2.1 wrapper sentinels are present
        3. Removes explanatory text before/after card
        4. Detects card type
        5. Generates and injects correct manifest
        6. Ensures proper formatting

        Args:
            apf_html: Raw APF HTML from LLM
            metadata: Note metadata
            manifest: Card manifest

        Returns:
            Post-processed APF HTML
        """
        # 1. Strip markdown code fences if present
        apf_html = re.sub(r"^```html\s*\n", "", apf_html, flags=re.MULTILINE)
        apf_html = re.sub(r"\n```\s*$", "", apf_html, flags=re.MULTILINE)

        # 2. Strip any text before PROMPT_VERSION or Card comment
        version_start = apf_html.find("<!-- PROMPT_VERSION:")
        card_start = apf_html.find("<!-- Card")

        if version_start >= 0:
            # Has PROMPT_VERSION, strip everything before it
            if version_start > 0:
                logger.debug(
                    "stripped_text_before_version",
                    slug=manifest.slug,
                    chars_removed=version_start,
                )
                apf_html = apf_html[version_start:]
        elif card_start > 0:
            # No PROMPT_VERSION, strip everything before Card
            logger.debug(
                "stripped_text_before_card",
                slug=manifest.slug,
                chars_removed=card_start,
            )
            apf_html = apf_html[card_start:]

        # 3. Strip any text after END_OF_CARDS or manifest comment
        end_of_cards_pos = apf_html.find("END_OF_CARDS")
        if end_of_cards_pos >= 0:
            # Keep until end of END_OF_CARDS line
            apf_html = apf_html[: end_of_cards_pos + len("END_OF_CARDS")]
        else:
            # Fallback: Strip after manifest comment
            manifest_match = re.search(
                r"(<!-- manifest:.*?-->)", apf_html, re.DOTALL)
            if manifest_match:
                end_pos = manifest_match.end()
                apf_html = apf_html[:end_pos]

        # 4. Extract tags from card header
        tags_match = re.search(r"Tags:\s*([^\]]+?)\s*-->", apf_html)
        if tags_match:
            # Use tags from model output
            tags = tags_match.group(1).strip().split()
            logger.debug("extracted_tags_from_output",
                         slug=manifest.slug, tags=tags)
        else:
            # Generate tags deterministically
            tags = self.tag_generator.generate_tags(metadata, manifest.lang) if self.tag_generator else []
            logger.debug("generated_tags", slug=manifest.slug, tags=tags)

        # 5. Detect card type
        if "{{c" in apf_html:
            card_type = "Missing"
        elif "<img " in apf_html and "svg" in apf_html.lower():
            card_type = "Draw"
        else:
            card_type = "Simple"

        logger.debug("detected_card_type", slug=manifest.slug, type=card_type)

        # 6. Generate correct manifest
        manifest_dict = {
            "slug": manifest.slug,
            "lang": manifest.lang,
            "type": card_type,
            "tags": tags,
        }
        correct_manifest = f"<!-- manifest: {json.dumps(manifest_dict, ensure_ascii=False)} -->"

        # 7. Replace existing manifest or append
        if "<!-- manifest:" in apf_html:
            apf_html = re.sub(
                r"<!-- manifest:.*?-->", correct_manifest, apf_html, flags=re.DOTALL
            )
            logger.debug("replaced_manifest", slug=manifest.slug)
        else:
            apf_html += "\n\n" + correct_manifest
            logger.debug("appended_manifest", slug=manifest.slug)

        # 8. Ensure APF v2.1 wrapper sentinels are present
        has_prompt_version = "<!-- PROMPT_VERSION: apf-v2.1 -->" in apf_html
        has_begin_cards = "<!-- BEGIN_CARDS -->" in apf_html
        has_end_cards = "<!-- END_CARDS -->" in apf_html
        has_end_of_cards = "END_OF_CARDS" in apf_html

        if not (
            has_prompt_version
            and has_begin_cards
            and has_end_cards
            and has_end_of_cards
        ):
            # Missing wrapper sentinels, add them
            logger.debug("adding_missing_wrapper_sentinels",
                         slug=manifest.slug)

            # Wrap the card content
            lines = []
            if not has_prompt_version:
                lines.append("<!-- PROMPT_VERSION: apf-v2.1 -->")
            if not has_begin_cards:
                lines.append("<!-- BEGIN_CARDS -->")
                lines.append("")

            lines.append(apf_html.strip())

            if not has_end_cards:
                lines.append("")
                lines.append("<!-- END_CARDS -->")
            if not has_end_of_cards:
                lines.append("END_OF_CARDS")

            apf_html = "\n".join(lines)

        # 9. Normalize card header to match validator expectations
        apf_html = self._normalize_card_header(
            apf_html, manifest, card_type, tags)

        # 10. Fix HTML validation issues: wrap standalone <code> in <pre><code>
        apf_html = self._fix_code_tags(apf_html)

        # 11. Final HTML validation and auto-fix using structured templates
        apf_html = self._validate_and_fix_html(apf_html, manifest)

        # 12. Ensure proper formatting
        apf_html = apf_html.strip()

        # 13. Post-generation cleanup: strip content after END_OF_CARDS and normalize formats
        apf_html = self._post_generation_cleanup(apf_html, manifest)

        return apf_html

    def _post_generation_cleanup(self, apf_html: str, manifest: Manifest) -> str:
        """Perform final cleanup after APF generation.

        Args:
            apf_html: APF HTML to clean
            manifest: Card manifest

        Returns:
            Cleaned APF HTML
        """
        # 1. Strip all content after END_OF_CARDS
        end_of_cards_pos = apf_html.find("END_OF_CARDS")
        if end_of_cards_pos != -1:
            # Keep only up to and including END_OF_CARDS
            apf_html = apf_html[: end_of_cards_pos + len("END_OF_CARDS")]
            logger.debug(
                "post_generation_cleanup_stripped_after_end_of_cards",
                slug=manifest.slug,
            )

        # 2. Normalize CardType format in headers (ensure CardType: not type:)
        # Replace any "type:" with "CardType:" in card headers
        apf_html = re.sub(
            r"(<!--\s*Card\s+\d+\s*\|\s*slug:\s*[^\|]+\s*\|\s*)type:(\s*[^\|]+\s*\|\s*Tags:)",
            r"\1CardType:\2",
            apf_html,
            flags=re.IGNORECASE,
        )

        # 3. Ensure END_OF_CARDS is the last line
        lines = apf_html.split("\n")
        # Remove empty lines at the end
        while lines and lines[-1].strip() == "":
            lines.pop()

        # Ensure last line is END_OF_CARDS
        if lines and lines[-1].strip() == "END_OF_CARDS":
            # Good, END_OF_CARDS is already the last line
            pass
        # Add END_OF_CARDS as the last line if missing
        elif lines and not lines[-1].strip().endswith("END_OF_CARDS"):
            lines.append("END_OF_CARDS")
            logger.debug(
                "post_generation_cleanup_added_end_of_cards",
                slug=manifest.slug,
            )

        return "\n".join(lines)

    def _normalize_card_header(
        self, apf_html: str, manifest: Manifest, card_type: str, tags: list[str]
    ) -> str:
        """Normalize card header to match validator's expected format.

        The validator expects exactly:
        <!-- Card N | slug: slug-name | CardType: Simple | Tags: tag1 tag2 tag3 -->

        Args:
            apf_html: APF HTML content
            manifest: Card manifest
            card_type: Card type (Simple, Missing, Draw)
            tags: List of tags

        Returns:
            APF HTML with normalized card header
        """
        # Build the correct header format
        correct_header = f"<!-- Card {manifest.card_index} | slug: {manifest.slug} | CardType: {card_type} | Tags: {' '.join(tags)} -->"

        # Find and replace existing card header
        # Pattern matches various card header formats the LLM might produce
        card_header_pattern = r"<!--\s*Card\s+\d+[^\]]*?-->"

        match = re.search(card_header_pattern, apf_html)
        if match:
            apf_html = (
                apf_html[: match.start()] + correct_header +
                apf_html[match.end():]
            )
            logger.debug(
                "normalized_card_header",
                slug=manifest.slug,
                old_header=match.group(0)[:100],
                new_header=correct_header,
            )
        else:
            logger.warning(
                "no_card_header_found_to_normalize",
                slug=manifest.slug,
                inserting_header=True,
            )
            # If no header found, insert it after BEGIN_CARDS
            begin_cards_pos = apf_html.find("<!-- BEGIN_CARDS -->")
            if begin_cards_pos >= 0:
                insert_pos = begin_cards_pos + len("<!-- BEGIN_CARDS -->")
                apf_html = (
                    apf_html[:insert_pos]
                    + "\n\n"
                    + correct_header
                    + apf_html[insert_pos:]
                )

        return apf_html

    def _fix_code_tags(self, html: str) -> str:
        """Fix HTML validation issues by wrapping standalone <code> in <pre><code>.

        Args:
            html: HTML content to fix

        Returns:
            Fixed HTML with all code blocks properly wrapped
        """
        # Match <code>...</code> tags (non-greedy to match individual tags)
        fixed_html = html
        code_pattern = r"<code(?:\s[^>]*)?>.*?</code>"
        matches = list(re.finditer(
            code_pattern, fixed_html, re.DOTALL | re.IGNORECASE))

        # Process matches in reverse order to avoid offset issues
        for match in reversed(matches):
            code_tag = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Check if this code tag is already inside a <pre> tag
            # Look backwards for <pre> tag
            context_before = fixed_html[max(0, start_pos - 500): start_pos]
            # Find the last <pre> tag before this code
            pre_matches = list(
                re.finditer(r"<pre(?:\s[^>]*)?>",
                            context_before, re.IGNORECASE)
            )
            if pre_matches:
                last_pre = pre_matches[-1]
                pre_start = last_pre.start() + (start_pos - 500)
                # Check if there's a closing </pre> between <pre> and our <code>
                between = fixed_html[pre_start:start_pos]
                if "</pre>" not in between and "</PRE>" not in between:
                    # <pre> is still open, this code is already wrapped
                    continue

            # This is a standalone <code> tag, wrap it
            wrapped = f"<pre>{code_tag}</pre>"
            fixed_html = fixed_html[:start_pos] + \
                wrapped + fixed_html[end_pos:]

        return fixed_html

    def _validate_and_fix_html(self, apf_html: str, manifest: Manifest) -> str:
        """Validate and auto-fix HTML using structured generation approach.

        Args:
            apf_html: APF HTML content to validate and fix
            manifest: Card manifest for context

        Returns:
            Validated and fixed HTML
        """
        # Validate the HTML
        validation_errors = validate_card_html(apf_html)

        if not validation_errors:
            # HTML is already valid
            return apf_html

        logger.debug(
            "html_validation_errors_found", slug=manifest.slug, errors=validation_errors
        )

        # Attempt auto-fix using the HTML generator
        html_generator = HTMLTemplateGenerator()

        # Try to extract card data and regenerate
        try:
            # Extract basic card data from the HTML
            card_data = self._extract_card_data_from_html(apf_html, manifest)

            if card_data:
                # Regenerate HTML using structured templates
                result = html_generator.generate_card_html(card_data)

                if result.is_valid:
                    logger.info(
                        "html_auto_fixed_with_templates",
                        slug=manifest.slug,
                        original_errors=len(validation_errors),
                        remaining_warnings=len(result.warnings),
                    )

                    if result.warnings:
                        for warning in result.warnings:
                            logger.warning(
                                "html_fix_warning", warning=warning, slug=manifest.slug
                            )

                    return result.html
                else:
                    logger.warning(
                        "html_template_regeneration_failed",
                        slug=manifest.slug,
                        errors=result.validation_errors,
                    )

        except Exception as e:
            logger.warning(
                "html_auto_fix_extraction_failed", slug=manifest.slug, error=str(e)
            )

        # Fallback: apply basic fixes
        fixed_html = self._apply_basic_html_fixes(apf_html, validation_errors)

        logger.debug(
            "applied_basic_html_fixes",
            slug=manifest.slug,
            original_errors=len(validation_errors),
        )

        return fixed_html

    def _extract_card_data_from_html(self, apf_html: str, manifest: Manifest) -> dict | None:
        """Extract card data from existing HTML for regeneration."""
        # Safe access to optional attributes with sensible defaults
        manifest_tags = getattr(manifest, "tags", [])

        card_data = {
            "card_index": 1,
            "slug": manifest.slug,
            "tags": manifest_tags,
            "title": "Generated Card",
            "question": "",
            "answer": "",
            "code_sample": None,
            "key_points": [],
            "other_notes": "",
            "references": "",
        }

        try:
            # Extract title
            title_match = re.search(
                r"<!-- Title -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if title_match:
                card_data["title"] = title_match.group(1).strip()

            # Extract question
            question_match = re.search(
                r"<!-- Question -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if question_match:
                card_data["question"] = question_match.group(1).strip()

            # Extract answer
            answer_match = re.search(
                r"<!-- Answer -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if answer_match:
                card_data["answer"] = answer_match.group(1).strip()

            # Extract key points
            key_points_match = re.search(
                r"<!-- Key point -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if key_points_match:
                key_points_html = key_points_match.group(1).strip()
                # Extract list items
                li_matches = re.findall(
                    r"<li>(.*?)</li>", key_points_html, re.IGNORECASE
                )
                if li_matches:
                    card_data["key_points"] = [li.strip() for li in li_matches]

            # Extract code samples
            code_match = re.search(
                r"<!-- Sample \(code block\) -->\s*\n(.*?)(?=\n<!--|\n*$)",
                apf_html,
                re.DOTALL,
            )
            if code_match:
                code_html = code_match.group(1).strip()
                # Extract code content
                code_content_match = re.search(
                    r"<pre><code[^>]*>(.*?)</code></pre>",
                    code_html,
                    re.DOTALL | re.IGNORECASE,
                )
                if code_content_match:
                    card_data["code_sample"] = code_content_match.group(
                        1).strip()

        except Exception as e:
            logger.debug(
                "card_data_extraction_failed", slug=manifest.slug, error=str(e)
            )
            return None

        return card_data

    def _apply_basic_html_fixes(self, html: str, errors: list[str]) -> str:
        """Apply basic HTML fixes for common validation errors."""
        fixed_html = html

        for error in errors:
            if "language- class" in error:
                # Add default language class to code elements without one
                fixed_html = re.sub(
                    r"<code(?![^>]*class=)",
                    r'<code class="language-text"',
                    fixed_html,
                    flags=re.IGNORECASE,
                )
            elif "wrap in <pre><code>" in error:
                # This is complex to fix automatically, log for manual review
                logger.debug("complex_html_fix_needed", error=error)

        return fixed_html
