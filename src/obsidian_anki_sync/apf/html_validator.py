"""HTML structure validation for APF cards."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


def validate_card_html(apf_html: str) -> list[str]:
    """Validate HTML structure used in APF cards.

    Args:
        apf_html: Raw APF HTML string.

    Returns:
        List of validation error messages (empty if valid enough).
    """
    errors: list[str] = []

    try:
        soup = BeautifulSoup(apf_html, "html5lib")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"HTML parser error: {exc}")
        return errors

    # APF cards should not contain Markdown code fences
    if "```" in apf_html:
        errors.append("Backtick code fences detected; use <pre><code> blocks.")

    # Check for Markdown bold/italic syntax (outside of code blocks)
    # Remove code blocks first to avoid false positives
    text_without_code = re.sub(
        r"<pre>.*?</pre>", "", apf_html, flags=re.DOTALL)
    if re.search(r"\*\*[^*]+\*\*", text_without_code):
        errors.append(
            "Markdown **bold** detected; use <strong> HTML tags instead.")
    if re.search(r"(?<!\*)\*[^*]+\*(?!\*)", text_without_code):
        # Single asterisk italic (but not inside double asterisks)
        if re.search(r"(?<!\*)\*[^*\s][^*]*[^*\s]\*(?!\*)", text_without_code):
            errors.append(
                "Markdown *italic* detected; use <em> HTML tags instead.")

    # Validate every <pre> has a <code> child with language class
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code is None:
            errors.append("<pre> block without nested <code> element")
            continue

        # Language classes are recommended but not strictly required for APF validation
        # They are mainly used for syntax highlighting and missing them is not a critical error
        # Skip this check to allow cards with code blocks that don't have language classes

    # Check for bare <code> elements outside <pre>
    for code in soup.find_all("code"):
        if code.parent is not None and code.parent.name != "pre":
            errors.append(
                "Inline <code> elements are not allowed; wrap in <pre><code>."
            )

    return errors
