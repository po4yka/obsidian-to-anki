"""HTML structure validation for APF cards."""

from __future__ import annotations

from typing import List

from bs4 import BeautifulSoup


def validate_card_html(apf_html: str) -> List[str]:
    """Validate HTML structure used in APF cards.

    Args:
        apf_html: Raw APF HTML string.

    Returns:
        List of validation error messages (empty if valid enough).
    """
    errors: List[str] = []

    try:
        soup = BeautifulSoup(apf_html, "html5lib")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"HTML parser error: {exc}")
        return errors

    # APF cards should not contain Markdown code fences
    if "```" in apf_html:
        errors.append("Backtick code fences detected; use <pre><code> blocks.")

    # Validate every <pre> has a <code> child with language class
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code is None:
            errors.append("<pre> block without nested <code> element")
            continue

        classes = code.get("class") or []
        if not any(cls.startswith("language-") for cls in classes):
            errors.append("<code> element missing language- class")

    # Check for bare <code> elements outside <pre>
    for code in soup.find_all("code"):
        if code.parent.name != "pre":
            errors.append("Inline <code> elements are not allowed; wrap in <pre><code>.")

    return errors
