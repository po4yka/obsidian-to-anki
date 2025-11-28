"""Extract content from <result></result> tags in LLM responses."""

import re

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


def extract_result_tag(content: str, require_tag: bool = False) -> str:
    """
    Extract content from <result></result> tags.

    Args:
        content: LLM response text
        require_tag: If True, raise error if no tag found; if False, return original content

    Returns:
        Extracted content or original content if no tag found

    Raises:
        ValueError: If require_tag is True and no tag is found
    """
    # Match <result>...</result> with optional whitespace
    pattern = r"<result>(.*?)</result>"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        extracted = match.group(1).strip()
        logger.debug("result_tag_extracted", length=len(extracted))
        return extracted

    if require_tag:
        msg = "No <result></result> tag found in response"
        raise ValueError(msg)

    # No tag found, return original content
    logger.debug("no_result_tag_found", returning_original=True)
    return content.strip()


def has_result_tag(content: str) -> bool:
    """
    Check if content contains <result></result> tags.

    Args:
        content: Text to check

    Returns:
        True if result tag is found
    """
    pattern = r"<result>.*?</result>"
    return bool(re.search(pattern, content, re.DOTALL | re.IGNORECASE))
