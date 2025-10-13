"""APF format validation and linting."""

import json
import re
from typing import Optional

from ..models import ValidationResult
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Constants
MAX_LINE_WIDTH = 88  # Maximum line width for APF cards (advisory)
MIN_TAGS = 3  # Minimum number of tags required
MAX_TAGS = 6  # Maximum number of tags allowed

# APF v2.1 specification from Doc A/D
REQUIRED_SENTINELS = [
    r"^<!-- PROMPT_VERSION: apf-v2\.1 -->$",
    r"^<!-- BEGIN_CARDS -->$",
    r"^<!-- END_CARDS -->$",
]

FIELD_HEADERS_ORDER = [
    "<!-- Title -->",
    "<!-- Subtitle (optional) -->",
    "<!-- Syntax (inline) (optional) -->",
    "<!-- Sample (caption) (optional) -->",
    "<!-- Sample (code block or image) (optional for Missing) -->",
    "<!-- Key point (code block / image) -->",
    "<!-- Key point notes -->",
    "<!-- Other notes (optional) -->",
    "<!-- Markdown (optional) -->",
]

# Tag taxonomy from Doc B
ALLOWED_LANGUAGES = {
    "kotlin",
    "java",
    "python",
    "javascript",
    "typescript",
    "swift",
    "objective_c",
    "c",
    "cpp",
    "rust",
    "go",
    "dart",
    "ruby",
    "php",
    "csharp",
    "sql",
    "yaml",
    "json",
    "bash",
    "powershell",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "gradle",
    "maven",
    "git",
    "regex",
}


def validate_apf(apf_html: str, slug: Optional[str] = None) -> ValidationResult:
    """
    Validate APF card format against specification.

    Args:
        apf_html: APF HTML content
        slug: Expected slug (optional)

    Returns:
        Validation result with errors and warnings
    """
    result = ValidationResult()
    lines = apf_html.split("\n")

    # Check sentinels
    _check_sentinels(lines, result)

    # Check final line
    if lines and lines[-1].strip() != "END_OF_CARDS":
        result.errors.append("Missing final 'END_OF_CARDS' line")

    # Extract card blocks
    card_blocks = _extract_card_blocks(apf_html)

    if not card_blocks:
        result.errors.append("No card blocks found")
        return result

    # Validate each card
    for idx, block in enumerate(card_blocks, 1):
        _validate_card_block(block, idx, slug, result)

    # Check for duplicate slugs
    _check_duplicate_slugs(card_blocks, result)

    logger.debug(
        "apf_validation_completed",
        slug=slug,
        errors=len(result.errors),
        warnings=len(result.warnings),
    )

    return result


def _check_sentinels(lines: list[str], result: ValidationResult) -> None:
    """Check required sentinel lines."""
    content = "\n".join(lines)

    # Check PROMPT_VERSION
    if not re.search(REQUIRED_SENTINELS[0], content, re.MULTILINE):
        result.errors.append("Missing '<!-- PROMPT_VERSION: apf-v2.1 -->' sentinel")

    # Check BEGIN_CARDS
    if not re.search(REQUIRED_SENTINELS[1], content, re.MULTILINE):
        result.errors.append("Missing '<!-- BEGIN_CARDS -->' sentinel")

    # Check END_CARDS
    if not re.search(REQUIRED_SENTINELS[2], content, re.MULTILINE):
        result.errors.append("Missing '<!-- END_CARDS -->' sentinel")


def _extract_card_blocks(apf_html: str) -> list[str]:
    """Extract individual card blocks from APF HTML."""
    # Find content between BEGIN_CARDS and END_CARDS
    match = re.search(
        r"<!-- BEGIN_CARDS -->(.*?)<!-- END_CARDS -->", apf_html, re.DOTALL
    )

    if not match:
        return []

    cards_content = match.group(1)

    # Split by card headers
    card_pattern = r"(<!-- Card \d+.*?-->.*?)(?=<!-- Card \d+|$)"
    blocks = re.findall(card_pattern, cards_content, re.DOTALL)

    return blocks


def _validate_card_block(
    block: str, card_num: int, expected_slug: Optional[str], result: ValidationResult
) -> None:
    """Validate a single card block."""
    lines = block.strip().split("\n")

    if not lines:
        result.errors.append(f"Card {card_num}: Empty card block")
        return

    # Validate card header
    header_match = re.match(
        r"<!-- Card (\d+) \| slug: ([a-z0-9-]+) \| CardType: (Simple|Missing|Draw) \| Tags: (.+?) -->",
        lines[0].strip(),
    )

    if not header_match:
        result.errors.append(f"Card {card_num}: Invalid card header format")
        return

    card_idx, slug, card_type, tags_str = header_match.groups()

    # Check slug matches expected
    if expected_slug and slug != expected_slug:
        result.warnings.append(
            f"Card {card_num}: Slug mismatch (expected {expected_slug}, got {slug})"
        )

    # Validate tags
    tags = tags_str.strip().split()
    _validate_tags(tags, card_num, result)

    # Check for manifest
    if "<!-- manifest:" not in block:
        result.errors.append(f"Card {card_num}: Missing manifest")
    else:
        _validate_manifest(block, slug, card_num, result)

    # Check field headers present
    _check_field_headers(block, card_num, result)

    # Validate cloze density for Missing cards
    if card_type == "Missing":
        _validate_cloze_density(block, card_num, result)

    # Check line width (advisory)
    for line_num, line in enumerate(lines, 1):
        # Skip SVG data URIs
        if "data:image/svg+xml" in line:
            continue
        if len(line) > MAX_LINE_WIDTH:
            result.warnings.append(
                f"Card {card_num}, line {line_num}: Line exceeds {MAX_LINE_WIDTH} characters ({len(line)})"
            )


def _validate_tags(tags: list[str], card_num: int, result: ValidationResult) -> None:
    """Validate tag count and format."""
    if not (MIN_TAGS <= len(tags) <= MAX_TAGS):
        result.errors.append(
            f"Card {card_num}: Must have {MIN_TAGS}-{MAX_TAGS} tags, found {len(tags)}"
        )

    # Check snake_case format
    for tag in tags:
        if not re.match(r"^[a-z0-9_]+$", tag):
            result.errors.append(
                f"Card {card_num}: Tag '{tag}' not in snake_case format"
            )

    # Check first tag is a language
    if tags and tags[0] not in ALLOWED_LANGUAGES:
        result.warnings.append(
            f"Card {card_num}: First tag should be a language/tool, got '{tags[0]}'"
        )

    # Check for at least one non-language tag
    non_lang_tags = [t for t in tags if t not in ALLOWED_LANGUAGES]
    if not non_lang_tags:
        result.errors.append(
            f"Card {card_num}: Must have at least one non-language tag"
        )


def _validate_manifest(
    block: str, expected_slug: str, card_num: int, result: ValidationResult
) -> None:
    """Validate manifest JSON."""
    # Extract manifest
    match = re.search(r"<!-- manifest: ({.*?}) -->", block)
    if not match:
        return

    try:
        manifest = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        result.errors.append(f"Card {card_num}: Invalid manifest JSON: {e}")
        return

    # Check required fields
    required_fields = ["slug", "lang", "type", "tags"]
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        result.errors.append(f"Card {card_num}: Manifest missing fields: {missing}")

    # Check slug matches
    if manifest.get("slug") != expected_slug:
        result.errors.append(
            f"Card {card_num}: Manifest slug mismatch (header: {expected_slug}, manifest: {manifest.get('slug')})"
        )


def _check_field_headers(block: str, card_num: int, result: ValidationResult) -> None:
    """Check that field headers are present and in order."""
    # Required headers
    required = ["<!-- Title -->", "<!-- Key point", "<!-- Key point notes -->"]

    for header in required:
        if header not in block:
            result.errors.append(
                f"Card {card_num}: Missing required field header '{header}'"
            )


def _validate_cloze_density(
    block: str, card_num: int, result: ValidationResult
) -> None:
    """Validate cloze numbering is dense (1..N)."""
    # Find all cloze indices
    cloze_matches = re.findall(r"\{\{c(\d+)::", block)

    if not cloze_matches:
        result.warnings.append(f"Card {card_num}: Missing card has no cloze deletions")
        return

    indices = sorted(set(int(m) for m in cloze_matches))

    # Check density
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        result.errors.append(
            f"Card {card_num}: Cloze indices not dense (expected {expected}, got {indices})"
        )


def _check_duplicate_slugs(card_blocks: list[str], result: ValidationResult) -> None:
    """Check for duplicate slugs across blocks."""
    slugs = []

    for block in card_blocks:
        match = re.search(r"slug: ([a-z0-9-]+)", block)
        if match:
            slugs.append(match.group(1))

    seen = set()
    for slug in slugs:
        if slug in seen:
            result.errors.append(f"Duplicate slug found: {slug}")
        seen.add(slug)
