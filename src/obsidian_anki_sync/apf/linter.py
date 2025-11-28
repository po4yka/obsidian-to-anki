"""APF format validation and linting."""

import json
import re

from obsidian_anki_sync.models import ValidationResult
from obsidian_anki_sync.utils.logging import get_logger

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


def validate_apf(apf_html: str, slug: str | None = None) -> ValidationResult:
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
    block: str, card_num: int, expected_slug: str | None, result: ValidationResult
) -> None:
    """Validate a single card block."""
    lines = block.strip().split("\n")

    if not lines:
        result.errors.append(f"Card {card_num}: Empty card block")
        return

    # Validate card header - strict format checking
    header_line = lines[0].strip()

    # First check for common mistakes before regex matching
    _validate_header_format_strict(header_line, card_num, result)
    if result.errors:
        return

    header_pattern = r"<!-- Card (\d+) \| slug: ([a-z0-9-]+) \| CardType: (Simple|Missing|Draw) \| Tags: (.+?) -->"
    header_match = re.match(header_pattern, header_line)

    if not header_match:
        # Provide detailed error message showing what was found
        actual_header = header_line[:200]  # First 200 chars
        logger.warning(
            "invalid_card_header",
            card_num=card_num,
            actual=actual_header,
            expected_pattern="<!-- Card N | slug: name | CardType: Simple/Missing/Draw | Tags: tag1 tag2 tag3 -->",
        )
        result.errors.append(
            f"Card {card_num}: Invalid card header format. "
            f"Found: '{actual_header[:100]}...' | "
            f"Expected: '<!-- Card N | slug: name | CardType: Simple/Missing/Draw | Tags: tag1 tag2 tag3 -->'"
        )
        return

    _card_idx, slug, card_type, tags_str = header_match.groups()

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


def _validate_header_format_strict(
    header_line: str, card_num: int, result: ValidationResult
) -> None:
    """Perform strict validation of card header format to catch common mistakes."""
    if not header_line.startswith("<!--"):
        result.errors.append(f"Card {card_num}: Header must start with '<!--'")
        return

    if not header_line.endswith("-->"):
        result.errors.append(f"Card {card_num}: Header must end with '-->'")
        return

    # Check for common capitalization mistakes
    if "type:" in header_line.lower() and "CardType:" not in header_line:
        result.errors.append(
            f"Card {card_num}: Use 'CardType:' not 'type:' (case-sensitive)"
        )
        return

    if "cardtype:" in header_line.lower() and "CardType:" not in header_line:
        result.errors.append(f"Card {card_num}: Use 'CardType:' with capital C and T")
        return

    # Check spacing around pipes
    if " |" not in header_line or "| " not in header_line:
        result.errors.append(
            f"Card {card_num}: Header must have spaces around pipe characters: ' | '"
        )
        return

    # Check for comma-separated tags instead of space-separated
    if "Tags:" in header_line and "," in header_line.split("Tags:")[1].split("-->")[0]:
        result.errors.append(
            f"Card {card_num}: Tags must be space-separated, not comma-separated"
        )
        return


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

    indices = sorted({int(m) for m in cloze_matches})

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
