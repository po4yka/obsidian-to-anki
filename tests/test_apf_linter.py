"""Tests for APF linter (UNIT-apf-a, UNIT-tag-01, LINT-cloze)."""

from obsidian_anki_sync.apf.linter import validate_apf


class TestAPFValidation:
    """Test APF format validation (UNIT-apf-a)."""

    def test_valid_simple_card(self) -> None:
        """Test validation of a valid Simple card."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: test-card-en | CardType: Simple | Tags: python testing unit_testing -->

<!-- Title -->
What is unit testing?

<!-- Subtitle (optional) -->

<!-- Syntax (inline) (optional) -->

<!-- Sample (caption) (optional) -->

<!-- Sample (code block or image) (optional for Missing) -->

<!-- Key point (code block / image) -->
<pre><code class="language-python">
# Unit testing example
def test_addition() -> None:
    assert 1 + 1 == 2
</code></pre>

<!-- Key point notes -->
<ul>
  <li>Tests small units of code in isolation</li>
  <li>Fast and reliable</li>
  <li>Catches bugs early</li>
</ul>

<!-- Other notes (optional) -->
<ul>
  <li>Ref: https://pytest.org</li>
</ul>

<!-- Markdown (optional) -->

<!-- manifest: {"slug":"test-card-en","lang":"python","type":"Simple","tags":["python","testing","unit_testing"]} -->

<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html, "test-card-en")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_sentinels(self) -> None:
        """Test error on missing sentinels."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
Answer
<!-- Key point notes -->
<ul><li>Note</li></ul>
"""
        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("PROMPT_VERSION" in e for e in result.errors)
        assert any("BEGIN_CARDS" in e for e in result.errors)
        assert any("END_CARDS" in e for e in result.errors)

    def test_missing_end_of_cards(self) -> None:
        """Test error on missing END_OF_CARDS."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
Answer
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- END_CARDS -->"""

        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("END_OF_CARDS" in e for e in result.errors)

    def test_invalid_card_header(self) -> None:
        """Test error on invalid card header format."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | invalid header -->
<!-- Title -->
Test
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("Invalid card header" in e for e in result.errors)


class TestTagValidation:
    """Test tag validation (UNIT-tag-01)."""

    def test_valid_tag_count(self) -> None:
        """Test validation of tag count (3-6)."""
        # Valid: 3 tags
        apf_html = self._make_card_html("python testing unit_testing")
        result = validate_apf(apf_html)
        assert result.is_valid

        # Valid: 6 tags
        apf_html = self._make_card_html(
            "python testing unit_testing pytest mocking fixtures"
        )
        result = validate_apf(apf_html)
        assert result.is_valid

    def test_too_few_tags(self) -> None:
        """Test error on too few tags."""
        apf_html = self._make_card_html("python testing")
        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("Must have 3-6 tags" in e for e in result.errors)

    def test_too_many_tags(self) -> None:
        """Test error on too many tags."""
        apf_html = self._make_card_html("a b c d e f g h")
        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("Must have 3-6 tags" in e for e in result.errors)

    def test_invalid_tag_format(self) -> None:
        """Test warning on non-lowercase tag formats (convention)."""
        apf_html = self._make_card_html("python TestCase camelCase")
        result = validate_apf(apf_html)

        # Uppercase tags are now warnings, not errors (convention enforcement)
        assert result.is_valid  # No errors, just warnings
        assert any("should be lowercase" in w for w in result.warnings)

    def test_missing_non_language_tag(self) -> None:
        """Test error when all tags are languages."""
        apf_html = self._make_card_html("python java kotlin")
        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("at least one non-language tag" in e for e in result.errors)

    def test_android_first_tag_valid(self) -> None:
        """Test that 'android' is accepted as a valid first tag."""
        apf_html = self._make_card_html("android jetpack app_startup")
        result = validate_apf(apf_html)

        assert result.is_valid
        # Should not have warnings about first tag
        assert not any("first tag" in w.lower() for w in result.warnings)

    def test_kebab_case_tags_valid(self) -> None:
        """Test that kebab-case tags are accepted."""
        apf_html = self._make_card_html("python app-startup dependency-injection")
        result = validate_apf(apf_html)

        assert result.is_valid

    def _make_card_html(self, tags: str) -> str:
        """Helper to create card HTML with specific tags."""
        return f"""<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: {tags} -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
<pre><code>answer</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- manifest: {{"slug":"test","lang":"python","type":"Simple","tags":[{",".join(f'"{t}"' for t in tags.split())}]}} -->
<!-- END_CARDS -->
END_OF_CARDS"""


class TestClozeValidation:
    """Test cloze deletion validation (LINT-cloze)."""

    def test_valid_cloze_density(self) -> None:
        """Test valid cloze numbering (1..N)."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Missing | Tags: python testing unit -->
<!-- Title -->
Fill in the blanks
<!-- Key point (code block / image) -->
<pre><code>
Testing uses {{c1::pytest}} framework with {{c2::fixtures}} and {{c3::mocks}}.
</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- manifest: {"slug":"test","lang":"python","type":"Missing","tags":["python","testing","unit"]} -->
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert result.is_valid

    def test_sparse_cloze_numbering(self) -> None:
        """Test error on sparse cloze numbering (e.g., 1, 3, 5)."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Missing | Tags: python testing unit -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
<pre><code>
{{c1::first}} and {{c3::third}} but no c2
</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- manifest: {"slug":"test","lang":"python","type":"Missing","tags":["python","testing","unit"]} -->
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("not dense" in e for e in result.errors)

    def test_missing_clozes_in_missing_card(self) -> None:
        """Test warning when Missing card has no clozes."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Missing | Tags: python testing unit -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
<pre><code>
No cloze deletions here
</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- manifest: {"slug":"test","lang":"python","type":"Missing","tags":["python","testing","unit"]} -->
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert any("no cloze deletions" in w for w in result.warnings)


class TestManifestValidation:
    """Test manifest validation."""

    def test_missing_manifest(self) -> None:
        """Test error when manifest is missing."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test | CardType: Simple | Tags: python testing unit -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
<pre><code>answer</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("Missing manifest" in e for e in result.errors)

    def test_slug_mismatch(self) -> None:
        """Test error when manifest slug doesn't match header."""
        apf_html = """<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->
<!-- Card 1 | slug: test-card | CardType: Simple | Tags: python testing unit -->
<!-- Title -->
Test
<!-- Key point (code block / image) -->
<pre><code>answer</code></pre>
<!-- Key point notes -->
<ul><li>Note</li></ul>
<!-- manifest: {"slug":"different-slug","lang":"python","type":"Simple","tags":["python","testing","unit"]} -->
<!-- END_CARDS -->
END_OF_CARDS"""

        result = validate_apf(apf_html)

        assert not result.is_valid
        assert any("slug mismatch" in e.lower() for e in result.errors)
