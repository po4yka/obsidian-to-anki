"""Tests for HTML validator ensures APF formatting correctness."""

import pytest

from obsidian_anki_sync.apf.html_validator import validate_card_html


def test_valid_html_passes():
    html = """<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->\n<!-- Card 1 | slug: sample | CardType: Simple | Tags: en testing -->\n<pre><code class=\"language-kotlin\">val test = 1\n</code></pre>\n<!-- END_CARDS -->\nEND_OF_CARDS"""
    assert validate_card_html(html) == []


def test_detects_backticks():
    html = """```kotlin\nprintln(1)\n```"""
    errors = validate_card_html(html)
    assert any("Backtick" in err for err in errors)


def test_detects_missing_code_in_pre():
    html = """<pre>missing code</pre>"""
    errors = validate_card_html(html)
    assert any("<pre> block without nested <code>" in err for err in errors)


def test_detects_missing_language_class():
    html = """<pre><code>no class</code></pre>"""
    errors = validate_card_html(html)
    assert any("language- class" in err for err in errors)


def test_detects_inline_code():
    html = """<code class=\"language-kotlin\">inline</code>"""
    errors = validate_card_html(html)
    assert any("Inline <code>" in err for err in errors)
