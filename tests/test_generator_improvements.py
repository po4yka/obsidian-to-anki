"""Tests for GeneratorAgent improvements (APF format review)."""

import json
import re

import pytest

from obsidian_anki_sync.agents.generator import GeneratorAgent
from obsidian_anki_sync.models import Manifest, NoteMetadata


@pytest.fixture
def generator_agent(mock_ollama_provider):
    """Create a GeneratorAgent instance for testing."""
    return GeneratorAgent(
        ollama_client=mock_ollama_provider, model="qwen3:32b", temperature=0.3
    )


@pytest.fixture
def sample_metadata():
    """Sample note metadata for testing."""
    return NoteMetadata(
        id="test-001",
        title="Test Question",
        topic="system-design",
        language_tags=["en", "ru"],
        created="2025-01-01",
        updated="2025-01-02",
        subtopics=["microservices", "scalability"],
        tags=["architecture", "distributed-systems"],
        difficulty="hard",
    )


@pytest.fixture
def sample_manifest():
    """Sample manifest for testing."""
    return Manifest(
        slug="test-card-1-en",
        slug_base="test-card",
        lang="en",
        source_path="test/path.md",
        source_anchor="qa-1",
        note_id="test-001",
        note_title="Test Question",
        card_index=1,
        guid="test-guid-123",
        hash6=None,
    )


class TestDetectCodeLanguage:
    """Tests for _detect_code_language method."""

    def test_detect_kotlin(self, generator_agent):
        """Test Kotlin code detection."""
        kotlin_code = """suspend fun fetchData(): Result<User> {
    val result = repository.getData()
    return result
}"""
        assert generator_agent._detect_code_language(kotlin_code) == "kotlin"

    def test_detect_kotlin_data_class(self, generator_agent):
        """Test Kotlin data class detection."""
        kotlin_code = "data class User(val name: String, val age: Int)"
        assert generator_agent._detect_code_language(kotlin_code) == "kotlin"

    def test_detect_java(self, generator_agent):
        """Test Java code detection."""
        java_code = """public class Main {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}"""
        assert generator_agent._detect_code_language(java_code) == "java"

    def test_detect_python(self, generator_agent):
        """Test Python code detection."""
        python_code = """async def fetch_data() -> User:
    result = await repository.get_data()
    return result"""
        assert generator_agent._detect_code_language(python_code) == "python"

    def test_detect_python_class(self, generator_agent):
        """Test Python class detection."""
        python_code = """class User:
    def __init__(self, name: str):
        self.name = name"""
        assert generator_agent._detect_code_language(python_code) == "python"

    def test_detect_javascript(self, generator_agent):
        """Test JavaScript code detection."""
        js_code = """const fetchData = async () => {
    const result = await fetch('/api/data');
    return result.json();
};"""
        assert generator_agent._detect_code_language(js_code) == "javascript"

    def test_detect_typescript(self, generator_agent):
        """Test TypeScript code detection."""
        ts_code = """interface User {
    name: string;
    age: number;
}

const getUser = (): User => {
    return { name: "John", age: 30 };
};"""
        assert generator_agent._detect_code_language(ts_code) == "typescript"

    def test_detect_bash(self, generator_agent):
        """Test Bash script detection."""
        bash_code = """#!/bin/bash
export PATH=/usr/local/bin:$PATH
echo "Hello World"
sudo apt-get update"""
        assert generator_agent._detect_code_language(bash_code) == "bash"

    def test_detect_yaml(self, generator_agent):
        """Test YAML detection."""
        yaml_code = """name: CI Pipeline
on:
  push:
    branches:
      - main
jobs:
  test:
    runs-on: ubuntu-latest"""
        assert generator_agent._detect_code_language(yaml_code) == "yaml"

    def test_detect_json(self, generator_agent):
        """Test JSON detection."""
        json_code = """{
    "name": "test",
    "version": "1.0.0",
    "dependencies": {
        "express": "^4.18.0"
    }
}"""
        assert generator_agent._detect_code_language(json_code) == "json"

    def test_detect_sql(self, generator_agent):
        """Test SQL detection."""
        sql_code = """SELECT u.name, u.email
FROM users u
WHERE u.active = true
ORDER BY u.created_at DESC;"""
        assert generator_agent._detect_code_language(sql_code) == "sql"

    def test_detect_empty_code(self, generator_agent):
        """Test empty code returns text."""
        assert generator_agent._detect_code_language("") == "text"
        assert generator_agent._detect_code_language("   ") == "text"


class TestGenerateTags:
    """Tests for _generate_tags method."""

    def test_generate_tags_basic(self, generator_agent, sample_metadata):
        """Test basic tag generation."""
        tags = generator_agent._generate_tags(sample_metadata, "en")

        assert len(tags) >= 3
        assert len(tags) <= 6
        # Check snake_case format
        for tag in tags:
            assert re.match(r"^[a-z0-9_]+$", tag)

    def test_generate_tags_includes_topic(self, generator_agent, sample_metadata):
        """Test that tags include topic."""
        tags = generator_agent._generate_tags(sample_metadata, "en")

        assert "system_design" in tags

    def test_generate_tags_includes_subtopics(self, generator_agent, sample_metadata):
        """Test that tags include subtopics."""
        tags = generator_agent._generate_tags(sample_metadata, "en")

        assert any(
            subtopic.replace("-", "_") in tags
            for subtopic in ["microservices", "scalability"]
        )

    def test_generate_tags_deterministic(self, generator_agent, sample_metadata):
        """Test that tag generation is deterministic."""
        tags1 = generator_agent._generate_tags(sample_metadata, "en")
        tags2 = generator_agent._generate_tags(sample_metadata, "en")

        assert tags1 == tags2

    def test_generate_tags_with_platform(self, generator_agent):
        """Test tag generation with platform tags."""
        metadata = NoteMetadata(
            id="test-002",
            title="Android Test",
            topic="android",
            language_tags=["en"],
            created="2025-01-01",
            updated="2025-01-02",
            subtopics=["lifecycle"],
            tags=["android", "kotlin"],
        )

        tags = generator_agent._generate_tags(metadata, "en")

        assert "android" in tags

    def test_generate_tags_with_difficulty(self, generator_agent, sample_metadata):
        """Test tag generation includes difficulty."""
        tags = generator_agent._generate_tags(sample_metadata, "en")

        # Should include difficulty if there's room
        if len(tags) < 6:
            assert any("difficulty" in tag for tag in tags)

    def test_generate_tags_minimum_count(self, generator_agent):
        """Test that at least 3 tags are generated."""
        minimal_metadata = NoteMetadata(
            id="test-003",
            title="Minimal Test",
            topic="test",
            language_tags=["en"],
            created="2025-01-01",
            updated="2025-01-02",
        )

        tags = generator_agent._generate_tags(minimal_metadata, "en")

        assert len(tags) >= 3


class TestGenerateManifest:
    """Tests for _generate_manifest method."""

    def test_generate_manifest_structure(self, generator_agent, sample_manifest):
        """Test manifest has correct structure."""
        tags = ["python", "testing", "pytest"]
        manifest_str = generator_agent._generate_manifest(
            sample_manifest, "Simple", tags
        )

        assert manifest_str.startswith("<!-- manifest:")
        assert manifest_str.endswith("-->")

        # Extract and parse JSON
        json_str = manifest_str.replace("<!-- manifest:", "").replace("-->", "").strip()
        manifest_data = json.loads(json_str)

        assert "slug" in manifest_data
        assert "lang" in manifest_data
        assert "type" in manifest_data
        assert "tags" in manifest_data

    def test_generate_manifest_values(self, generator_agent, sample_manifest):
        """Test manifest contains correct values."""
        tags = ["kotlin", "android", "coroutines"]
        manifest_str = generator_agent._generate_manifest(
            sample_manifest, "Missing", tags
        )

        json_str = manifest_str.replace("<!-- manifest:", "").replace("-->", "").strip()
        manifest_data = json.loads(json_str)

        assert manifest_data["slug"] == sample_manifest.slug
        assert manifest_data["lang"] == sample_manifest.lang
        assert manifest_data["type"] == "Missing"
        assert manifest_data["tags"] == tags


class TestPostProcessAPF:
    """Tests for _post_process_apf method."""

    def test_post_process_strips_markdown_fences(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test that markdown code fences are stripped."""
        apf_html = """```html
<!-- Card 1 | slug: test | CardType: Simple | Tags: python testing -->
<!-- Title -->
Test question
```"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        assert not result.startswith("```")
        assert not result.endswith("```")
        # Should have APF v2.1 wrapper sentinels
        assert result.startswith("<!-- PROMPT_VERSION: apf-v2.1 -->")
        assert "<!-- BEGIN_CARDS -->" in result
        assert "<!-- Card" in result

    def test_post_process_strips_text_before_card(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test that explanatory text before card is removed."""
        apf_html = """Here's the generated card:

<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Test question"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        # Should have APF v2.1 wrapper sentinels
        assert result.startswith("<!-- PROMPT_VERSION: apf-v2.1 -->")
        assert "<!-- BEGIN_CARDS -->" in result
        assert "<!-- Card" in result
        assert "Here's the generated card" not in result

    def test_post_process_injects_manifest(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test that manifest is injected if missing."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: python testing -->
<!-- Title -->
Test question

<!-- Key point (code block) -->
<pre><code class="language-python">
result = test()
</code></pre>

<!-- Key point notes -->
<ul>
  <li>Test explanation</li>
</ul>"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        assert "<!-- manifest:" in result
        # Verify manifest is valid JSON
        manifest_match = re.search(r"<!-- manifest: ({.*?}) -->", result, re.DOTALL)
        assert manifest_match
        manifest_data = json.loads(manifest_match.group(1))
        assert manifest_data["slug"] == sample_manifest.slug

    def test_post_process_replaces_invalid_manifest(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test that invalid manifest is replaced."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Test question

<!-- manifest: {"slug": "wrong-slug", "lang": "en"} -->"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        manifest_match = re.search(r"<!-- manifest: ({.*?}) -->", result, re.DOTALL)
        assert manifest_match
        manifest_data = json.loads(manifest_match.group(1))
        assert manifest_data["slug"] == sample_manifest.slug

    def test_post_process_detects_card_type_missing(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test card type detection for Missing cards."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Complete the code

<!-- Key point (code block with cloze) -->
<pre><code class="language-python">
def test():
    return {{c1::42}}
</code></pre>"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        manifest_match = re.search(r"<!-- manifest: ({.*?}) -->", result, re.DOTALL)
        manifest_data = json.loads(manifest_match.group(1))
        assert manifest_data["type"] == "Missing"

    def test_post_process_detects_card_type_draw(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test card type detection for Draw cards."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: python -->
<!-- Title -->
Draw the architecture

<!-- Key point (image) -->
<img src="data:image/svg+xml;utf8,<svg></svg>" alt="diagram"/>"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        manifest_match = re.search(r"<!-- manifest: ({.*?}) -->", result, re.DOTALL)
        manifest_data = json.loads(manifest_match.group(1))
        assert manifest_data["type"] == "Draw"

    def test_post_process_uses_model_tags(
        self, generator_agent, sample_metadata, sample_manifest
    ):
        """Test that tags from model output are used if valid."""
        apf_html = """<!-- Card 1 | slug: test | CardType: Simple | Tags: kotlin android coroutines -->
<!-- Title -->
Test question"""

        result = generator_agent._post_process_apf(
            apf_html, sample_metadata, sample_manifest
        )

        manifest_match = re.search(r"<!-- manifest: ({.*?}) -->", result, re.DOTALL)
        manifest_data = json.loads(manifest_match.group(1))
        assert "kotlin" in manifest_data["tags"]
        assert "android" in manifest_data["tags"]
        assert "coroutines" in manifest_data["tags"]
