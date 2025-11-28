"""Code language detection utilities for Q&A content."""

import re

from obsidian_anki_sync.models import NoteMetadata


def detect_code_language_from_metadata(metadata: NoteMetadata) -> str:
    """Derive language hint for code blocks from metadata.

    Checks tags, subtopics, and topic for known programming languages.

    Args:
        metadata: Note metadata containing tags, subtopics, and topic

    Returns:
        Language identifier (e.g., 'python', 'kotlin') or 'plaintext'
    """
    candidates = list(metadata.tags) + metadata.subtopics + [metadata.topic]
    known_languages = {
        "kotlin",
        "java",
        "python",
        "swift",
        "cpp",
        "c",
        "csharp",
        "go",
        "rust",
        "javascript",
        "typescript",
        "sql",
        "bash",
        "shell",
        "yaml",
        "json",
        "html",
        "css",
        "gradle",
        "groovy",
    }

    for raw in candidates:
        if not raw:
            continue
        normalized = (
            raw.lower()
            .replace("language/", "")
            .replace("lang/", "")
            .replace("topic/", "")
            .replace("/", "_")
        )
        if normalized in known_languages:
            return normalized

    return "plaintext"


def detect_code_language_from_content(content: str) -> str | None:
    """Detect programming language from code content heuristics.

    Analyzes code patterns to infer the language.

    Args:
        content: Code snippet to analyze

    Returns:
        Language identifier or None if not detected
    """
    if not content or len(content.strip()) < 5:
        return None

    # Kotlin indicators
    if any(
        pattern in content
        for pattern in [
            "fun ",
            "val ",
            "var ",
            "data class",
            "sealed class",
            "object ",
            "companion object",
        ]
    ):
        return "kotlin"

    # Java indicators
    if (
        re.search(r"\bpublic\s+class\s+\w+", content)
        or re.search(r"\bprivate\s+\w+\s+\w+\(", content)
    ) and "fun " not in content:  # Distinguish from Kotlin
        return "java"

    # Python indicators
    if any(
        pattern in content
        for pattern in [
            "def ",
            "import ",
            "from ",
            "class ",
            "__init__",
            "self.",
        ]
    ):
        return "python"

    # Swift indicators
    if (
        any(
            pattern in content
            for pattern in ["func ", "var ", "let ", "import Foundation", "struct "]
        )
        and "fun " not in content
    ):  # Distinguish from Kotlin
        return "swift"

    # JavaScript/TypeScript indicators
    if any(
        pattern in content
        for pattern in [
            "const ",
            "let ",
            "var ",
            "function ",
            "=>",
            "console.log",
        ]
    ):
        if re.search(r":\s*\w+\s*=", content) or ": string" in content:
            return "typescript"
        return "javascript"

    # YAML indicators (check before SQL due to similar syntax)
    if re.search(r"^[a-z_]+:\s", content, re.MULTILINE) and "-" in content:
        return "yaml"

    # JSON indicators
    if content.strip().startswith("{") and '"' in content and ":" in content:
        return "json"

    # SQL indicators
    if re.search(
        r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\s+", content, re.IGNORECASE
    ):
        return "sql"

    # Shell/Bash indicators
    if content.strip().startswith("#!") or any(
        pattern in content for pattern in ["#!/bin/bash", "#!/bin/sh", "echo "]
    ):
        return "bash"

    # Go indicators
    if any(
        pattern in content
        for pattern in ["package ", "func ", "import (", "type ", "var "]
    ):
        return "go"

    # Rust indicators
    if any(
        pattern in content for pattern in ["fn ", "let mut", "impl ", "use ", "pub "]
    ):
        return "rust"

    # C/C++ indicators
    if re.search(r"#include\s+<", content) or re.search(r"\bvoid\s+\w+\(", content):
        if "std::" in content or "namespace " in content:
            return "cpp"
        return "c"

    # No clear indicators
    return None
