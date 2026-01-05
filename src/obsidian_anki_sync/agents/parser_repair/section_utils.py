"""Section extraction helpers for parser repair."""

from __future__ import annotations

from collections.abc import Iterable


def _count_prefix(keys: Iterable[str], prefix: str) -> int:
    return len([k for k in keys if k.startswith(prefix)])


def identify_sections(content: str) -> dict[str, str]:
    """Identify sections in note content."""
    sections: dict[str, str] = {}
    lines = content.split("\n")

    frontmatter_start = None
    frontmatter_end = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            if frontmatter_start is None:
                frontmatter_start = i
            else:
                frontmatter_end = i
                break

    if frontmatter_start is not None and frontmatter_end is not None:
        sections["frontmatter"] = "\n".join(
            lines[frontmatter_start : frontmatter_end + 1]
        )

    current_section = None
    current_section_lines = []

    for line in lines:
        if line.strip().startswith("# Question") or line.strip().startswith("# Вопрос"):
            if current_section:
                sections[current_section] = "\n".join(current_section_lines)
            current_section = f"question_{_count_prefix(sections.keys(), 'question')}"
            current_section_lines = [line]
        elif line.strip().startswith("## Answer") or line.strip().startswith(
            "## Ответ"
        ):
            if current_section:
                sections[current_section] = "\n".join(current_section_lines)
            current_section = f"answer_{_count_prefix(sections.keys(), 'answer')}"
            current_section_lines = [line]
        elif current_section:
            current_section_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_section_lines)

    if not sections:
        sections["content"] = content

    return sections
