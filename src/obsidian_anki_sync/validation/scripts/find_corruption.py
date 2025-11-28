#!/usr/bin/env python3
"""Identify corruption issues introduced by previous sub-agents."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def analyze_file(filepath: Path, vault_root: Path) -> dict | None:
    """Analyze file for corruption issues."""
    try:
        content = filepath.read_text(encoding="utf-8")

        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = yaml.safe_load(parts[1])
        body = parts[2]

        issues: list[dict] = []

        # Check for missing sections
        has_ru_question = bool(re.search(r"^# Вопрос \(RU\)", body, re.MULTILINE))
        has_en_question = bool(re.search(r"^# Question \(EN\)", body, re.MULTILINE))
        has_ru_answer = bool(re.search(r"^## Ответ \(RU\)", body, re.MULTILINE))
        has_en_answer = bool(re.search(r"^## Answer \(EN\)", body, re.MULTILINE))

        missing_sections: list[str] = []
        if not has_ru_question:
            missing_sections.append("# Вопрос (RU)")
        if not has_en_question:
            missing_sections.append("# Question (EN)")
        if not has_ru_answer:
            missing_sections.append("## Ответ (RU)")
        if not has_en_answer:
            missing_sections.append("## Answer (EN)")

        if missing_sections:
            issues.append(
                {
                    "type": "missing_sections",
                    "severity": "CRITICAL",
                    "missing": missing_sections,
                }
            )

        # Check for missing difficulty tag
        tags = frontmatter.get("tags", [])
        difficulty = frontmatter.get("difficulty", "")
        expected_tag = f"difficulty/{difficulty}"

        if difficulty and expected_tag not in tags:
            issues.append(
                {
                    "type": "missing_difficulty_tag",
                    "severity": "ERROR",
                    "expected_tag": expected_tag,
                    "current_tags": tags,
                }
            )

        # Check for too many subtopics
        subtopics = frontmatter.get("subtopics", [])
        if len(subtopics) > 3:
            issues.append(
                {
                    "type": "too_many_subtopics",
                    "severity": "WARNING",
                    "count": len(subtopics),
                    "subtopics": subtopics,
                }
            )

        # Check for invalid/malformed subtopics
        if frontmatter.get("topic") == "android":
            invalid_subtopics = [
                st for st in subtopics if len(st) <= 2 or not isinstance(st, str)
            ]
            if invalid_subtopics:
                issues.append(
                    {
                        "type": "invalid_subtopics",
                        "severity": "ERROR",
                        "invalid": invalid_subtopics,
                        "all_subtopics": subtopics,
                    }
                )

        # Check for trailing whitespace
        lines_with_trailing: list[int] = []
        for i, line in enumerate(content.split("\n"), 1):
            if line.endswith((" ", "\t")):
                lines_with_trailing.append(i)

        if lines_with_trailing:
            issues.append(
                {
                    "type": "trailing_whitespace",
                    "severity": "INFO",
                    "lines": lines_with_trailing[:10],  # First 10 lines only
                }
            )

        if issues:
            return {
                "filepath": str(filepath.relative_to(vault_root)),
                "filename": filepath.name,
                "title": frontmatter.get("title", ""),
                "difficulty": difficulty,
                "topic": frontmatter.get("topic", ""),
                "subtopics": subtopics,
                "tags": tags,
                "issues": issues,
            }

        return None

    except Exception as e:
        return None


def main() -> None:
    """Main entry point for corruption issue identification."""
    vault_root = Path.cwd()
    android_dir = vault_root / "40-Android"

    corruption_issues: dict[str, list[dict]] = {
        "missing_sections": [],
        "missing_difficulty_tag": [],
        "too_many_subtopics": [],
        "invalid_subtopics": [],
        "trailing_whitespace": [],
    }

    note_files = sorted(android_dir.glob("q-*.md"))
    analyzed = 0

    for filepath in note_files:
        try:
            content = filepath.read_text(encoding="utf-8")
            if "status: reviewed" not in content:
                continue

            analyzed += 1
            result = analyze_file(filepath, vault_root)

            if result:
                for issue in result["issues"]:
                    issue_type = issue["type"]
                    if issue_type in corruption_issues:
                        corruption_issues[issue_type].append(
                            {
                                "file_info": {
                                    k: v for k, v in result.items() if k != "issues"
                                },
                                "issue": issue,
                            }
                        )

        except Exception as e:
            pass

    # Create repair work packages
    output_dir = vault_root / "repair_work_packages"
    output_dir.mkdir(exist_ok=True)

    agents: list[dict] = []

    # Agent 1: Restore difficulty tags
    if corruption_issues["missing_difficulty_tag"]:
        agents.append(
            {
                "agent_id": "difficulty-tag-restorer",
                "task_type": "missing_difficulty_tag",
                "description": "Restore missing difficulty tags",
                "files": corruption_issues["missing_difficulty_tag"],
                "count": len(corruption_issues["missing_difficulty_tag"]),
            }
        )

    # Agent 2: Fix trailing whitespace
    if corruption_issues["trailing_whitespace"]:
        agents.append(
            {
                "agent_id": "whitespace-cleaner",
                "task_type": "trailing_whitespace",
                "description": "Remove trailing whitespace",
                "files": corruption_issues["trailing_whitespace"],
                "count": len(corruption_issues["trailing_whitespace"]),
            }
        )

    # Agent 3: Fix too many subtopics
    if corruption_issues["too_many_subtopics"]:
        agents.append(
            {
                "agent_id": "subtopics-consolidator",
                "task_type": "too_many_subtopics",
                "description": "Consolidate subtopics to 1-3 values",
                "files": corruption_issues["too_many_subtopics"],
                "count": len(corruption_issues["too_many_subtopics"]),
            }
        )

    # Agent 4: Fix invalid subtopics
    if corruption_issues["invalid_subtopics"]:
        agents.append(
            {
                "agent_id": "subtopics-validator",
                "task_type": "invalid_subtopics",
                "description": "Fix invalid/malformed subtopics",
                "files": corruption_issues["invalid_subtopics"],
                "count": len(corruption_issues["invalid_subtopics"]),
            }
        )

    # Agent 5-6: Restore missing sections (split into 2 batches)
    if corruption_issues["missing_sections"]:
        files: list[dict[str, Any]] = corruption_issues["missing_sections"]
        mid = len(files) // 2

        agents.append(
            {
                "agent_id": "sections-restorer-1",
                "task_type": "missing_sections",
                "description": "Restore missing sections (batch 1/2)",
                "files": files[:mid],
                "count": mid,
            }
        )

        agents.append(
            {
                "agent_id": "sections-restorer-2",
                "task_type": "missing_sections",
                "description": "Restore missing sections (batch 2/2)",
                "files": files[mid:],
                "count": len(files) - mid,
            }
        )

    for agent in agents:
        pass

    # Save work packages
    for agent in agents:
        agent_file = output_dir / f"{agent['agent_id']}.json"
        with open(agent_file, "w", encoding="utf-8") as f:
            json.dump(agent, f, indent=2, ensure_ascii=False)

    summary = {
        "total_agents": len(agents),
        "total_files": sum(a["count"] for a in agents),
        "agents": [
            {"id": a["agent_id"], "task": a["task_type"], "files": a["count"]}
            for a in agents
        ],
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
