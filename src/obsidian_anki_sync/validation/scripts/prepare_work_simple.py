#!/usr/bin/env python3
"""Prepare work packages for sub-agents - simplified version without validator dependencies."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def analyze_file(
    filepath: Path, vault_root: Path, android_valid_subtopics: List[str]
) -> Optional[Dict]:
    """Analyze a file for issues."""
    try:
        content = filepath.read_text(encoding="utf-8")

        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter = yaml.safe_load(parts[1])
        body = parts[2]

        issues_found: List[Dict] = []

        # Check for invalid Android subtopics
        if frontmatter.get("topic") == "android":
            subtopics = frontmatter.get("subtopics", [])
            invalid = [st for st in subtopics if st not in android_valid_subtopics]

            if invalid:
                issues_found.append(
                    {
                        "type": "invalid_subtopics",
                        "severity": "ERROR",
                        "message": f"Invalid Android subtopics: {', '.join(invalid)}",
                        "current_subtopics": subtopics,
                        "invalid_subtopics": invalid,
                    }
                )

        # Check for missing concept links
        concept_pattern = r"\[\[c-[^\]]+\]\]"
        concept_links = re.findall(concept_pattern, body)

        if not concept_links:
            # Extract EN answer preview
            en_match = re.search(
                r"^## Answer \(EN\)\s*\n(.*?)(?=^##|\Z)",
                body,
                re.MULTILINE | re.DOTALL,
            )
            en_preview = ""
            if en_match:
                answer = en_match.group(1).strip()
                en_preview = answer[:300] + "..." if len(answer) > 300 else answer

            issues_found.append(
                {
                    "type": "missing_concept_links",
                    "severity": "WARNING",
                    "message": "No concept links found",
                    "en_answer_preview": en_preview,
                }
            )

        # Check for broken wikilinks in related field
        related = frontmatter.get("related", [])
        all_notes = index_all_notes(vault_root)

        broken_links = [link for link in related if link not in all_notes]
        if broken_links:
            issues_found.append(
                {
                    "type": "broken_wikilinks",
                    "severity": "CRITICAL",
                    "message": f"Broken links in related field: {', '.join(broken_links)}",
                    "related_field": related,
                    "broken_links": broken_links,
                }
            )

        # Check for wrong folder
        topic = frontmatter.get("topic", "")
        current_folder = filepath.parent.name
        expected_folder = get_expected_folder(topic)

        if expected_folder and current_folder != expected_folder:
            issues_found.append(
                {
                    "type": "wrong_folder",
                    "severity": "CRITICAL",
                    "message": f"File should be in {expected_folder}/, found in {current_folder}/",
                    "current_folder": current_folder,
                    "expected_folder": expected_folder,
                }
            )

        if issues_found:
            return {
                "filepath": str(filepath.relative_to(vault_root)),
                "filename": filepath.name,
                "title": frontmatter.get("title", ""),
                "topic": frontmatter.get("topic", ""),
                "subtopics": frontmatter.get("subtopics", []),
                "difficulty": frontmatter.get("difficulty", ""),
                "related": frontmatter.get("related", []),
                "issues_found": issues_found,
            }

        return None

    except Exception as e:
        print(f"Error analyzing {filepath.name}: {e}", file=sys.stderr)
        return None


def index_all_notes(vault_root: Path) -> set:
    """Index all note filenames."""
    notes = set()
    for md_file in vault_root.rglob("*.md"):
        if any(part.startswith(".") for part in md_file.parts):
            continue
        notes.add(md_file.stem)
    return notes


def get_expected_folder(topic: str) -> str:
    """Get expected folder for topic."""
    mapping = {
        "algorithms": "20-Algorithms",
        "system-design": "30-System-Design",
        "android": "40-Android",
        "databases": "50-Backend",
        "kotlin": "70-Kotlin",
        "tools": "80-Tools",
        "operating-systems": "60-CompSci",
        "networking": "60-CompSci",
        "security": "60-CompSci",
        "cs": "60-CompSci",
    }
    return mapping.get(topic, "")


def load_android_subtopics(vault_root: Path) -> List[str]:
    """Load valid Android subtopics from TAXONOMY.md."""
    taxonomy_path = vault_root / "00-Administration/Vault-Rules/TAXONOMY.md"
    try:
        content = taxonomy_path.read_text(encoding="utf-8")

        # Find Android subtopics section
        in_section = False
        subtopics: List[str] = []

        for line in content.split("\n"):
            if "### Android Subtopics" in line:
                in_section = True
                continue
            elif line.startswith("###") and in_section:
                break

            if in_section and line.strip().startswith("-"):
                subtopic = line.strip().lstrip("- `").rstrip("`").strip()
                if subtopic:
                    subtopics.append(subtopic)

        return subtopics
    except Exception as e:
        print(f"Error loading Android subtopics: {e}", file=sys.stderr)
        return []


def main() -> None:
    """Main entry point for simplified work package preparation."""
    vault_root = Path.cwd()
    android_dir = vault_root / "40-Android"

    print("Loading Android subtopics from TAXONOMY.md...")
    android_subtopics = load_android_subtopics(vault_root)
    print(f"Loaded {len(android_subtopics)} valid Android subtopics")
    print()

    print("Analyzing reviewed Android notes...")
    work_packages: Dict[str, List[Dict]] = {
        "invalid_subtopics": [],
        "missing_concept_links": [],
        "broken_wikilinks": [],
        "wrong_folder": [],
    }

    files = sorted(android_dir.glob("q-*.md"))
    analyzed = 0

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            if "status: reviewed" not in content:
                continue

            analyzed += 1
            result = analyze_file(filepath, vault_root, android_subtopics)

            if result:
                for issue in result["issues_found"]:
                    issue_type = issue["type"]
                    if issue_type in work_packages:
                        work_packages[issue_type].append(
                            {
                                "file_info": {
                                    k: v for k, v in result.items() if k != "issues_found"
                                },
                                "issue": issue,
                            }
                        )

        except Exception as e:
            print(f"Error processing {filepath.name}: {e}", file=sys.stderr)

    print(f"Analyzed {analyzed} reviewed files")
    print()
    print("=" * 80)
    print("ISSUES FOUND")
    print("=" * 80)
    print(
        f"Invalid Android subtopics: {len(work_packages['invalid_subtopics'])} files"
    )
    print(
        f"Missing concept links:     {len(work_packages['missing_concept_links'])} files"
    )
    print(f"Broken wikilinks:          {len(work_packages['broken_wikilinks'])} files")
    print(
        f"Wrong folder placement:    {len(work_packages['wrong_folder'])} files"
    )
    print("=" * 80)
    print()

    # Distribute work
    agents: List[Dict] = []

    # Agent 1: Invalid subtopics
    if work_packages["invalid_subtopics"]:
        agents.append(
            {
                "agent_id": "subtopics-fixer",
                "task_type": "invalid_subtopics",
                "description": "Fix invalid Android subtopics",
                "files": work_packages["invalid_subtopics"],
                "count": len(work_packages["invalid_subtopics"]),
            }
        )

    # Agent 2: Broken wikilinks
    if work_packages["broken_wikilinks"]:
        agents.append(
            {
                "agent_id": "wikilinks-fixer",
                "task_type": "broken_wikilinks",
                "description": "Fix broken wikilinks",
                "files": work_packages["broken_wikilinks"],
                "count": len(work_packages["broken_wikilinks"]),
            }
        )

    # Agent 3: Wrong folder
    if work_packages["wrong_folder"]:
        agents.append(
            {
                "agent_id": "folder-fixer",
                "task_type": "wrong_folder",
                "description": "Move files to correct folders",
                "files": work_packages["wrong_folder"],
                "count": len(work_packages["wrong_folder"]),
            }
        )

    # Agents 4-9: Missing concept links (distribute across 6 agents)
    if work_packages["missing_concept_links"]:
        files = work_packages["missing_concept_links"]
        files_per_agent = max(1, len(files) // 6)

        for i in range(6):
            start_idx = i * files_per_agent
            end_idx = start_idx + files_per_agent if i < 5 else len(files)

            if start_idx < len(files):
                agents.append(
                    {
                        "agent_id": f"concepts-{i+1}",
                        "task_type": "missing_concept_links",
                        "description": f"Add concept links (batch {i+1}/6)",
                        "files": files[start_idx:end_idx],
                        "count": end_idx - start_idx,
                    }
                )

    print("WORK DISTRIBUTION")
    print("=" * 80)
    for agent in agents:
        print(
            f"{agent['agent_id']:20} - {agent['count']:3} files - {agent['description']}"
        )
    print("=" * 80)
    print(f"Total: {sum(a['count'] for a in agents)} files across {len(agents)} agents")
    print()

    # Save work packages
    output_dir = vault_root / "subagent_work_packages"
    output_dir.mkdir(exist_ok=True)

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

    print(f"Saved {len(agents)} work packages to {output_dir}/")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
