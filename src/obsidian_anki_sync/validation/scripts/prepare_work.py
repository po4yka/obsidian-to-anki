#!/usr/bin/env python3
"""Prepare work packages for sub-agents to fix remaining issues."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from ..android_validator import AndroidValidator
from ..link_validator import LinkValidator


class WorkPackagePreparator:
    """Prepares work packages for sub-agents."""

    def __init__(self, vault_root: Path):
        self.vault_root = vault_root
        self.work_packages: Dict[str, List[Dict]] = {
            "invalid_subtopics": [],
            "missing_concept_links": [],
            "broken_wikilinks": [],
            "wrong_folder": [],
        }

        # Load taxonomy
        taxonomy_path = vault_root / "00-Administration/Vault-Rules/TAXONOMY.md"
        self.valid_topics = self._load_valid_topics(taxonomy_path)

    def _load_valid_topics(self, taxonomy_path: Path) -> Set[str]:
        """Load valid topics from TAXONOMY.md."""
        try:
            content = taxonomy_path.read_text(encoding="utf-8")
            topics: Set[str] = set()
            in_topics_section = False

            for line in content.split("\n"):
                if "## Topics" in line:
                    in_topics_section = True
                    continue
                elif line.startswith("##") and in_topics_section:
                    break

                if in_topics_section and line.strip().startswith("-"):
                    topic = line.strip().lstrip("- `").rstrip("`")
                    if topic and not topic.startswith("#"):
                        topics.add(topic)

            return topics
        except Exception as e:
            print(f"Error loading taxonomy: {e}", file=sys.stderr)
            return set()

    def analyze_file(self, filepath: Path) -> Optional[Dict]:
        """Analyze a file and identify issues."""
        try:
            content = filepath.read_text(encoding="utf-8")

            # Parse YAML
            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            frontmatter = yaml.safe_load(parts[1])
            body = parts[2]

            issues: Dict[str, any] = {
                "filepath": str(filepath.relative_to(self.vault_root)),
                "filename": filepath.name,
                "title": frontmatter.get("title", ""),
                "topic": frontmatter.get("topic", ""),
                "subtopics": frontmatter.get("subtopics", []),
                "difficulty": frontmatter.get("difficulty", ""),
                "related": frontmatter.get("related", []),
                "issues_found": [],
            }

            # Check for invalid Android subtopics
            if frontmatter.get("topic") == "android":
                validator = AndroidValidator(content, frontmatter, str(filepath))
                android_issues = validator.validate()

                for issue in android_issues:
                    if "Invalid Android subtopics" in issue.message:
                        issues["issues_found"].append(
                            {
                                "type": "invalid_subtopics",
                                "severity": issue.severity.value,
                                "message": issue.message,
                                "current_subtopics": frontmatter.get("subtopics", []),
                            }
                        )

            # Check for missing concept links
            concept_pattern = r"\[\[c-[^\]]+\]\]"
            concept_links = re.findall(concept_pattern, body)

            if not concept_links:
                issues["issues_found"].append(
                    {
                        "type": "missing_concept_links",
                        "severity": "WARNING",
                        "message": "No concept links found",
                        "en_answer_preview": self._extract_en_answer_preview(body),
                    }
                )

            # Check for broken wikilinks
            link_validator = LinkValidator(
                content, frontmatter, str(filepath), self.vault_root
            )
            link_issues = link_validator.validate()

            for issue in link_issues:
                if "non-existent notes" in issue.message:
                    issues["issues_found"].append(
                        {
                            "type": "broken_wikilinks",
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "related_field": frontmatter.get("related", []),
                        }
                    )

            # Check for wrong folder placement
            topic = frontmatter.get("topic", "")
            current_folder = filepath.parent.name
            expected_folder = self._get_expected_folder(topic)

            if expected_folder and current_folder != expected_folder:
                issues["issues_found"].append(
                    {
                        "type": "wrong_folder",
                        "severity": "CRITICAL",
                        "message": f"File in wrong folder. Topic '{topic}' should be in '{expected_folder}/', found in '{current_folder}/'",
                        "current_folder": current_folder,
                        "expected_folder": expected_folder,
                    }
                )

            return issues if issues["issues_found"] else None

        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}", file=sys.stderr)
            return None

    def _extract_en_answer_preview(self, body: str) -> str:
        """Extract preview of EN answer for context."""
        match = re.search(
            r"^## Answer \(EN\)\s*\n(.*?)(?=^##|\Z)",
            body,
            re.MULTILINE | re.DOTALL,
        )
        if match:
            answer = match.group(1).strip()
            # Get first 300 chars
            return answer[:300] + "..." if len(answer) > 300 else answer
        return ""

    def _get_expected_folder(self, topic: str) -> str:
        """Get expected folder for topic."""
        folder_mapping = {
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
        return folder_mapping.get(topic, "")

    def analyze_directory(self, directory: Path, status_filter: Optional[str] = None):
        """Analyze all files in directory."""
        files = sorted(directory.glob("q-*.md"))

        for filepath in files:
            if status_filter:
                try:
                    content = filepath.read_text(encoding="utf-8")
                    if f"status: {status_filter}" not in content:
                        continue
                except Exception:
                    continue

            result = self.analyze_file(filepath)
            if result:
                # Categorize by issue types
                for issue in result["issues_found"]:
                    issue_type = issue["type"]
                    if issue_type in self.work_packages:
                        self.work_packages[issue_type].append(
                            {
                                "file_info": {
                                    "filepath": result["filepath"],
                                    "filename": result["filename"],
                                    "title": result["title"],
                                    "topic": result["topic"],
                                    "subtopics": result["subtopics"],
                                    "difficulty": result["difficulty"],
                                    "related": result["related"],
                                },
                                "issue": issue,
                            }
                        )

    def distribute_work(self, num_agents: int = 4) -> List[Dict]:
        """Distribute work across agents by issue type."""
        # Create specialized agents for different issue types
        agents: List[Dict] = []

        # Agent 1-2: Invalid subtopics
        if self.work_packages["invalid_subtopics"]:
            files = self.work_packages["invalid_subtopics"]
            agents.append(
                {
                    "agent_id": "subtopics-fixer",
                    "task_type": "invalid_subtopics",
                    "description": "Fix invalid Android subtopics by mapping to valid TAXONOMY values",
                    "files": files,
                    "count": len(files),
                }
            )

        # Agent 3: Broken wikilinks
        if self.work_packages["broken_wikilinks"]:
            files = self.work_packages["broken_wikilinks"]
            agents.append(
                {
                    "agent_id": "wikilinks-fixer",
                    "task_type": "broken_wikilinks",
                    "description": "Fix broken wikilinks by removing non-existent references or creating missing files",
                    "files": files,
                    "count": len(files),
                }
            )

        # Agent 4: Wrong folder
        if self.work_packages["wrong_folder"]:
            files = self.work_packages["wrong_folder"]
            agents.append(
                {
                    "agent_id": "folder-fixer",
                    "task_type": "wrong_folder",
                    "description": "Move files to correct folders based on topic",
                    "files": files,
                    "count": len(files),
                }
            )

        # Agents 5-10: Missing concept links (~21 per agent)
        if self.work_packages["missing_concept_links"]:
            files = self.work_packages["missing_concept_links"]
            files_per_agent = len(files) // 6

            for i in range(6):
                start_idx = i * files_per_agent
                end_idx = start_idx + files_per_agent if i < 5 else len(files)

                agents.append(
                    {
                        "agent_id": f"concepts-{i+1}",
                        "task_type": "missing_concept_links",
                        "description": f"Add relevant concept links to notes (batch {i+1}/6)",
                        "files": files[start_idx:end_idx],
                        "count": end_idx - start_idx,
                    }
                )

        return agents

    def save_work_packages(self, agents: List[Dict], output_dir: Path) -> Dict:
        """Save work packages for agents."""
        output_dir.mkdir(exist_ok=True)

        for agent in agents:
            agent_file = output_dir / f"{agent['agent_id']}.json"
            with open(agent_file, "w", encoding="utf-8") as f:
                json.dump(agent, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            "total_agents": len(agents),
            "by_task_type": {},
            "total_files": sum(agent["count"] for agent in agents),
        }

        for agent in agents:
            task_type = agent["task_type"]
            if task_type not in summary["by_task_type"]:
                summary["by_task_type"][task_type] = {"agents": 0, "files": 0}
            summary["by_task_type"][task_type]["agents"] += 1
            summary["by_task_type"][task_type]["files"] += agent["count"]

        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved {len(agents)} work packages to {output_dir}/")
        print(f"Summary saved to {summary_file}")

        return summary

    def print_summary(self) -> None:
        """Print summary of issues found."""
        print("\n" + "=" * 80)
        print("WORK PACKAGE PREPARATION SUMMARY")
        print("=" * 80)
        print(
            f"Invalid Android subtopics: {len(self.work_packages['invalid_subtopics'])} files"
        )
        print(
            f"Missing concept links:     {len(self.work_packages['missing_concept_links'])} files"
        )
        print(
            f"Broken wikilinks:          {len(self.work_packages['broken_wikilinks'])} files"
        )
        print(
            f"Wrong folder placement:    {len(self.work_packages['wrong_folder'])} files"
        )
        print("=" * 80)


def main() -> None:
    """Main entry point for work package preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare work packages for sub-agents"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="40-Android",
        help="Directory to analyze (default: 40-Android)",
    )
    parser.add_argument(
        "--status",
        choices=["draft", "reviewed", "ready"],
        default="reviewed",
        help="Only analyze files with specific status (default: reviewed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subagent_work_packages"),
        help="Output directory for work packages (default: subagent_work_packages)",
    )

    args = parser.parse_args()

    vault_root = Path.cwd()
    dir_path = vault_root / args.path

    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        sys.exit(1)

    preparator = WorkPackagePreparator(vault_root)

    print(f"Analyzing {args.path}...")
    print(f"Status filter: {args.status}")
    print()

    preparator.analyze_directory(dir_path, args.status)
    preparator.print_summary()

    print("\nDistributing work across agents...")
    agents = preparator.distribute_work()

    for agent in agents:
        print(
            f"  {agent['agent_id']}: {agent['count']} files - {agent['description']}"
        )

    summary = preparator.save_work_packages(agents, args.output_dir)

    print("\n" + "=" * 80)
    print("DISTRIBUTION SUMMARY")
    print("=" * 80)
    for task_type, stats in summary["by_task_type"].items():
        print(f"{task_type}: {stats['files']} files across {stats['agents']} agent(s)")
    print(
        f"\nTotal: {summary['total_files']} files across {summary['total_agents']} agents"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
