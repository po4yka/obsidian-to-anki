#!/usr/bin/env python3
"""Identify files with missing sections and prepare for parallel processing."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class MissingSectionIdentifier:
    """Identifies files with missing RU/EN sections."""

    def __init__(self, vault_root: Path):
        self.vault_root = vault_root
        self.missing_sections: Dict[str, List[Dict]] = {
            "missing_ru_question": [],
            "missing_en_question": [],
            "missing_ru_answer": [],
            "missing_en_answer": [],
        }

    def analyze_file(self, filepath: Path) -> Optional[Dict]:
        """Analyze a file for missing sections."""
        try:
            content = filepath.read_text(encoding="utf-8")

            # Parse YAML frontmatter
            if not content.startswith("---"):
                return None

            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            yaml_str = parts[1]
            body = parts[2]

            try:
                frontmatter = yaml.safe_load(yaml_str)
            except yaml.YAMLError:
                return None

            # Check for sections
            has_ru_question = bool(re.search(r"^# Вопрос \(RU\)", body, re.MULTILINE))
            has_en_question = bool(re.search(r"^# Question \(EN\)", body, re.MULTILINE))
            has_ru_answer = bool(re.search(r"^## Ответ \(RU\)", body, re.MULTILINE))
            has_en_answer = bool(re.search(r"^## Answer \(EN\)", body, re.MULTILINE))

            missing: List[str] = []
            if not has_ru_question:
                missing.append("ru_question")
            if not has_en_question:
                missing.append("en_question")
            if not has_ru_answer:
                missing.append("ru_answer")
            if not has_en_answer:
                missing.append("en_answer")

            if not missing:
                return None

            # Extract existing sections for context
            sections: Dict[str, str] = {}

            # Extract EN question if present
            en_q_match = re.search(
                r"^# Question \(EN\)\s*\n(.*?)(?=^#|\Z)",
                body,
                re.MULTILINE | re.DOTALL,
            )
            if en_q_match:
                sections["en_question"] = en_q_match.group(1).strip()

            # Extract RU question if present
            ru_q_match = re.search(
                r"^# Вопрос \(RU\)\s*\n(.*?)(?=^#|\Z)",
                body,
                re.MULTILINE | re.DOTALL,
            )
            if ru_q_match:
                sections["ru_question"] = ru_q_match.group(1).strip()

            # Extract EN answer if present
            en_a_match = re.search(
                r"^## Answer \(EN\)\s*\n(.*?)(?=^##|\Z)",
                body,
                re.MULTILINE | re.DOTALL,
            )
            if en_a_match:
                sections["en_answer"] = en_a_match.group(1).strip()[
                    :500
                ]  # First 500 chars

            # Extract RU answer if present
            ru_a_match = re.search(
                r"^## Ответ \(RU\)\s*\n(.*?)(?=^##|\Z)",
                body,
                re.MULTILINE | re.DOTALL,
            )
            if ru_a_match:
                sections["ru_answer"] = ru_a_match.group(1).strip()[
                    :500
                ]  # First 500 chars

            return {
                "filepath": str(filepath.relative_to(self.vault_root)),
                "filename": filepath.name,
                "title": frontmatter.get("title", ""),
                "difficulty": frontmatter.get("difficulty", ""),
                "topic": frontmatter.get("topic", ""),
                "missing": missing,
                "existing_sections": sections,
            }

        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}", file=sys.stderr)
            return None

    def analyze_directory(self, directory: Path, status_filter: Optional[str] = None):
        """Analyze all Q&A files in directory."""
        files = sorted(directory.glob("q-*.md"))

        for filepath in files:
            # Filter by status if requested
            if status_filter:
                try:
                    content = filepath.read_text(encoding="utf-8")
                    if f"status: {status_filter}" not in content:
                        continue
                except Exception:
                    continue

            result = self.analyze_file(filepath)
            if result:
                # Categorize by missing sections
                for missing_type in result["missing"]:
                    key = f"missing_{missing_type}"
                    if key in self.missing_sections:
                        self.missing_sections[key].append(result)

    def distribute_work(self, num_agents: int) -> List[Dict]:
        """Distribute files across agents."""
        # Prioritize RU questions
        all_files: Dict[str, Dict] = {}

        for filepath_info in self.missing_sections["missing_ru_question"]:
            filepath = filepath_info["filepath"]
            if filepath not in all_files:
                all_files[filepath] = filepath_info

        for filepath_info in self.missing_sections["missing_en_answer"]:
            filepath = filepath_info["filepath"]
            if filepath not in all_files:
                all_files[filepath] = filepath_info

        files_list = list(all_files.values())
        files_per_agent = len(files_list) // num_agents
        remainder = len(files_list) % num_agents

        batches: List[Dict] = []
        start_idx = 0

        for i in range(num_agents):
            # Give extra file to first 'remainder' agents
            batch_size = files_per_agent + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size

            batches.append(
                {
                    "agent_id": i + 1,
                    "files": files_list[start_idx:end_idx],
                    "count": batch_size,
                }
            )

            start_idx = end_idx

        return batches

    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("MISSING SECTIONS SUMMARY")
        print("=" * 80)
        print(
            f"Missing RU Question: {len(self.missing_sections['missing_ru_question'])} files"
        )
        print(
            f"Missing EN Question: {len(self.missing_sections['missing_en_question'])} files"
        )
        print(
            f"Missing RU Answer:   {len(self.missing_sections['missing_ru_answer'])} files"
        )
        print(
            f"Missing EN Answer:   {len(self.missing_sections['missing_en_answer'])} files"
        )
        print("=" * 80)

    def save_batches(self, batches: List[Dict], output_dir: Path) -> None:
        """Save batch files for agents."""
        output_dir.mkdir(exist_ok=True)

        for batch in batches:
            batch_file = output_dir / f"batch_{batch['agent_id']}.json"
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(batch, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = output_dir / "summary.json"
        summary = {
            "total_files": sum(batch["count"] for batch in batches),
            "num_agents": len(batches),
            "missing_ru_question": len(self.missing_sections["missing_ru_question"]),
            "missing_en_question": len(self.missing_sections["missing_en_question"]),
            "missing_ru_answer": len(self.missing_sections["missing_ru_answer"]),
            "missing_en_answer": len(self.missing_sections["missing_en_answer"]),
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved {len(batches)} batch files to {output_dir}/")
        print(f"Summary saved to {summary_file}")


def main() -> None:
    """Main entry point for missing section identification."""
    parser = argparse.ArgumentParser(
        description="Identify files with missing sections and prepare for parallel processing"
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
        help="Only analyze files with specific status",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=10,
        help="Number of agents to distribute work across (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("missing_sections_batches"),
        help="Output directory for batch files (default: missing_sections_batches)",
    )

    args = parser.parse_args()

    vault_root = Path.cwd()
    dir_path = vault_root / args.path

    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}")
        sys.exit(1)

    identifier = MissingSectionIdentifier(vault_root)

    print(f"Analyzing {args.path}...")
    if args.status:
        print(f"Filtering by status: {args.status}")
    print()

    identifier.analyze_directory(dir_path, args.status)
    identifier.print_summary()

    print(f"\nDistributing work across {args.num_agents} agents...")
    batches = identifier.distribute_work(args.num_agents)

    for batch in batches:
        print(f"  Agent {batch['agent_id']}: {batch['count']} files")

    identifier.save_batches(batches, args.output_dir)


if __name__ == "__main__":
    main()
