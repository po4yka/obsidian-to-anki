"""Skill loader for dynamic loading of specialized prompt instructions.

Skills are markdown documents containing domain-specific guidance that agents
can load just-in-time, avoiding permanent context overhead.
"""

from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SkillLoader:
    """Load Skills (markdown prompt documents) dynamically.

    Skills are stored in `.docs/skills/` directory and can be loaded
    on-demand by agents based on task requirements.
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize skill loader.

        Args:
            base_path: Base path to skills directory. Defaults to project root.
        """
        if base_path is None:
            # Default to project root (.docs/skills/)
            # Assuming this is called from project root
            project_root = Path(__file__).parents[3]
            self.base_path = project_root / ".docs" / "skills"
        else:
            self.base_path = Path(base_path)

        if not self.base_path.exists():
            logger.warning(
                "skills_directory_not_found",
                path=str(self.base_path),
                message="Skills directory does not exist. Skills will not be available.",
            )

    def load(self, skill_name: str) -> str:
        """Load a skill by name.

        Args:
            skill_name: Name of skill file (without .md extension)

        Returns:
            Skill content as string

        Raises:
            FileNotFoundError: If skill file doesn't exist
        """
        skill_path = self.base_path / f"{skill_name}.md"

        if not skill_path.exists():
            error_msg = f"Skill '{skill_name}' not found at {skill_path}"
            logger.error("skill_not_found", skill_name=skill_name, path=str(skill_path))
            raise FileNotFoundError(error_msg)

        try:
            content = skill_path.read_text(encoding="utf-8")
            logger.debug("skill_loaded", skill_name=skill_name, path=str(skill_path))
            return content
        except Exception as e:
            logger.error(
                "skill_load_error",
                skill_name=skill_name,
                path=str(skill_path),
                error=str(e),
            )
            raise

    def load_multiple(self, skill_names: list[str]) -> list[str]:
        """Load multiple skills.

        Args:
            skill_names: List of skill names to load

        Returns:
            List of skill contents in order
        """
        return [self.load(name) for name in skill_names]

    def combine(self, skill_names: list[str], separator: str = "\n\n---\n\n") -> str:
        """Load and combine multiple skills into single string.

        Args:
            skill_names: List of skill names to load and combine
            separator: String to insert between skills

        Returns:
            Combined skill content
        """
        skills = self.load_multiple(skill_names)
        return separator.join(skills)

    def list_available(self) -> list[str]:
        """List all available skills.

        Returns:
            List of skill names (without .md extension)
        """
        if not self.base_path.exists():
            return []

        skills = []
        for skill_file in self.base_path.glob("*.md"):
            if skill_file.name != "README.md":  # Exclude README
                skills.append(skill_file.stem)

        return sorted(skills)

    def skill_exists(self, skill_name: str) -> bool:
        """Check if a skill exists.

        Args:
            skill_name: Name of skill to check

        Returns:
            True if skill exists, False otherwise
        """
        skill_path = self.base_path / f"{skill_name}.md"
        return skill_path.exists()

