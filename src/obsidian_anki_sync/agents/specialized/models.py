"""Shared models for specialized agents."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class RepairResult:
    """Result from content repair operation."""

    success: bool
    repaired_content: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    error_message: str | None = None
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ProblemDomain(Enum):
    """Types of problems that can be handled by specialized agents."""

    YAML_FRONTMATTER = "yaml_frontmatter"
    CONTENT_STRUCTURE = "content_structure"
    CONTENT_CORRUPTION = "content_corruption"
    CODE_BLOCKS = "code_blocks"
    HTML_VALIDATION = "html_validation"
    QA_EXTRACTION = "qa_extraction"
    QUALITY_ASSURANCE = "quality_assurance"


@dataclass
class AgentResult:
    """Result from a specialized agent."""

    success: bool
    content: str | None = None
    metadata: dict[str, Any | None] = None
    qa_pairs: list[dict[str, Any | None]] = None
    confidence: float = 0.0
    reasoning: str = ""
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
