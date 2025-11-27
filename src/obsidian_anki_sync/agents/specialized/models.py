"""Shared models for specialized agents."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class RepairResult:
    """Result from content repair operation."""

    success: bool
    repaired_content: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    error_message: Optional[str] = None
    warnings: List[str] = None

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
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    qa_pairs: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    reasoning: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
