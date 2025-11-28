"""Specialized LLM agents for handling different types of note processing problems."""

from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

from .base import BaseSpecializedAgent, ContentRepairAgent, FallbackAgent
from .code import CodeBlockAgent
from .corruption import ContentCorruptionAgent
from .html import HTMLValidationAgent
from .models import AgentResult, ProblemDomain, RepairResult
from .qa import QAExtractionAgent
from .quality import QualityAssuranceAgent
from .router import ProblemRouter
from .structure import ContentStructureAgent
from .yaml import YAMLFrontmatterAgent

logger = get_logger(__name__)

# Global router instance
problem_router = ProblemRouter()


def diagnose_and_solve_problems(
    content: str, error_context: dict[str, Any]
) -> list[AgentResult]:
    """Diagnose problems and attempt solutions using specialized agents.

    Args:
        content: The problematic content
        error_context: Context about the error

    Returns:
        List of agent results (successful repairs will be included)
    """
    diagnoses = problem_router.diagnose_and_route(content, error_context)
    results = []

    for domain, confidence in diagnoses:
        logger.info(
            "trying_specialized_agent", domain=domain.value, confidence=confidence
        )

        result = problem_router.solve_problem(domain, content, error_context)
        results.append(result)

        # If successful, we can stop here
        if result.success:
            logger.info(
                "specialized_agent_success",
                domain=domain.value,
                confidence=result.confidence,
            )
            break

    return results


__all__ = [
    "AgentResult",
    "BaseSpecializedAgent",
    "CodeBlockAgent",
    "ContentCorruptionAgent",
    "ContentRepairAgent",
    "ContentStructureAgent",
    "FallbackAgent",
    "HTMLValidationAgent",
    "ProblemDomain",
    "ProblemRouter",
    "QAExtractionAgent",
    "QualityAssuranceAgent",
    "RepairResult",
    "YAMLFrontmatterAgent",
    "diagnose_and_solve_problems",
    "problem_router",
]
