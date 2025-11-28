"""Domain-specific reflection criteria for self-reflection feature.

This module defines specialized reflection criteria for different content domains,
allowing the reflection system to apply domain-specific quality checks and prompts.
"""

from dataclasses import dataclass


@dataclass
class DomainCriteria:
    """Domain-specific criteria for reflection and revision."""

    name: str
    keywords: list[str]  # For metadata matching (topic/tags)
    quality_checks: list[str]  # Domain-specific quality assessment checks
    revision_thresholds: dict[str, float]  # severity -> threshold mappings
    prompt_additions: str  # Additional prompt content for this domain
    issue_weights: dict[str, float]  # Issue type -> weight for prioritization


# Domain registry with specialized criteria for each domain
DOMAIN_REGISTRY: dict[str, DomainCriteria] = {}


def _register_domain(criteria: DomainCriteria) -> None:
    """Register a domain in the global registry."""
    DOMAIN_REGISTRY[criteria.name] = criteria


# Programming domain - code accuracy, syntax, examples
_register_domain(
    DomainCriteria(
        name="programming",
        keywords=[
            "python",
            "javascript",
            "java",
            "cpp",
            "c++",
            "rust",
            "go",
            "typescript",
            "programming",
            "code",
            "coding",
            "algorithm",
            "data structure",
            "software",
            "development",
            "web development",
            "mobile development",
            "backend",
            "frontend",
        ],
        quality_checks=[
            "Code syntax is correct and executable",
            "Examples are runnable and demonstrate key concepts",
            "Error handling and edge cases are covered",
            "Time/space complexity is appropriate for the problem",
            "Best practices and patterns are followed",
            "Variable/function names are descriptive and consistent",
            "Code comments explain complex logic",
        ],
        revision_thresholds={"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 0.9},
        prompt_additions="""
PROGRAMMING-SPECIFIC CRITERIA:
- Code examples must be syntactically correct and runnable
- Answers should include time/space complexity analysis when relevant
- Error handling and edge cases must be addressed
- Variable naming should follow language conventions
- Code should demonstrate best practices for the language/framework
- Examples should be practical and solve real problems
- Consider whether the code is production-ready or just illustrative""",
        issue_weights={
            "syntax_error": 0.9,  # Critical - code won't run
            "logic_error": 0.8,  # High - incorrect behavior
            "missing_edge_case": 0.7,  # Medium - incomplete solution
            "poor_naming": 0.4,  # Low - readability issue
            "formatting": 0.2,  # Low - style issue
            "missing_comments": 0.3,  # Low - documentation issue
        },
    )
)

# Medical domain - factual accuracy, terminology, sources
_register_domain(
    DomainCriteria(
        name="medical",
        keywords=[
            "medical",
            "medicine",
            "health",
            "clinical",
            "pharma",
            "pharmaceutical",
            "anatomy",
            "physiology",
            "pathology",
            "diagnosis",
            "treatment",
            "drug",
            "surgery",
            "oncology",
            "cardiology",
            "neurology",
            "pediatrics",
            "radiology",
            "laboratory",
            "microbiology",
            "immunology",
            "pharmacology",
            "toxicology",
        ],
        quality_checks=[
            "Medical facts are accurate and up-to-date",
            "Proper medical terminology is used consistently",
            "Source citations are included for clinical claims",
            "Risks and contraindications are mentioned",
            "Evidence-based medicine principles are followed",
            "Differential diagnosis considerations are included",
            "Treatment guidelines are current and appropriate",
        ],
        revision_thresholds={"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.8},
        prompt_additions="""
MEDICAL-SPECIFIC CRITERIA:
- All medical claims must be factually accurate and supported by evidence
- Use proper medical terminology consistently throughout
- Include source citations for clinical recommendations
- Address potential risks, contraindications, and side effects
- Consider patient safety implications of the information
- Ensure information is current (within last 5 years for clinical topics)
- Include differential diagnosis when relevant to symptoms
- Follow evidence-based medicine guidelines""",
        issue_weights={
            "factual_error": 0.9,  # Critical - wrong medical info
            "missing_citation": 0.8,  # High - unsupported claim
            "terminology_error": 0.7,  # Medium - incorrect medical terms
            "missing_risks": 0.8,  # High - safety issue
            "outdated_info": 0.7,  # Medium - stale information
            "incomplete_differential": 0.6,  # Medium - diagnostic oversight
        },
    )
)

# Interview questions domain - completeness, examples, structure
_register_domain(
    DomainCriteria(
        name="interview",
        keywords=[
            "interview",
            "job",
            "career",
            "behavioral",
            "technical",
            "system design",
            "coding interview",
            "case study",
            "leadership",
            "management",
            "senior",
            "principal",
            "architect",
            "consulting",
            "case interview",
            "resume",
            "cv",
        ],
        quality_checks=[
            "Answers demonstrate STAR method (Situation, Task, Action, Result)",
            "Technical concepts are explained clearly for beginners",
            "Practical examples from real experience are included",
            "Answers show problem-solving approach, not just solutions",
            "Leadership and collaboration aspects are covered",
            "Answers are concise yet comprehensive",
            "Follow-up questions are anticipated",
        ],
        revision_thresholds={"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.9},
        prompt_additions="""
INTERVIEW-SPECIFIC CRITERIA:
- Use STAR method for behavioral questions (Situation, Task, Action, Result)
- Include specific examples with measurable outcomes
- Show problem-solving process, not just final answers
- Demonstrate leadership, collaboration, and communication skills
- Anticipate follow-up questions and prepare responses
- Keep answers concise while being comprehensive
- Include both technical depth and business context
- Address edge cases and error handling in technical answers""",
        issue_weights={
            "missing_star": 0.7,  # Medium - structure issue
            "vague_example": 0.6,  # Medium - lacks specificity
            "no_outcome": 0.8,  # High - missing results
            "poor_structure": 0.5,  # Low - organization issue
            "missing_followup": 0.4,  # Low - incomplete answer
            "too_verbose": 0.3,  # Low - conciseness issue
            "missing_context": 0.5,  # Low - lacks business perspective
        },
    )
)

# General domain - fallback for non-specialized content
_register_domain(
    DomainCriteria(
        name="general",
        keywords=[],  # Default fallback
        quality_checks=[
            "Information is accurate and well-supported",
            "Answers are clear and easy to understand",
            "Key concepts are explained adequately",
            "Structure follows logical flow",
            "Content is appropriate for the intended audience",
        ],
        revision_thresholds={"low": 0.4, "medium": 0.7, "high": 0.9, "critical": 0.95},
        prompt_additions="""
GENERAL KNOWLEDGE CRITERIA:
- Ensure factual accuracy of all claims
- Use clear, accessible language appropriate for the audience
- Structure information logically with clear relationships
- Provide sufficient context for understanding
- Consider whether the answer is complete and unambiguous""",
        issue_weights={
            "factual_error": 0.8,
            "unclear_explanation": 0.6,
            "poor_structure": 0.4,
            "missing_context": 0.5,
            "incomplete_answer": 0.7,
        },
    )
)


def get_domain_criteria(domain_name: str) -> DomainCriteria | None:
    """Get domain criteria by name.

    Args:
        domain_name: Name of the domain

    Returns:
        DomainCriteria if found, None otherwise
    """
    return DOMAIN_REGISTRY.get(domain_name)


def list_available_domains() -> list[str]:
    """List all available domain names.

    Returns:
        List of domain names
    """
    return list(DOMAIN_REGISTRY.keys())
