"""Specialized LLM agents for handling different types of note processing problems.

This module provides a collection of specialized agents, each focused on a specific
type of issue that can occur during note processing. This prevents failures by
routing problems to the appropriate expert agent.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..utils.logging import get_logger
from ..utils.resilience import (
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    ConfidenceValidator,
    LowConfidenceError,
    RateLimiter,
    RateLimitExceededError,
    ResourceExhaustedError,
)

logger = get_logger(__name__)


@dataclass
class RepairResult:
    """Result from content repair operation."""

    success: bool
    repaired_content: str | None = None
    confidence: float = 0.0
    reasoning: str = ""
    error_message: str | None = None
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


class ContentRepairAgent:
    """Simple content repair agent for specialized agents."""

    def __init__(self, model: str = "qwen3:8b"):
        self.model = model

    def generate_repair(
        self, content: str, prompt: str, max_retries: int = 2
    ) -> RepairResult:
        """Generate content repair using LLM."""
        return RepairResult(
            success=False,
            error_message="LLM repair not implemented in test environment",
            warnings=["LLM-based repair requires full LLM integration"],
        )


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
    metadata: dict[str, Any | None] | None = None
    qa_pairs: list[dict[str, Any | None]] | None = None
    confidence: float = 0.0
    reasoning: str = ""
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


class ProblemRouter:
    """Routes problems to appropriate specialized agents."""

    def __init__(
        self,
        circuit_breaker_config: dict[str, dict[str, Any | None]] | None = None,
        rate_limit_config: dict[str, int | None] | None = None,
        bulkhead_config: dict[str, int | None] | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize problem router with resilience patterns.

        Args:
            circuit_breaker_config: Per-domain circuit breaker configuration
            rate_limit_config: Per-domain rate limit configuration (calls per minute)
            bulkhead_config: Per-domain bulkhead configuration (max concurrent)
            confidence_threshold: Minimum confidence threshold for validation
        """
        self.agents: dict[ProblemDomain, Any] = {}
        self.circuit_breakers: dict[ProblemDomain, CircuitBreaker] = {}
        self.rate_limiters: dict[ProblemDomain, RateLimiter] = {}
        self.bulkheads: dict[ProblemDomain, Bulkhead] = {}
        self.confidence_validator = ConfidenceValidator(
            min_confidence=confidence_threshold
        )

        self._initialize_agents()
        # Convert None values to empty dicts and filter None values from configs
        cb_config = circuit_breaker_config or {}
        rate_config: dict[str, int] = {k: v for k, v in (
            rate_limit_config or {}).items() if v is not None}
        bulkhead_config_clean: dict[str, int] = {k: v for k, v in (
            bulkhead_config or {}).items() if v is not None}
        self._initialize_resilience_patterns(
            cb_config,
            rate_config,
            bulkhead_config_clean,
        )

    def _initialize_agents(self) -> None:
        """Initialize all specialized agents."""
        agents: dict[ProblemDomain, Any] = {}

        # Initialize agents that don't require external dependencies
        agents[ProblemDomain.YAML_FRONTMATTER] = YAMLFrontmatterAgent()
        agents[ProblemDomain.CONTENT_STRUCTURE] = ContentStructureAgent()
        agents[ProblemDomain.CONTENT_CORRUPTION] = ContentCorruptionAgent()
        agents[ProblemDomain.CODE_BLOCKS] = CodeBlockAgent()
        agents[ProblemDomain.HTML_VALIDATION] = HTMLValidationAgent()
        agents[ProblemDomain.QUALITY_ASSURANCE] = QualityAssuranceAgent()

        # Try to initialize agents that require dependencies
        try:
            agents[ProblemDomain.QA_EXTRACTION] = QAExtractionAgent()
        except Exception as e:
            logger.warning(
                "qa_extraction_agent_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Create a fallback agent
            agents[ProblemDomain.QA_EXTRACTION] = FallbackAgent(
                "QA extraction requires LLM provider"
            )

        self.agents = agents

    def _initialize_resilience_patterns(
        self,
        circuit_breaker_config: dict[str, dict[str, Any]],
        rate_limit_config: dict[str, int],
        bulkhead_config: dict[str, int],
    ) -> None:
        """Initialize resilience patterns for each agent domain.

        Args:
            circuit_breaker_config: Per-domain circuit breaker config
            rate_limit_config: Per-domain rate limit config
            bulkhead_config: Per-domain bulkhead config
        """
        default_cb_config = CircuitBreakerConfig()
        default_rate_limit = 20  # calls per minute
        default_bulkhead = 3  # max concurrent

        for domain in ProblemDomain:
            # Circuit breaker
            cb_config_dict = circuit_breaker_config.get(
                domain.value, circuit_breaker_config.get("default", {})
            )
            if cb_config_dict:
                cb_config = CircuitBreakerConfig(**cb_config_dict)
            else:
                cb_config = default_cb_config

            self.circuit_breakers[domain] = CircuitBreaker(
                name=f"{domain.value}_agent", config=cb_config
            )

            # Rate limiter
            rate_limit_raw = rate_limit_config.get(
                domain.value, rate_limit_config.get(
                    "default", default_rate_limit)
            )
            rate_limit = rate_limit_raw if rate_limit_raw is not None else default_rate_limit
            self.rate_limiters[domain] = RateLimiter(
                max_calls_per_minute=rate_limit)

            # Bulkhead
            bulkhead_max_raw = bulkhead_config.get(
                domain.value, bulkhead_config.get("default", default_bulkhead)
            )
            bulkhead_max = bulkhead_max_raw if bulkhead_max_raw is not None else default_bulkhead
            self.bulkheads[domain] = Bulkhead(max_concurrent=bulkhead_max)

    def diagnose_and_route(
        self, content: str, error_context: dict[str, Any]
    ) -> list[tuple[ProblemDomain, float]]:
        """
        Diagnose the problem and return prioritized list of agents to try.

        Args:
            content: The problematic content
            error_context: Context about the error (error message, stage, etc.)

        Returns:
            List of (domain, confidence) tuples, ordered by priority
        """
        diagnoses = []

        error_msg = error_context.get("error_message", "").lower()
        processing_stage = error_context.get("processing_stage", "")

        # YAML/Frontmatter issues
        if any(
            keyword in error_msg
            for keyword in ["yaml", "frontmatter", "mapping values", "expected"]
        ):
            diagnoses.append((ProblemDomain.YAML_FRONTMATTER, 0.9))

        # Content structure issues
        if any(
            keyword in error_msg
            for keyword in ["missing", "section", "required for language"]
        ):
            diagnoses.append((ProblemDomain.CONTENT_STRUCTURE, 0.8))

        # Content corruption (patterns like a1a1a1, corruption markers)
        if self._detect_content_corruption(content):
            diagnoses.append((ProblemDomain.CONTENT_CORRUPTION, 0.85))

        # Code block issues
        if any(keyword in error_msg for keyword in ["code fence", "backtick", "```"]):
            diagnoses.append((ProblemDomain.CODE_BLOCKS, 0.8))

        # HTML validation issues
        if processing_stage == "card_generation" and any(
            keyword in error_msg for keyword in ["html", "language-", "<code>", "<pre>"]
        ):
            diagnoses.append((ProblemDomain.HTML_VALIDATION, 0.9))

        # QA extraction issues
        if processing_stage in ["parsing", "indexing"] and "qa" in error_msg.lower():
            diagnoses.append((ProblemDomain.QA_EXTRACTION, 0.7))

        # Quality assurance (fallback for general issues)
        if not diagnoses:
            diagnoses.append((ProblemDomain.QUALITY_ASSURANCE, 0.5))

        # Sort by confidence (highest first)
        return sorted(diagnoses, key=lambda x: x[1], reverse=True)

    def _detect_content_corruption(self, content: str) -> bool:
        """
        Detect patterns indicative of actual content corruption.

        Only detects genuine corruption (binary garbage, encoding errors),
        not legitimate markdown/code syntax.
        """
        # Pattern 1: Control characters (binary corruption)
        # Exclude legitimate whitespace (tab, newline, carriage return)
        for char in content:
            if ord(char) < 32 and char not in "\n\r\t":
                return True

        # Pattern 2: Excessive Unicode replacement characters
        # U+FFFD indicates characters that couldn't be decoded
        if content.count("\ufffd") > 5:
            return True

        return False

    def solve_problem(
        self, domain: ProblemDomain, content: str, context: dict[str, Any]
    ) -> AgentResult:
        """
        Route problem to the appropriate specialized agent with resilience patterns.

        Args:
            domain: The problem domain to address
            content: The content to repair
            context: Additional context about the problem

        Returns:
            AgentResult with the repair attempt
        """
        if domain not in self.agents:
            return AgentResult(
                success=False,
                reasoning=f"No agent available for domain: {domain}",
                warnings=["Unknown problem domain"],
            )

        agent = self.agents[domain]

        # Wrap execution with resilience patterns
        def _execute_agent() -> AgentResult:
            """Execute agent with resilience protection."""
            logger.info(
                "routing_to_specialized_agent",
                domain=domain.value,
                agent_type=agent.__class__.__name__,
                content_length=len(content),
            )

            result = agent.solve(content, context)

            # Validate confidence before returning
            validation = self.confidence_validator.validate(result)
            if not validation.is_valid:
                logger.warning(
                    "confidence_validation_failed",
                    domain=domain.value,
                    confidence=result.confidence,
                    reason=validation.reason,
                    patterns=validation.suspicious_patterns,
                )
                raise LowConfidenceError(validation.reason)

            logger.info(
                "specialized_agent_result",
                domain=domain.value,
                success=result.success,
                confidence=result.confidence,
                warnings=len(result.warnings),
            )

            return result  # type: ignore[no-any-return]

        try:
            # Execute with bulkhead isolation
            def _execute_with_bulkhead() -> AgentResult:
                """Execute with bulkhead protection."""
                # Check rate limit
                if not self.rate_limiters[domain].acquire():
                    logger.warning("rate_limit_exceeded", domain=domain.value)
                    raise RateLimitExceededError(
                        f"Rate limit exceeded for {domain.value}"
                    )

                # Execute with circuit breaker
                return self.circuit_breakers[domain].call(_execute_agent)

            # Execute with bulkhead
            result = self.bulkheads[domain].execute(_execute_with_bulkhead)

            return result

        except CircuitBreakerError as e:
            logger.warning("circuit_breaker_open",
                           domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Circuit breaker is open: {e}",
                warnings=["Circuit breaker prevented execution"],
            )

        except RateLimitExceededError as e:
            logger.warning("rate_limit_exceeded",
                           domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Rate limit exceeded: {e}",
                warnings=["Rate limit prevented execution"],
            )

        except ResourceExhaustedError as e:
            logger.warning("bulkhead_exhausted",
                           domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Resources exhausted: {e}",
                warnings=["Bulkhead prevented execution"],
            )

        except LowConfidenceError as e:
            logger.warning("low_confidence_rejected",
                           domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Low confidence rejected: {e}",
                warnings=["Confidence validation failed"],
            )

        except Exception as e:
            logger.error("specialized_agent_failed",
                         domain=domain.value, error=str(e))

            return AgentResult(
                success=False,
                reasoning=f"Agent failed: {e}",
                warnings=["Agent execution failed"],
            )


class BaseSpecializedAgent:
    """Base class for specialized agents."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        enable_retry: bool = True,
        max_retries: int = 3,
    ):
        """Initialize base specialized agent.

        Args:
            model: Model name for LLM-based agents
            enable_retry: Whether to enable retry with jitter
            max_retries: Maximum retry attempts
        """
        self.model = model
        self.agent = None  # Will be initialized by subclasses
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_with_jitter = None

        if enable_retry:
            from ..utils.resilience import RetryWithJitter

            self.retry_with_jitter = RetryWithJitter(
                max_retries=max_retries,
                initial_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True,
            )

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Solve the problem - implemented by subclasses.

        This method can be wrapped with retry logic by subclasses if needed.
        """
        raise NotImplementedError

    def _solve_with_retry(
        self, content: str, context: dict[str, Any], solve_func: Callable
    ) -> AgentResult:
        """Execute solve function with retry and jitter.

        Args:
            content: Content to process
            context: Error context
            solve_func: Function to execute (should return AgentResult)

        Returns:
            AgentResult from solve function
        """
        if self.retry_with_jitter:
            return self.retry_with_jitter.execute(  # type: ignore[no-any-return]
                solve_func,
                exceptions=(Exception,),
                content=content,
                context=context,
            )
        else:
            return solve_func(content, context)  # type: ignore[no-any-return]

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create the prompt for the agent - implemented by subclasses."""
        raise NotImplementedError


class FallbackAgent(BaseSpecializedAgent):
    """Fallback agent for when specialized agents are not available."""

    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Return failure result with explanation."""
        return AgentResult(
            success=False,
            reasoning=f"Agent not available: {self.reason}",
            warnings=["Specialized agent dependency not met"],
        )


class YAMLFrontmatterAgent(BaseSpecializedAgent):
    """Agent specialized in repairing YAML frontmatter issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(
            model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair YAML frontmatter issues."""
        prompt = self._create_prompt(content, context)

        try:
            if self.agent is None:
                return AgentResult(
                    success=False,
                    reasoning="ContentRepairAgent not initialized",
                    warnings=["Agent not available"],
                )
            result = self.agent.generate_repair(
                content=content, prompt=prompt, max_retries=2
            )

            if result.success and result.repaired_content:
                # Validate the repair
                if self._validate_yaml_repair(result.repaired_content):
                    return AgentResult(
                        success=True,
                        content=result.repaired_content,
                        confidence=result.confidence,
                        reasoning=result.reasoning,
                        warnings=result.warnings,
                    )
                else:
                    return AgentResult(
                        success=False,
                        reasoning="YAML repair validation failed",
                        warnings=["Repaired content is still invalid YAML"],
                    )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message or "YAML repair failed",
                    warnings=["YAML frontmatter repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"YAML agent error: {e}",
                warnings=["YAML agent execution failed"],
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create YAML repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a YAML frontmatter repair specialist. Fix the corrupted YAML frontmatter in this Obsidian note.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify and fix YAML syntax errors (missing quotes, indentation, etc.)
2. Preserve all metadata fields (id, title, tags, etc.)
3. Ensure proper YAML structure with --- delimiters
4. Fix multi-line values that are improperly formatted
5. Maintain the exact same semantic meaning

Return ONLY the repaired frontmatter section, properly formatted as valid YAML."""

    def _validate_yaml_repair(self, content: str) -> bool:
        """Validate that the YAML repair is correct."""
        try:
            import frontmatter

            # Try to parse the frontmatter
            post = frontmatter.loads(content)
            return post.metadata is not None
        except Exception:
            return False


class ContentStructureAgent(BaseSpecializedAgent):
    """Agent specialized in repairing content structure issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(
            model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair content structure issues like missing sections."""
        # First try rule-based repair
        rule_based_result = self._rule_based_repair(content, context)
        if rule_based_result.success:
            return rule_based_result

        # Fall back to LLM-based repair
        prompt = self._create_prompt(content, context)

        try:
            if self.agent is None:
                return AgentResult(
                    success=False,
                    reasoning="ContentRepairAgent not initialized",
                    warnings=["Agent not available"],
                )
            result = self.agent.generate_repair(
                content=content, prompt=prompt, max_retries=2
            )

            if result.success and result.repaired_content:
                return AgentResult(
                    success=True,
                    content=result.repaired_content,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    warnings=result.warnings,
                )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message or "Content structure repair failed",
                    warnings=["Content structure repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Content structure agent error: {e}",
                warnings=["Content structure agent execution failed"],
            )

    def _rule_based_repair(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Try rule-based repair first."""
        try:
            # Extract languages from frontmatter
            languages = self._extract_languages_from_frontmatter(content)
            if not languages:
                return AgentResult(
                    success=False, reasoning="Could not determine languages"
                )

            lines = content.splitlines()
            repaired_lines = list(lines)

            # Find where to insert missing sections
            frontmatter_end = self._find_frontmatter_end(lines)

            # Check for missing question sections
            for lang in languages:
                question_marker = f"# Question ({lang.upper()})"
                if not any(question_marker in line for line in lines):
                    # Add missing question section
                    insert_pos = frontmatter_end + 1
                    repaired_lines.insert(insert_pos, "")
                    repaired_lines.insert(insert_pos + 1, question_marker)
                    repaired_lines.insert(insert_pos + 2, "")
                    repaired_lines.insert(
                        insert_pos + 3, f"[Question content in {lang.upper()}]"
                    )

            # Check for missing answer sections
            for lang in languages:
                answer_marker = f"## Answer ({lang.upper()})"
                if not any(answer_marker in line for line in lines):
                    # Add missing answer section after question
                    question_idx = next(
                        (
                            i
                            for i, line in enumerate(repaired_lines)
                            if question_marker in line
                        ),
                        -1,
                    )
                    if question_idx >= 0:
                        # Find end of question section
                        insert_pos = question_idx + 1
                        while insert_pos < len(repaired_lines) and not repaired_lines[
                            insert_pos
                        ].startswith("##"):
                            insert_pos += 1

                        repaired_lines.insert(insert_pos, "")
                        repaired_lines.insert(insert_pos + 1, answer_marker)
                        repaired_lines.insert(insert_pos + 2, "")
                        repaired_lines.insert(
                            insert_pos +
                            3, f"[Answer content in {lang.upper()}]"
                        )

            repaired_content = "\n".join(repaired_lines)

            if repaired_content != content:
                return AgentResult(
                    success=True,
                    content=repaired_content,
                    confidence=0.8,
                    reasoning="Added missing structural sections using rules",
                    warnings=["Added placeholder content for missing sections"],
                )

            return AgentResult(
                success=False,
                reasoning="No structural issues found with rule-based approach",
            )

        except Exception as e:
            return AgentResult(
                success=False, reasoning=f"Rule-based repair failed: {e}"
            )

    def _extract_languages_from_frontmatter(self, content: str) -> list[str]:
        """Extract language tags from frontmatter."""
        lines = content.splitlines()
        in_frontmatter = False

        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                if not in_frontmatter:
                    break
                continue

            if in_frontmatter and line.startswith("language_tags:"):
                # Extract from YAML list format
                match = re.search(r"language_tags:\s*\[(.*?)\]", line)
                if match:
                    return [
                        lang.strip().strip("\"'") for lang in match.group(1).split(",")
                    ]

        return ["en"]  # Default fallback

    def _find_frontmatter_end(self, lines: list[str]) -> int:
        """Find the end of frontmatter."""
        for i, line in enumerate(lines):
            if line.strip() == "---" and i > 0:
                return i
        return 0

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create content structure repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a content structure repair specialist. Fix missing or malformed sections in this Obsidian interview note.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify missing required sections based on language_tags in frontmatter
2. Add missing question headers (# Question (EN), # Вопрос (RU), etc.)
3. Add missing answer headers (## Answer (EN), ## Ответ (RU), etc.)
4. Ensure proper bilingual structure
5. Do NOT modify the existing content, only add missing structural elements

Return the complete repaired content with proper structure."""


class ContentCorruptionAgent(BaseSpecializedAgent):
    """Agent specialized in repairing content corruption issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(
            model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair content corruption like repetitive patterns."""
        # First try rule-based corruption repair
        rule_based_result = self._rule_based_corruption_repair(content)
        if rule_based_result.success:
            return rule_based_result

        # Fall back to LLM-based repair
        prompt = self._create_prompt(content, context)

        try:
            if self.agent is None:
                return AgentResult(
                    success=False,
                    reasoning="ContentRepairAgent not initialized",
                    warnings=["Agent not available"],
                )
            result = self.agent.generate_repair(
                content=content, prompt=prompt, max_retries=2
            )

            if result.success and result.repaired_content:
                return AgentResult(
                    success=True,
                    content=result.repaired_content,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    warnings=result.warnings,
                )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message
                    or "Content corruption repair failed",
                    warnings=["Content corruption repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Content corruption agent error: {e}",
                warnings=["Content corruption agent execution failed"],
            )

    def _rule_based_corruption_repair(self, content: str) -> AgentResult:
        """Try rule-based corruption pattern removal."""
        try:
            original_content = content

            # Remove repetitive alphanumeric corruption patterns
            # Remove "a1", "b2", etc. patterns
            content = re.sub(r"[a-zA-Z]\d{1,2}\s*", "", content)
            # Remove "1a", "2b", etc. patterns
            content = re.sub(r"\d{1,2}[a-zA-Z]\s*", "", content)
            # Remove excessive character repetition (more than 2)
            content = re.sub(r"(.)\1{2,}", r"\1", content)

            # Clean up spacing issues caused by removals
            content = re.sub(r"\s+", " ", content)  # Normalize multiple spaces
            # Remove excessive blank lines
            content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

            if content != original_content:
                return AgentResult(
                    success=True,
                    content=content,
                    confidence=0.7,
                    reasoning="Removed corruption patterns using rules",
                    warnings=["Content was cleaned of corruption patterns"],
                )

            return AgentResult(
                success=False, reasoning="No corruption patterns detected"
            )

        except Exception as e:
            return AgentResult(
                success=False, reasoning=f"Rule-based corruption repair failed: {e}"
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create content corruption repair prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a content corruption repair specialist. Fix corrupted text patterns in this document.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify corrupted text patterns (repetitive characters like "a1a1a1", "b2b2b2", etc.)
2. Replace corrupted sections with appropriate Russian/English text
3. Maintain the document's meaning and structure
4. Preserve code blocks and technical content
5. Only repair obviously corrupted text, leave valid content unchanged

Return the complete repaired content with corruption removed."""


class CodeBlockAgent(BaseSpecializedAgent):
    """Agent specialized in repairing code block issues."""

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair code block formatting and fence issues."""
        repaired_content = self._repair_code_fences(content)

        if repaired_content != content:
            return AgentResult(
                success=True,
                content=repaired_content,
                confidence=0.8,
                reasoning="Repaired code fence issues with pattern matching",
                warnings=[],
            )
        else:
            return AgentResult(
                success=False,
                reasoning="No code fence issues detected or repairable",
                warnings=["Code block agent could not find issues to repair"],
            )

    def _repair_code_fences(self, content: str) -> str:
        """Repair code fence issues using pattern matching."""
        lines = content.splitlines()
        repaired_lines = []
        fence_stack: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if fence_stack:
                    # Close current fence
                    fence_stack.pop()
                    repaired_lines.append(line)
                else:
                    # Start new fence
                    fence_stack.append(stripped)
                    repaired_lines.append(line)
            else:
                repaired_lines.append(line)

        # Close any remaining open fences
        while fence_stack:
            repaired_lines.append("```")
            fence_stack.pop()

        return "\n".join(repaired_lines)


class HTMLValidationAgent(BaseSpecializedAgent):
    """Agent specialized in repairing HTML validation issues."""

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Repair HTML validation issues."""
        # Import here to avoid circular imports
        from ..apf.html_generator import HTMLTemplateGenerator

        try:
            # Try to extract and regenerate HTML using templates
            html_generator = HTMLTemplateGenerator()
            card_data = self._extract_card_data(content)

            if card_data:
                result = html_generator.generate_card_html(card_data)

                if result.is_valid:
                    return AgentResult(
                        success=True,
                        content=result.html,
                        confidence=0.9,
                        reasoning="Regenerated HTML using structured templates",
                        warnings=result.warnings,
                    )

            return AgentResult(
                success=False,
                reasoning="Could not extract card data for HTML regeneration",
                warnings=["HTML validation agent needs structured card data"],
            )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"HTML validation agent error: {e}",
                warnings=["HTML validation agent execution failed"],
            )

    def _extract_card_data(self, content: str) -> dict[str, Any | None] | None:
        """Extract card data from HTML content for regeneration."""
        # This would need more sophisticated parsing
        # For now, return None to indicate we can't extract
        return None


class QAExtractionAgent(BaseSpecializedAgent):
    """Agent specialized in Q/A pair extraction from corrupted content."""

    def __init__(self) -> None:
        super().__init__()
        self.qa_agent = None

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """Extract Q/A pairs from content."""
        # QA extraction requires full LLM integration with a provider
        # This is not available in the specialized agent context
        return AgentResult(
            success=False,
            reasoning="QA extraction requires LLM provider integration. Use the main QAExtractorAgent from qa_extractor.py instead.",
            warnings=[
                "QA extraction agent not available in specialized agents context"
            ],
        )


class QualityAssuranceAgent(BaseSpecializedAgent):
    """General quality assurance agent for unspecified issues."""

    def __init__(self) -> None:
        super().__init__()
        self.agent = ContentRepairAgent(
            model=self.model)  # type: ignore[assignment]

    def solve(self, content: str, context: dict[str, Any]) -> AgentResult:
        """General quality assurance and repair."""
        prompt = self._create_prompt(content, context)

        try:
            if self.agent is None:
                return AgentResult(
                    success=False,
                    reasoning="ContentRepairAgent not initialized",
                    warnings=["Agent not available"],
                )
            result = self.agent.generate_repair(
                content=content, prompt=prompt, max_retries=2
            )

            if result.success and result.repaired_content:
                return AgentResult(
                    success=True,
                    content=result.repaired_content,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    warnings=result.warnings,
                )
            else:
                return AgentResult(
                    success=False,
                    reasoning=result.error_message or "Quality assurance repair failed",
                    warnings=["Quality assurance repair unsuccessful"],
                )

        except Exception as e:
            return AgentResult(
                success=False,
                reasoning=f"Quality assurance agent error: {e}",
                warnings=["Quality assurance agent execution failed"],
            )

    def _create_prompt(self, content: str, context: dict[str, Any]) -> str:
        """Create general quality assurance prompt."""
        error_msg = context.get("error_message", "")

        return f"""You are a general quality assurance specialist. Fix any issues in this document to make it valid and usable.

ERROR: {error_msg}

CONTENT:
{content}

INSTRUCTIONS:
1. Identify and fix any structural or formatting issues
2. Ensure the document follows proper markdown conventions
3. Preserve all meaningful content
4. Make minimal changes necessary to resolve the issues

Return the repaired content."""


# Global router instance
problem_router = ProblemRouter()


def diagnose_and_solve_problems(
    content: str, error_context: dict[str, Any]
) -> list[AgentResult]:
    """
    Diagnose problems and attempt solutions using specialized agents.

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
