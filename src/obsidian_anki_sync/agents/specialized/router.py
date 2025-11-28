"""Problem router for specialized agents.

Routes problems to appropriate specialized agents with resilience patterns.
"""

from typing import Any

from ...utils.logging import get_logger
from ...utils.resilience import (
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
from .base import FallbackAgent
from .models import AgentResult, ProblemDomain

logger = get_logger(__name__)


class ProblemRouter:
    """Routes problems to appropriate specialized agents."""

    def __init__(
        self,
        circuit_breaker_config: dict[str, dict[str, Any | None]] = None,
        rate_limit_config: dict[str, int | None] = None,
        bulkhead_config: dict[str, int | None] = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize problem router with resilience patterns.

        Args:
            circuit_breaker_config: Per-domain circuit breaker configuration
            rate_limit_config: Per-domain rate limit configuration (calls per minute)
            bulkhead_config: Per-domain bulkhead configuration (max concurrent)
            confidence_threshold: Minimum confidence threshold for validation
        """
        self.agents = {}
        self.circuit_breakers: dict[ProblemDomain, CircuitBreaker] = {}
        self.rate_limiters: dict[ProblemDomain, RateLimiter] = {}
        self.bulkheads: dict[ProblemDomain, Bulkhead] = {}
        self.confidence_validator = ConfidenceValidator(
            min_confidence=confidence_threshold
        )

        self._initialize_agents()
        self._initialize_resilience_patterns(
            circuit_breaker_config or {},
            rate_limit_config or {},
            bulkhead_config or {},
        )

    def _initialize_agents(self):
        """Initialize all specialized agents."""
        from .code import CodeBlockAgent
        from .corruption import ContentCorruptionAgent
        from .html import HTMLValidationAgent
        from .qa import QAExtractionAgent
        from .quality import QualityAssuranceAgent
        from .structure import ContentStructureAgent
        from .yaml import YAMLFrontmatterAgent

        agents = {}

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
        default_rate_limit = 20
        default_bulkhead = 3

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
            rate_limit = rate_limit_config.get(
                domain.value, rate_limit_config.get("default", default_rate_limit)
            )
            self.rate_limiters[domain] = RateLimiter(max_calls_per_minute=rate_limit)

            # Bulkhead
            bulkhead_max = bulkhead_config.get(
                domain.value, bulkhead_config.get("default", default_bulkhead)
            )
            self.bulkheads[domain] = Bulkhead(max_concurrent=bulkhead_max)

    def diagnose_and_route(
        self, content: str, error_context: dict[str, Any]
    ) -> list[tuple[ProblemDomain, float]]:
        """Diagnose the problem and return prioritized list of agents to try.

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

        # Content corruption
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

        # Quality assurance (fallback)
        if not diagnoses:
            diagnoses.append((ProblemDomain.QUALITY_ASSURANCE, 0.5))

        # Sort by confidence (highest first)
        return sorted(diagnoses, key=lambda x: x[1], reverse=True)

    def _detect_content_corruption(self, content: str) -> bool:
        """Detect patterns indicative of actual content corruption.

        Only detects genuine corruption (binary garbage, encoding errors),
        not legitimate markdown/code syntax.
        """
        # Pattern 1: Control characters (binary corruption)
        for char in content:
            if ord(char) < 32 and char not in "\n\r\t":
                return True

        # Pattern 2: Excessive Unicode replacement characters
        if content.count("\ufffd") > 5:
            return True

        return False

    def solve_problem(
        self, domain: ProblemDomain, content: str, context: dict[str, Any]
    ) -> AgentResult:
        """Route problem to the appropriate specialized agent with resilience patterns.

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

            return result

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
            logger.warning("circuit_breaker_open", domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Circuit breaker is open: {e}",
                warnings=["Circuit breaker prevented execution"],
            )

        except RateLimitExceededError as e:
            logger.warning("rate_limit_exceeded", domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Rate limit exceeded: {e}",
                warnings=["Rate limit prevented execution"],
            )

        except ResourceExhaustedError as e:
            logger.warning("bulkhead_exhausted", domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Resources exhausted: {e}",
                warnings=["Bulkhead prevented execution"],
            )

        except LowConfidenceError as e:
            logger.warning("low_confidence_rejected", domain=domain.value, error=str(e))
            return AgentResult(
                success=False,
                reasoning=f"Low confidence rejected: {e}",
                warnings=["Confidence validation failed"],
            )

        except Exception as e:
            logger.error("specialized_agent_failed", domain=domain.value, error=str(e))

            return AgentResult(
                success=False,
                reasoning=f"Agent failed: {e}",
                warnings=["Agent execution failed"],
            )
