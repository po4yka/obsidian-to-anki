"""Base classes for specialized agents."""

from collections.abc import Callable
from typing import Any, Dict

from ...utils.logging import get_logger
from .models import AgentResult, RepairResult

logger = get_logger(__name__)


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
        self.agent = None
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_with_jitter = None

        if enable_retry:
            from ...utils.resilience import RetryWithJitter

            self.retry_with_jitter = RetryWithJitter(
                max_retries=max_retries,
                initial_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0,
                jitter=True,
            )

    def solve(self, content: str, context: Dict[str, Any]) -> AgentResult:
        """Solve the problem - implemented by subclasses.

        This method can be wrapped with retry logic by subclasses if needed.
        """
        raise NotImplementedError

    def _solve_with_retry(
        self, content: str, context: Dict[str, Any], solve_func: Callable
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
            return self.retry_with_jitter.execute(
                solve_func,
                exceptions=(Exception,),
                content=content,
                context=context,
            )
        else:
            return solve_func(content, context)

    def _create_prompt(self, content: str, context: Dict[str, Any]) -> str:
        """Create the prompt for the agent - implemented by subclasses."""
        raise NotImplementedError


class FallbackAgent(BaseSpecializedAgent):
    """Fallback agent for when specialized agents are not available."""

    def __init__(self, reason: str):
        super().__init__()
        self.reason = reason

    def solve(self, content: str, context: Dict[str, Any]) -> AgentResult:
        """Return failure result with explanation."""
        from .models import AgentResult

        return AgentResult(
            success=False,
            reasoning=f"Agent not available: {self.reason}",
            warnings=["Specialized agent dependency not met"],
        )
