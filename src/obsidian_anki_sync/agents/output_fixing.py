"""OutputFixingParser pattern for PydanticAI models.

This module implements LangChain's OutputFixingParser pattern for PydanticAI,
providing automatic retry with fixing for structured outputs that fail validation.
"""

import json
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from obsidian_anki_sync.utils.logging import get_logger

from .exceptions import StructuredOutputError
from .models import normalize_enrichment_label

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class OutputFixingParser(Generic[T]):
    """Wrapper for PydanticAI agents that automatically fixes validation errors.

    Implements LangChain's OutputFixingParser pattern:
    - Wraps an agent that returns structured outputs
    - On validation failure, uses an LLM to fix the output
    - Retries with fixed output
    - Tracks repair success rates

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel("gpt-4o-mini")
        agent = Agent(model, output_type=MyModel)

        fixing_parser = OutputFixingParser(agent, fix_model=model)
        result = await fixing_parser.run("Generate structured output")
        ```
    """

    def __init__(
        self,
        agent: Agent[T],
        fix_model: Model | None = None,
        max_fix_attempts: int = 2,
        fix_temperature: float = 0.0,
        repair_confidence_threshold: float = 0.7,
    ):
        """Initialize OutputFixingParser.

        Args:
            agent: PydanticAI agent to wrap
            fix_model: Model to use for fixing (defaults to agent's model)
            max_fix_attempts: Maximum number of fix attempts
            fix_temperature: Temperature for fix model
            repair_confidence_threshold: Minimum confidence to accept repair
        """
        self.agent = agent
        self.fix_model = fix_model or agent.model
        self.max_fix_attempts = max_fix_attempts
        self.fix_temperature = fix_temperature
        self.repair_confidence_threshold = repair_confidence_threshold

        # Track repair metrics
        self.repair_attempts = 0
        self.repair_successes = 0
        self.repair_failures = 0

    async def run(
        self,
        prompt: str,
        system: str | None = None,
        deps: Any = None,
        **kwargs: Any,
    ) -> T:
        """Run agent with automatic output fixing.

        Args:
            prompt: Input prompt
            system: Optional system prompt
            deps: Optional dependencies for agent
            **kwargs: Additional arguments for agent.run()

        Returns:
            Validated structured output

        Raises:
            StructuredOutputError: If all fix attempts fail
        """
        attempt = 0
        last_error: Exception | None = None
        last_output: str | None = None

        while attempt <= self.max_fix_attempts:
            try:
                if attempt == 0:
                    # First attempt: normal execution
                    # Note: output_retries is configured in Agent constructor, not in run()
                    if deps is not None:
                        result = await self.agent.run(prompt, deps=deps, **kwargs)
                    else:
                        result = await self.agent.run(
                            prompt,
                            instructions=system,
                            **kwargs,
                        )
                    return result.output  # type: ignore[no-any-return]
                else:
                    # Retry with fixed prompt
                    if last_output is None:
                        msg = "No previous output to fix"
                        raise ValueError(msg)

                    fixed_prompt = await self._fix_output(
                        prompt=prompt,
                        system=system,
                        invalid_output=last_output,
                        error=str(last_error),
                        attempt=attempt,
                    )

                    # Retry with fixed prompt
                    if deps is not None:
                        result = await self.agent.run(
                            fixed_prompt,
                            deps=deps,
                            **kwargs,
                        )
                    else:
                        result = await self.agent.run(
                            fixed_prompt,
                            instructions=system,
                            **kwargs,
                        )
                    self.repair_successes += 1
                    logger.info(
                        "output_fixing_success",
                        attempt=attempt,
                        error_type=(
                            type(last_error).__name__ if last_error else "unknown"
                        ),
                    )
                    return result.output  # type: ignore[no-any-return]

            except (ValueError, StructuredOutputError, UnexpectedModelBehavior) as e:
                # PydanticAI raises UnexpectedModelBehavior for output validation errors
                # It may also raise ValueError for other validation issues
                # Our code raises StructuredOutputError
                last_error = e
                last_output = self._extract_output_from_error(e)

                # Extract additional details from UnexpectedModelBehavior
                error_body = getattr(e, "body", None)
                error_message = getattr(e, "message", str(e))

                # Get output type name for debugging
                output_type_name = "unknown"
                if hasattr(self.agent, "output_type"):
                    output_type_name = getattr(
                        self.agent.output_type, "__name__", str(self.agent.output_type)
                    )

                # Log the raw LLM output that failed validation for debugging
                logger.warning(
                    "output_validation_failed",
                    attempt=attempt,
                    error_type=type(e).__name__,
                    error_message=error_message[:500],
                    error_body_preview=str(error_body)[:300] if error_body else None,
                    raw_output_preview=last_output[:500] if last_output else None,
                    output_length=len(last_output) if last_output else 0,
                    output_type=output_type_name,
                )

                (
                    last_output,
                    validated_output,
                ) = self._auto_correct_literal_errors(last_output)
                if validated_output is not None:
                    logger.info(
                        "output_literal_autocorrect_success",
                        attempt=attempt,
                        error_type=(
                            type(last_error).__name__ if last_error else "unknown"
                        ),
                    )
                    return validated_output
                attempt += 1
                self.repair_attempts += 1

                if attempt > self.max_fix_attempts:
                    self.repair_failures += 1
                    logger.error(
                        "output_fixing_exhausted",
                        attempts=attempt,
                        max_attempts=self.max_fix_attempts,
                        error_type=type(e).__name__,
                        error_message=error_message[:500],
                        error_body_preview=str(error_body)[:300]
                        if error_body
                        else None,
                        raw_output_preview=last_output[:300] if last_output else None,
                        output_type=output_type_name,
                    )
                    msg = f"Failed to fix output after {attempt} attempts: {e!s}"
                    raise StructuredOutputError(
                        msg,
                        details={
                            "attempts": attempt,
                            "error": str(e),
                            "output_type": output_type_name,
                        },
                    ) from e

                logger.info(
                    "output_fixing_retry",
                    attempt=attempt,
                    max_attempts=self.max_fix_attempts,
                    output_type=output_type_name,
                    error=error_message[:200],
                )

            except Exception as e:
                # Non-validation errors: don't retry
                # Get output type name for debugging
                output_type_name = "unknown"
                if hasattr(self.agent, "output_type"):
                    output_type_name = getattr(
                        self.agent.output_type, "__name__", str(self.agent.output_type)
                    )
                logger.error(
                    "output_fixing_unexpected_error",
                    error=str(e)[:500],
                    error_type=type(e).__name__,
                    output_type=output_type_name,
                )
                raise
        # This should never be reached, but mypy needs it
        msg = "Unexpected loop exit"
        raise StructuredOutputError(msg, details={})

    async def _fix_output(
        self,
        prompt: str,
        system: str | None,
        invalid_output: str,
        error: str,
        attempt: int,
    ) -> str:
        """Use LLM to fix invalid output by improving the prompt.

        Args:
            prompt: Original prompt
            system: Original system prompt
            invalid_output: Invalid output that needs fixing
            error: Validation error message
            attempt: Current attempt number

        Returns:
            Improved prompt string that should generate valid output
        """
        fix_prompt = f"""<task>
The previous attempt to generate structured output failed with this validation error:

Error: {error}

Original prompt was:
{prompt[:1000]}

The model generated this invalid output:
{invalid_output[:1000]}

Expected output format: {self._get_expected_format()}

Please regenerate the prompt to ensure the model produces valid structured output matching the expected format exactly.
Focus on:
1. Being more explicit about the required format
2. Providing examples if helpful
3. Emphasizing required fields and their types
4. Ensuring JSON validity

Return an improved prompt that will generate valid output.
</task>"""

        fix_system = """<role>
You are a prompt improvement agent. Your job is to fix prompts that generate invalid structured outputs.
Improve prompts to ensure they generate valid, correctly formatted outputs.
</role>

<approach>
1. Analyze the validation error carefully
2. Identify what format requirements were violated
3. Improve the prompt to be more explicit about requirements
4. Add format constraints and examples if needed
5. Return the improved prompt
</approach>"""

        try:
            # Use fix model to generate improved prompt
            from pydantic_ai import Agent as FixAgent

            fix_agent = FixAgent(self.fix_model, output_type=str)
            result = await fix_agent.run(fix_prompt, instructions=fix_system)
            improved_prompt = result.output

            logger.info(
                "output_fixing_prompt_improved",
                attempt=attempt,
                original_length=len(prompt),
                improved_length=len(improved_prompt),
            )

            return str(improved_prompt)

        except Exception as e:
            logger.error("fix_output_llm_failed", error=str(e))
            # Fallback: return original prompt with error context appended
            return str(
                f"{prompt}\n\nIMPORTANT: Previous attempt failed with error: {error}. Please ensure output matches format exactly: {self._get_expected_format()}"
            )

    def _get_expected_format(self) -> str:
        """Get description of expected output format."""
        # Get the output type from agent - this is a generic type parameter
        # We need to introspect it from the agent's configuration
        try:
            # Try to get it from the agent if it has output_type attribute
            if hasattr(self.agent, "output_type"):
                # type: ignore[attr-defined]
                result_type = self.agent.output_type
                # Check if it's a Pydantic model
                if isinstance(result_type, type) and issubclass(result_type, BaseModel):
                    # Try to get schema
                    try:
                        schema = result_type.model_json_schema()
                        return json.dumps(schema, indent=2)
                    except Exception:
                        return f"Pydantic model: {result_type.__name__}"
                else:
                    return "String output"
            else:
                # Fallback for generic agents
                return "Structured output"
        except (AttributeError, TypeError):
            return "Structured output"

    def _extract_output_from_error(self, error: Exception) -> str:
        """Extract output string from validation error.

        Args:
            error: Exception (ValueError, StructuredOutputError, or UnexpectedModelBehavior)

        Returns:
            Output string if extractable, empty string otherwise
        """
        # Try to extract from UnexpectedModelBehavior.body (PydanticAI specific)
        if hasattr(error, "body") and error.body:
            return str(error.body)

        # Try to extract from error details dict (our StructuredOutputError)
        if hasattr(error, "details") and isinstance(error.details, dict):
            if "raw_output" in error.details:
                return str(error.details["raw_output"])
            if "output" in error.details:
                return str(error.details["output"])

        # Try to extract from error message
        error_str = str(error)

        # Look for JSON-like content in error message
        json_match = re.search(r"\{.*\}", error_str, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Fallback: return error message
        return error_str[:500]

    def _fallback_fix(self, invalid_output: str, error: str) -> str:
        """Fallback fix method when LLM fixing fails.

        Args:
            invalid_output: Invalid output
            error: Error message

        Returns:
            Attempted fix (may still be invalid)
        """
        # Try basic JSON fixes
        try:
            # Try to parse as JSON and fix common issues
            if invalid_output.strip().startswith("```"):
                # Remove markdown code fences
                lines = invalid_output.split("\n")
                filtered = [
                    line for line in lines if not line.strip().startswith("```")
                ]
                invalid_output = "\n".join(filtered)

            # Try to parse and re-stringify
            parsed = json.loads(invalid_output)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            # Return original if we can't fix it
            logger.warning("fallback_fix_failed", error=error)
            return invalid_output

    def _auto_correct_literal_errors(
        self, raw_output: str | None
    ) -> tuple[str | None, T | None]:
        """Sanitize known literal errors such as enrichment aliases."""
        if not raw_output:
            return raw_output, None

        sanitized = self._sanitize_enrichment_literals(raw_output)
        if sanitized == raw_output:
            return raw_output, None

        validated = self._revalidate_output(sanitized)
        return sanitized, validated

    def _sanitize_enrichment_literals(self, raw_output: str) -> str:
        """Normalize enrichment_type values in serialized output."""
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output

        changed = False

        def _walk(node: Any) -> None:
            nonlocal changed
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "enrichment_type" and isinstance(value, list):
                        normalized: list[str] = []
                        for item in value:
                            normalized_value = normalize_enrichment_label(
                                str(item), apply_aliases=True
                            )
                            if normalized_value != item:
                                changed = True
                            normalized.append(normalized_value)
                        node[key] = normalized
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)

        if not changed:
            return raw_output

        return json.dumps(payload, ensure_ascii=False)

    def _revalidate_output(self, serialized: str) -> T | None:
        """Attempt to validate sanitized output without another LLM call."""
        try:
            parsed = json.loads(serialized)
        except json.JSONDecodeError:
            return None

        output_type = getattr(self.agent, "output_type", None)
        if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
            return None

        try:
            return output_type.model_validate(parsed)  # type: ignore[return-value]
        except Exception:
            return None

    def get_repair_metrics(self) -> dict[str, Any]:
        """Get repair success metrics.

        Returns:
            Dictionary with repair statistics
        """
        total = self.repair_attempts
        success_rate = self.repair_successes / total if total > 0 else 0.0

        return {
            "repair_attempts": self.repair_attempts,
            "repair_successes": self.repair_successes,
            "repair_failures": self.repair_failures,
            "success_rate": success_rate,
        }


def wrap_agent_with_fixing(
    agent: Agent[T],
    fix_model: Model | None = None,
    max_fix_attempts: int = 2,
    **kwargs: Any,
) -> OutputFixingParser[T]:
    """Convenience function to wrap an agent with OutputFixingParser.

    Args:
        agent: PydanticAI agent to wrap
        fix_model: Model to use for fixing
        max_fix_attempts: Maximum fix attempts
        **kwargs: Additional arguments for OutputFixingParser

    Returns:
        OutputFixingParser instance

    Note:
        Configure output_retries in the Agent constructor, not here.
    """
    return OutputFixingParser(
        agent=agent,
        fix_model=fix_model,
        max_fix_attempts=max_fix_attempts,
        **kwargs,
    )
