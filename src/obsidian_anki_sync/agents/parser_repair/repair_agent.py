"""Parser-Repair agent implementation.

This agent activates only when rule-based parsing fails:
- Diagnoses parsing errors
- Suggests fixes for common issues
- Repairs content in-memory (doesn't modify source files)
- Re-attempts parsing with repairs
"""

import contextlib
import json
from pathlib import Path

from obsidian_anki_sync.agents.json_schemas import get_parser_repair_schema
from obsidian_anki_sync.agents.langgraph.retry_policies import (
    classify_error_category,
    select_repair_strategy,
)
from obsidian_anki_sync.agents.models import (
    NoteCorrectionResult,
    RepairDiagnosis,
    RepairQualityScore,
)
from obsidian_anki_sync.agents.repair_learning import get_repair_learning_system
from obsidian_anki_sync.agents.repair_metrics import get_repair_metrics_collector
from obsidian_anki_sync.exceptions import ParserError
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.logging import get_logger

from .prompt_builder import (
    PROACTIVE_ANALYSIS_SYSTEM_PROMPT,
    PROACTIVE_CORRECTION_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    build_proactive_analysis_prompt,
    build_proactive_correction_prompt,
    build_repair_prompt,
)
from .validators import validate_repaired_content

logger = get_logger(__name__)


class ParserRepairAgent:
    """Agent for repairing malformed notes that fail parsing.

    Uses lightweight model (qwen3:8b) for fast analysis and repair.
    Only invoked when rule-based parser fails.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:8b",
        temperature: float = 0.0,
        enable_content_generation: bool = True,
        repair_missing_sections: bool = True,
    ):
        """Initialize parser-repair agent.

        Args:
            ollama_client: LLM provider instance
            model: Model to use for repair
            temperature: Sampling temperature (0.0 for deterministic)
            enable_content_generation: Allow LLM to generate missing content
            repair_missing_sections: Generate missing language sections
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.enable_content_generation = enable_content_generation
        self.repair_missing_sections = repair_missing_sections
        logger.info(
            "parser_repair_agent_initialized",
            model=model,
            enable_content_generation=enable_content_generation,
            repair_missing_sections=repair_missing_sections,
        )

    def _build_repair_prompt(
        self, content: str, error: str, enable_content_gen: bool = True
    ) -> str:
        """Build repair prompt using shared builder."""
        return build_repair_prompt(
            content=content,
            error=error,
            enable_content_generation=enable_content_gen
            and self.enable_content_generation,
            repair_missing_sections=self.repair_missing_sections,
        )

    def repair_and_parse(
        self, file_path: Path, original_error: Exception
    ) -> tuple[NoteMetadata, list[QAPair]] | None:
        """Attempt to repair a note that failed parsing.

        Args:
            file_path: Path to the note file
            original_error: Original parsing error

        Returns:
            Tuple of (metadata, qa_pairs) if successful, None if unrepairable

        Raises:
            ParserError: If repair also fails
        """
        import time

        start_time = time.time()
        metrics_collector = get_repair_metrics_collector()

        # Classify error for metrics
        error_category = classify_error_category(original_error)

        # Try to get suggested strategy from learning system
        learning_system = get_repair_learning_system()
        suggested_strategy = learning_system.suggest_strategy(
            error_category=error_category,
            error_type=type(original_error).__name__,
            error_message=str(original_error),
        )

        # Use suggested strategy if available, otherwise use default selection
        if suggested_strategy:
            from .models import RepairStrategy

            repair_strategy = RepairStrategy(
                strategy_type=suggested_strategy,
                priority=1,  # Learned strategies get high priority
                stages=[suggested_strategy],
                confidence_threshold=0.7,
            )
            logger.info(
                "repair_strategy_from_learning",
                category=error_category,
                strategy=suggested_strategy,
            )
        else:
            repair_strategy = select_repair_strategy(original_error)

        strategy_type = repair_strategy.strategy_type

        logger.info(
            "parser_repair_attempt",
            file=str(file_path),
            error=str(original_error),
            category=error_category,
            strategy=strategy_type,
        )

        # Read original content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("parser_repair_read_failed", file=str(file_path), error=str(e))
            msg = f"Cannot read file for repair: {e}"
            raise ParserError(msg) from e

        # Build repair prompt
        prompt = self._build_repair_prompt(
            content,
            str(original_error),
            enable_content_gen=self.enable_content_generation,
        )

        # System prompt for note repair agent
        system_prompt = REPAIR_SYSTEM_PROMPT

        # Call LLM for repair analysis
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get JSON schema for structured output
                json_schema = get_parser_repair_schema()

                repair_result = self.ollama_client.generate_json(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    temperature=self.temperature,
                    json_schema=json_schema,
                )
                break  # Success, exit retry loop

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "parser_repair_json_retry",
                        file=str(file_path),
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    continue
                logger.error(
                    "parser_repair_invalid_json",
                    file=str(file_path),
                    error=str(e),
                )
                return None
            except Exception as e:
                logger.error(
                    "parser_repair_llm_failed", file=str(file_path), error=str(e)
                )
                return None

        # Check if repairable
        if not repair_result.get("is_repairable", False):
            repair_time = time.time() - start_time
            logger.warning(
                "parser_repair_unrepairable",
                file=str(file_path),
                diagnosis=repair_result.get("diagnosis", "Unknown"),
            )
            # Record failed attempt
            metrics_collector.record_attempt(
                error_category=error_category,
                error_type=type(original_error).__name__,
                strategy_used=strategy_type,
                success=False,
                repair_time=repair_time,
                error_message=str(original_error),
            )
            return None

        # Get repaired content
        repaired_content = repair_result.get("repaired_content")
        if not repaired_content:
            logger.warning(
                "parser_repair_no_content",
                file=str(file_path),
            )
            return None

        # Log repairs applied
        repairs = repair_result.get("repairs", [])
        content_generation_applied = repair_result.get(
            "content_generation_applied", False
        )
        generated_sections = repair_result.get("generated_sections", [])

        # Parse error diagnosis if present
        error_diagnosis = None
        error_diag_data = repair_result.get("error_diagnosis")
        if error_diag_data:
            try:
                error_diagnosis = RepairDiagnosis(**error_diag_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_diagnosis",
                    file=str(file_path),
                    error=str(e),
                )

        # Parse quality scores if present
        quality_before = None
        quality_after = None
        quality_before_data = repair_result.get("quality_before")
        quality_after_data = repair_result.get("quality_after")

        if quality_before_data:
            try:
                quality_before = RepairQualityScore(**quality_before_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_quality_before",
                    file=str(file_path),
                    error=str(e),
                )

        if quality_after_data:
            try:
                quality_after = RepairQualityScore(**quality_after_data)
            except Exception as e:
                logger.warning(
                    "parser_repair_invalid_quality_after",
                    file=str(file_path),
                    error=str(e),
                )

        logger.info(
            "parser_repair_applied",
            file=str(file_path),
            diagnosis=repair_result.get("diagnosis", "N/A"),
            repairs_count=len(repairs),
            content_generation_applied=content_generation_applied,
            generated_sections_count=len(generated_sections),
            error_category=error_diagnosis.error_category if error_diagnosis else None,
            severity=error_diagnosis.severity if error_diagnosis else None,
            quality_before=quality_before.overall_score if quality_before else None,
            quality_after=quality_after.overall_score if quality_after else None,
        )

        for repair in repairs:
            logger.debug(
                "parser_repair_detail",
                repair_type=repair.get("type", "unknown"),
                description=repair.get("description", ""),
            )

        for gen_section in generated_sections:
            logger.info(
                "parser_repair_content_generated",
                file=str(file_path),
                section_type=gen_section.get("section_type", "unknown"),
                method=gen_section.get("method", "unknown"),
                description=gen_section.get("description", ""),
            )

        # Validate repaired content against APF/Obsidian requirements
        validation_errors = validate_repaired_content(repaired_content, file_path)
        if validation_errors:
            logger.warning(
                "parser_repair_validation_warnings",
                file=str(file_path),
                errors=validation_errors,
            )

        # Try parsing repaired content
        # Import here to avoid circular dependency
        from obsidian_anki_sync.obsidian.parser import parse_frontmatter, parse_qa_pairs

        try:
            # Write to temporary path for parsing
            temp_content_for_parse = repaired_content

            # Parse frontmatter from repaired content
            metadata = parse_frontmatter(temp_content_for_parse, file_path)

            # Parse Q/A pairs from repaired content
            qa_pairs = parse_qa_pairs(temp_content_for_parse, metadata, file_path)

            if not qa_pairs:
                logger.warning(
                    "parser_repair_no_qa_pairs",
                    file=str(file_path),
                )
                return None

            repair_time = time.time() - start_time
            quality_improvement = (
                quality_after.overall_score - quality_before.overall_score
                if quality_before and quality_after
                else None
            )

            logger.info(
                "parser_repair_success",
                file=str(file_path),
                qa_pairs_count=len(qa_pairs),
                quality_improvement=quality_improvement,
            )

            # Record successful repair
            metrics_collector.record_attempt(
                error_category=error_category,
                error_type=type(original_error).__name__,
                strategy_used=strategy_type,
                success=True,
                quality_before=quality_before.overall_score if quality_before else None,
                quality_after=quality_after.overall_score if quality_after else None,
                repair_time=repair_time,
            )

            # Learn from successful repair
            quality_improvement = (
                quality_after.overall_score - quality_before.overall_score
                if quality_before and quality_after
                else None
            )
            learning_system.learn_from_success(
                error_category=error_category,
                error_type=type(original_error).__name__,
                error_message=str(original_error),
                strategy_used=strategy_type,
                quality_improvement=quality_improvement,
                repair_steps=[repair.get("type", "") for repair in repairs],
            )

            return metadata, qa_pairs

        except ParserError as e:
            repair_time = time.time() - start_time
            logger.error(
                "parser_repair_reparse_failed",
                file=str(file_path),
                error=str(e),
            )
            # Record failed attempt
            metrics_collector.record_attempt(
                error_category=error_category,
                error_type=type(original_error).__name__,
                strategy_used=strategy_type,
                success=False,
                repair_time=repair_time,
                error_message=str(e),
            )
            return None
        return None

    async def analyze_and_correct_proactively_async(
        self, content: str, file_path: Path | None = None
    ) -> NoteCorrectionResult:
        """Proactively analyze note quality and apply corrections before parsing asynchronously.

        This method analyzes notes for common issues before they cause parsing failures,
        enabling early correction and better quality.

        Args:
            content: Note content to analyze
            file_path: Optional file path for context

        Returns:
            NoteCorrectionResult with analysis and corrections
        """
        import time

        start_time = time.time()
        logger.info(
            "proactive_note_analysis_async_start",
            file=str(file_path) if file_path else "unknown",
        )

        # Build proactive analysis prompt
        analysis_prompt = build_proactive_analysis_prompt(content)

        system_prompt = PROACTIVE_ANALYSIS_SYSTEM_PROMPT

        try:
            # Call LLM for analysis
            analysis_result = await self.ollama_client.generate_json_async(
                model=self.model,
                prompt=analysis_prompt,
                system=system_prompt,
                temperature=self.temperature,
            )

            needs_correction = analysis_result.get("needs_correction", False)
            quality_score = analysis_result.get("quality_score", 0.0)
            issues_found = analysis_result.get("issues_found", [])
            confidence = analysis_result.get("confidence", 0.0)

            # If correction is needed, apply it
            corrected_content = None
            corrections_applied = []
            quality_after = None

            if needs_correction and self.enable_content_generation:
                logger.info(
                    "proactive_correction_needed_async",
                    file=str(file_path),
                    issues=issues_found,
                )

                # Build correction prompt
                correction_prompt = build_proactive_correction_prompt(
                    content=content,
                    issues_found=issues_found,
                    suggested_corrections=analysis_result.get(
                        "suggested_corrections", []
                    ),
                )

                # Call LLM for correction
                correction_result_data = await self.ollama_client.generate_json_async(
                    model=self.model,
                    prompt=correction_prompt,
                    system=system_prompt,
                    temperature=self.temperature,
                )

                corrected_content = correction_result_data.get("corrected_content")
                corrections_applied = correction_result_data.get(
                    "corrections_applied", []
                )
                quality_after_data = correction_result_data.get("quality_after")

                if quality_after_data:
                    with contextlib.suppress(Exception):
                        quality_after = RepairQualityScore(**quality_after_data)

            correction_time = time.time() - start_time

            return NoteCorrectionResult(
                needs_correction=needs_correction,
                quality_score=quality_score,
                issues_found=issues_found,
                corrections_applied=corrections_applied,
                corrected_content=corrected_content,
                quality_after=quality_after,
                confidence=confidence,
                correction_time=correction_time,
            )

        except Exception as e:
            logger.error(
                "proactive_analysis_async_failed",
                file=str(file_path),
                error=str(e),
            )
            return NoteCorrectionResult(
                needs_correction=False,
                quality_score=0.0,
                issues_found=[f"Analysis failed: {e!s}"],
                corrections_applied=[],
                confidence=0.0,
                correction_time=time.time() - start_time,
            )

    def analyze_and_correct_proactively(
        self, content: str, file_path: Path | None = None
    ) -> NoteCorrectionResult:
        """Proactively analyze note quality and apply corrections before parsing.

        This method analyzes notes for common issues before they cause parsing failures,
        enabling early correction and better quality.

        Args:
            content: Note content to analyze
            file_path: Optional file path for context

        Returns:
            NoteCorrectionResult with analysis and corrections
        """
        import time

        start_time = time.time()
        logger.info(
            "proactive_note_analysis_start",
            file=str(file_path) if file_path else "unknown",
        )

        # Build proactive analysis prompt
        analysis_prompt = build_proactive_analysis_prompt(content)

        system_prompt = PROACTIVE_ANALYSIS_SYSTEM_PROMPT

        try:
            # Get JSON schema for structured output
            json_schema = {
                "type": "object",
                "properties": {
                    "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "issues_found": {"type": "array", "items": {"type": "string"}},
                    "needs_correction": {"type": "boolean"},
                    "suggested_corrections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": [
                    "quality_score",
                    "issues_found",
                    "needs_correction",
                    "confidence",
                ],
            }

            analysis_result = self.ollama_client.generate_json(
                model=self.model,
                prompt=analysis_prompt,
                system=system_prompt,
                temperature=self.temperature,
                json_schema=json_schema,
            )

            quality_score = analysis_result.get("quality_score", 0.5)
            issues_found = analysis_result.get("issues_found", [])
            needs_correction = analysis_result.get("needs_correction", False)
            suggested_corrections = analysis_result.get("suggested_corrections", [])
            confidence = analysis_result.get("confidence", 0.5)

            corrected_content = None
            corrections_applied = []
            quality_after = None

            # If correction is needed and we have suggestions, attempt repair
            if needs_correction and suggested_corrections:
                logger.info(
                    "proactive_correction_needed",
                    issues_count=len(issues_found),
                    suggestions_count=len(suggested_corrections),
                )

                # Create a synthetic error for repair
                synthetic_error = f"Proactive correction needed. Issues: {', '.join(issues_found[:3])}"

                # Attempt repair using existing repair logic
                repair_prompt = self._build_repair_prompt(
                    content,
                    synthetic_error,
                    enable_content_gen=self.enable_content_generation,
                )

                repair_system_prompt = PROACTIVE_CORRECTION_SYSTEM_PROMPT

                try:
                    repair_json_schema = get_parser_repair_schema()
                    repair_result = self.ollama_client.generate_json(
                        model=self.model,
                        prompt=repair_prompt,
                        system=repair_system_prompt,
                        temperature=self.temperature,
                        json_schema=repair_json_schema,
                    )

                    if repair_result.get("is_repairable", False):
                        corrected_content = repair_result.get("repaired_content")
                        repairs = repair_result.get("repairs", [])
                        corrections_applied = [
                            repair.get("description", "") for repair in repairs
                        ]

                        # Extract quality after if available
                        quality_after_data = repair_result.get("quality_after")
                        if quality_after_data:
                            with contextlib.suppress(Exception):
                                quality_after = RepairQualityScore(**quality_after_data)

                        logger.info(
                            "proactive_correction_applied",
                            corrections_count=len(corrections_applied),
                        )
                    else:
                        logger.info("proactive_correction_not_repairable")
                except Exception as e:
                    logger.warning("proactive_repair_failed", error=str(e))

            correction_time = time.time() - start_time

            result = NoteCorrectionResult(
                needs_correction=needs_correction,
                corrected_content=corrected_content,
                quality_score=quality_score,
                issues_found=issues_found,
                corrections_applied=corrections_applied,
                confidence=confidence,
                correction_time=correction_time,
                quality_after=quality_after,
            )

            logger.info(
                "proactive_note_analysis_complete",
                needs_correction=needs_correction,
                quality_score=quality_score,
                issues_count=len(issues_found),
                corrections_applied=len(corrections_applied),
                time=correction_time,
            )

            return result

        except Exception as e:
            logger.error("proactive_analysis_failed", error=str(e))
            # Return permissive result on error
            return NoteCorrectionResult(
                needs_correction=False,
                quality_score=0.5,
                issues_found=[f"Analysis failed: {e!s}"],
                corrections_applied=[],
                confidence=0.0,
                correction_time=time.time() - start_time,
            )


def attempt_repair(
    file_path: Path,
    original_error: Exception,
    ollama_client: BaseLLMProvider,
    model: str = "qwen3:8b",
    enable_content_generation: bool = True,
    repair_missing_sections: bool = True,
) -> tuple[NoteMetadata, list[QAPair]] | None:
    """Helper function to attempt repair of a failed parse.

    Args:
        file_path: Path to the note file
        original_error: Original parsing error
        ollama_client: LLM provider instance
        model: Model to use for repair
        enable_content_generation: Allow LLM to generate missing content
        repair_missing_sections: Generate missing language sections

    Returns:
        Tuple of (metadata, qa_pairs) if successful, None if unrepairable
    """
    agent = ParserRepairAgent(
        ollama_client,
        model,
        enable_content_generation=enable_content_generation,
        repair_missing_sections=repair_missing_sections,
    )
    return agent.repair_and_parse(file_path, original_error)
