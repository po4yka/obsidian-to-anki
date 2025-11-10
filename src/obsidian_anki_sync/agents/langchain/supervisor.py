"""Supervisor Orchestrator - Coordinates the multi-agent card generation pipeline.

The Supervisor manages the flow through:
1. Card Mapping Agent
2. Schema Validation Tool
3. QA Agent
4. Style/Hint Agent (optional)
5. Card Diff Agent (for updates)

With retry/repair logic and error handling.
"""

from typing import Any, Optional

from obsidian_anki_sync.agents.langchain.langchain_provider import (
    create_langchain_chat_model,
)
from obsidian_anki_sync.agents.langchain.models import (
    CardDecision,
    NoteContext,
    NoteSections,
    SyncAction,
)
from obsidian_anki_sync.agents.langchain.tools.card_differ import CardDifferTool
from obsidian_anki_sync.agents.langchain.tools.card_mapper import CardMapperTool
from obsidian_anki_sync.agents.langchain.tools.qa_checker import QACheckerTool
from obsidian_anki_sync.agents.langchain.tools.schema_validator import (
    SchemaValidatorTool,
)
from obsidian_anki_sync.agents.langchain.tools.style_polisher import StylePolisherTool
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisorConfig:
    """Configuration for Supervisor Orchestrator."""

    def __init__(
        self,
        max_mapping_retries: int = 2,
        min_qa_score: float = 0.8,
        allow_auto_fix: bool = True,
        allow_content_updates: bool = True,
        allow_structural_updates: bool = False,
        max_llm_calls_per_card: int = 8,
        enable_style_polish: bool = False,
        strict_schema_validation: bool = False,
    ):
        """Initialize supervisor configuration.

        Args:
            max_mapping_retries: Maximum retries for card mapping
            min_qa_score: Minimum QA score to pass
            allow_auto_fix: Allow automatic fixing based on feedback
            allow_content_updates: Allow updates that change card content
            allow_structural_updates: Allow model/deck changes
            max_llm_calls_per_card: Maximum total LLM calls per card
            enable_style_polish: Enable style polishing step
            strict_schema_validation: Treat warnings as errors
        """
        self.max_mapping_retries = max_mapping_retries
        self.min_qa_score = min_qa_score
        self.allow_auto_fix = allow_auto_fix
        self.allow_content_updates = allow_content_updates
        self.allow_structural_updates = allow_structural_updates
        self.max_llm_calls_per_card = max_llm_calls_per_card
        self.enable_style_polish = enable_style_polish
        self.strict_schema_validation = strict_schema_validation


class LangChainSupervisor:
    """Supervisor that orchestrates the LangChain agent pipeline."""

    def __init__(
        self,
        config: Any,
        supervisor_config: Optional[SupervisorConfig] = None,
        anki_client: Optional[Any] = None,
    ):
        """Initialize the supervisor.

        Args:
            config: Application config object
            supervisor_config: Optional supervisor-specific config
            anki_client: Optional AnkiConnect client for dynamic schema fetching
        """
        self.config = config
        self.supervisor_config = supervisor_config or SupervisorConfig()
        self.anki_client = anki_client

        # Create LLMs
        self.mapping_llm = create_langchain_chat_model(
            config, model_name=getattr(config, "generator_model", "qwen3:32b")
        )
        self.qa_llm = create_langchain_chat_model(
            config, model_name=getattr(config, "post_validator_model", "qwen3:14b")
        )
        self.style_llm = create_langchain_chat_model(
            config, model_name=getattr(config, "pre_validator_model", "qwen3:8b")
        )

        # Initialize tools
        self.card_mapper = CardMapperTool(
            llm=self.mapping_llm,
            max_retries=self.supervisor_config.max_mapping_retries,
        )

        self.schema_validator = SchemaValidatorTool(
            anki_client=anki_client,
            strict_mode=self.supervisor_config.strict_schema_validation,
        )

        self.qa_checker = QACheckerTool(
            llm=self.qa_llm,
            min_acceptable_score=self.supervisor_config.min_qa_score,
        )

        self.style_polisher = StylePolisherTool(
            llm=self.style_llm,
            enabled=self.supervisor_config.enable_style_polish,
        )

        self.card_differ = CardDifferTool(
            allow_content_updates=self.supervisor_config.allow_content_updates,
            allow_structural_updates=self.supervisor_config.allow_structural_updates,
        )

        # Call counter
        self._llm_calls = 0

        logger.info(
            "supervisor_initialized",
            max_retries=self.supervisor_config.max_mapping_retries,
            min_qa_score=self.supervisor_config.min_qa_score,
            auto_fix=self.supervisor_config.allow_auto_fix,
        )

    def process_note(self, note_context: NoteContext) -> CardDecision:
        """Process a note through the complete agent pipeline.

        Args:
            note_context: The note to process

        Returns:
            CardDecision with final action and all reports
        """
        logger.info("supervisor_process_start", slug=note_context.slug)

        # Reset call counter
        self._llm_calls = 0

        try:
            # Step 1: Map to ProposedCard
            proposed_card = self._map_with_retries(note_context)

            # Step 2: Validate schema
            schema_result = self.schema_validator.validate(proposed_card)

            if not schema_result.valid:
                # If schema invalid after retries, manual review
                logger.warning(
                    "supervisor_schema_failed",
                    slug=note_context.slug,
                    errors=len(schema_result.errors),
                )

                return CardDecision(
                    action=SyncAction.MANUAL_REVIEW,
                    proposed_card=proposed_card,
                    qa_report=self.qa_checker.check(note_context, proposed_card),
                    schema_validation=schema_result,
                    diff=None,
                    messages=[
                        "Schema validation failed after retries.",
                        f"Errors: {', '.join(e.message for e in schema_result.errors)}",
                    ],
                    slug=note_context.slug,
                    note_sections=NoteSections(
                        question=note_context.sections.question,
                        answer=note_context.sections.answer,
                        extra=note_context.sections.extra,
                    ),
                )

            # Step 3: Optional style polish
            if self.supervisor_config.enable_style_polish:
                proposed_card = self.style_polisher.polish(proposed_card, note_context)
                self._llm_calls += 1

            # Step 4: QA check
            qa_report = self.qa_checker.check(note_context, proposed_card)
            self._llm_calls += 1

            if qa_report.qa_score < self.supervisor_config.min_qa_score:
                if self.supervisor_config.allow_auto_fix and self._can_retry():
                    # Retry mapping with QA feedback
                    logger.info(
                        "supervisor_qa_retry",
                        slug=note_context.slug,
                        score=qa_report.qa_score,
                    )

                    feedback = self._build_qa_feedback(qa_report)
                    proposed_card = self.card_mapper.map(
                        note_context, feedback=feedback
                    )
                    self._llm_calls += 1

                    # Re-validate
                    schema_result = self.schema_validator.validate(proposed_card)
                    qa_report = self.qa_checker.check(note_context, proposed_card)
                    self._llm_calls += 1

            # Step 5: Handle existing card diff
            diff_result = None
            if note_context.existing_anki_note:
                diff_result = self.card_differ.compare(
                    note_context.existing_anki_note, proposed_card
                )

                if not diff_result.should_update:
                    return CardDecision(
                        action=SyncAction.SKIP,
                        proposed_card=proposed_card,
                        qa_report=qa_report,
                        schema_validation=schema_result,
                        diff=diff_result,
                        messages=[f"Skip: {diff_result.reason}"],
                        slug=note_context.slug,
                        note_sections=NoteSections(
                            question=note_context.sections.question,
                            answer=note_context.sections.answer,
                            extra=note_context.sections.extra,
                        ),
                    )

            # Step 6: Determine final action
            action = self._determine_action(
                schema_result, qa_report, diff_result, note_context
            )

            messages = self._build_decision_messages(
                action, schema_result, qa_report, diff_result
            )

            logger.info(
                "supervisor_process_complete",
                slug=note_context.slug,
                action=action.value,
                qa_score=qa_report.qa_score,
                llm_calls=self._llm_calls,
            )

            return CardDecision(
                action=action,
                proposed_card=proposed_card,
                qa_report=qa_report,
                schema_validation=schema_result,
                diff=diff_result,
                messages=messages,
                slug=note_context.slug,
                note_sections=NoteSections(
                    question=note_context.sections.question,
                    answer=note_context.sections.answer,
                    extra=note_context.sections.extra,
                ),
            )

        except Exception as e:
            logger.error(
                "supervisor_process_error",
                slug=note_context.slug,
                error=str(e),
            )

            # Fallback: manual review
            return self._create_error_decision(note_context, str(e))

    def _map_with_retries(self, note_context: NoteContext) -> Any:
        """Map note with schema validation retries."""
        for attempt in range(self.supervisor_config.max_mapping_retries + 1):
            proposed_card = self.card_mapper.map(note_context)
            self._llm_calls += 1

            schema_result = self.schema_validator.validate(proposed_card)

            if schema_result.valid or not self._can_retry():
                return proposed_card

            # Build feedback from schema errors
            feedback = self._build_schema_feedback(schema_result)
            logger.info(
                "supervisor_schema_retry",
                slug=note_context.slug,
                attempt=attempt + 1,
            )

            proposed_card = self.card_mapper.map(note_context, feedback=feedback)
            self._llm_calls += 1

        return proposed_card

    def _can_retry(self) -> bool:
        """Check if we can make more LLM calls."""
        return self._llm_calls < self.supervisor_config.max_llm_calls_per_card

    def _build_schema_feedback(self, schema_result: Any) -> str:
        """Build feedback string from schema validation errors."""
        feedback_parts = ["Schema validation failed with the following errors:"]
        for error in schema_result.errors:
            feedback_parts.append(f"- {error.message}")
        return "\n".join(feedback_parts)

    def _build_qa_feedback(self, qa_report: Any) -> str:
        """Build feedback string from QA issues."""
        feedback_parts = [
            f"QA check scored {qa_report.qa_score:.2f} (threshold: {self.supervisor_config.min_qa_score})",
            "Issues found:",
        ]
        for issue in qa_report.issues:
            feedback_parts.append(f"- [{issue.severity.value}] {issue.message}")
            if issue.suggested_change:
                feedback_parts.append(f"  Suggestion: {issue.suggested_change}")
        return "\n".join(feedback_parts)

    def _determine_action(
        self,
        schema_result: Any,
        qa_report: Any,
        diff_result: Any,
        note_context: NoteContext,
    ) -> SyncAction:
        """Determine final sync action."""
        # Check for critical failures
        if not schema_result.valid:
            return SyncAction.MANUAL_REVIEW

        if qa_report.has_high_severity_issues:
            return SyncAction.MANUAL_REVIEW

        if qa_report.qa_score < self.supervisor_config.min_qa_score:
            return SyncAction.MANUAL_REVIEW

        # If existing card, return UPDATE
        if note_context.existing_anki_note:
            return SyncAction.UPDATE

        # Otherwise, CREATE
        return SyncAction.CREATE

    def _build_decision_messages(
        self, action: SyncAction, schema_result: Any, qa_report: Any, diff_result: Any
    ) -> list[str]:
        """Build human-readable decision messages."""
        messages = []

        if action == SyncAction.CREATE:
            messages.append(f"Card passed all checks (QA: {qa_report.qa_score:.2f})")
        elif action == SyncAction.UPDATE:
            messages.append(f"Update approved (QA: {qa_report.qa_score:.2f})")
            if diff_result:
                messages.append(
                    f"Changes: {len(diff_result.changes)}, Risk: {diff_result.risk_level.value}"
                )
        elif action == SyncAction.SKIP:
            messages.append("No update needed")
        elif action == SyncAction.MANUAL_REVIEW:
            messages.append("Requires manual review")

        # Add warnings
        if schema_result.warnings:
            messages.append(f"Schema warnings: {len(schema_result.warnings)}")

        if qa_report.issues:
            messages.append(f"QA issues: {len(qa_report.issues)}")

        return messages

    def _create_error_decision(
        self, note_context: NoteContext, error: str
    ) -> CardDecision:
        """Create a manual review decision for errors."""
        from obsidian_anki_sync.agents.langchain.models import (
            BilingualMode,
            CardType,
            NoteContextOrigin,
            ProposedCard,
            QAReport,
            SchemaValidationResult,
        )

        # Create minimal proposed card
        proposed_card = ProposedCard(
            card_type=CardType.BASIC,
            model_name="APF: Simple (3.0.0)",
            deck_name="Interview::Error",
            fields={
                "Front": note_context.sections.question,
                "Back": note_context.sections.answer,
            },
            tags=["error"],
            language=note_context.frontmatter.lang,
            bilingual_mode=BilingualMode.NONE,
            slug=note_context.slug,
            origin=NoteContextOrigin(
                note_path=note_context.note_path,
                source_note_lang=note_context.frontmatter.lang,
            ),
            confidence=0.0,
            notes=f"Error during processing: {error}",
        )

        return CardDecision(
            action=SyncAction.MANUAL_REVIEW,
            proposed_card=proposed_card,
            qa_report=QAReport(qa_score=0.0),
            schema_validation=SchemaValidationResult(valid=False),
            diff=None,
            messages=[f"Error: {error}"],
            slug=note_context.slug,
            note_sections=(
                note_context.sections if hasattr(note_context, "sections") else None
            ),
        )
