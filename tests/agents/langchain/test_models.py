"""Tests for LangChain agent system Pydantic models."""

import pytest
from pydantic import ValidationError

from obsidian_anki_sync.agents.langchain.models import (
    BilingualMode,
    CardDecision,
    CardDiffResult,
    CardFieldChange,
    CardType,
    ChangeSeverity,
    Difficulty,
    ExistingAnkiNote,
    IssueSeverity,
    IssueType,
    Language,
    NoteContext,
    NoteContextFrontmatter,
    NoteContextOrigin,
    NoteContextSections,
    ProposedCard,
    QAIssue,
    QAReport,
    RiskLevel,
    SchemaValidationError,
    SchemaValidationResult,
    SyncAction,
    ValidationErrorCode,
)


class TestNoteContext:
    """Tests for NoteContext model."""

    def test_create_minimal_note_context(self):
        """Test creating a minimal NoteContext."""
        frontmatter = NoteContextFrontmatter(
            title="Test Note",
            lang=Language.EN,
            topic="Testing",
            difficulty=Difficulty.EASY,
        )

        sections = NoteContextSections(
            question="What is testing?",
            answer="Testing is verification",
        )

        note_context = NoteContext(
            slug="test-note",
            note_path="/vault/test.md",
            vault_root="/vault",
            frontmatter=frontmatter,
            sections=sections,
        )

        assert note_context.slug == "test-note"
        assert note_context.frontmatter.title == "Test Note"
        assert note_context.sections.question == "What is testing?"

    def test_note_context_with_existing_anki_note(self):
        """Test NoteContext with existing Anki note."""
        frontmatter = NoteContextFrontmatter(
            title="Test",
            lang=Language.EN,
            topic="Test",
            difficulty=Difficulty.MEDIUM,
        )

        sections = NoteContextSections(
            question="Q",
            answer="A",
        )

        existing = ExistingAnkiNote(
            note_id=12345,
            model_name="APF::Simple",
            deck_name="Test Deck",
            fields={"Front": "Q", "Back": "A"},
            tags=["test"],
            last_sync_ts="2025-01-01T00:00:00Z",
            slug="test-note",
        )

        note_context = NoteContext(
            slug="test-note",
            note_path="/test.md",
            vault_root="/vault",
            frontmatter=frontmatter,
            sections=sections,
            existing_anki_note=existing,
        )

        assert note_context.existing_anki_note is not None
        assert note_context.existing_anki_note.note_id == 12345


class TestProposedCard:
    """Tests for ProposedCard model."""

    def test_create_basic_proposed_card(self):
        """Test creating a basic ProposedCard."""
        card = ProposedCard(
            card_type=CardType.BASIC,
            model_name="APF::Simple",
            deck_name="Test Deck",
            fields={"Front": "Question", "Back": "Answer"},
            tags=["test"],
            language=Language.EN,
            bilingual_mode=BilingualMode.NONE,
            slug="test-card",
            origin=NoteContextOrigin(
                note_path="/test.md",
                source_note_lang=Language.EN,
            ),
        )

        assert card.card_type == CardType.BASIC
        assert card.fields["Front"] == "Question"

    def test_proposed_card_requires_front_field(self):
        """Test that ProposedCard requires Front field."""
        with pytest.raises(ValidationError):
            ProposedCard(
                card_type=CardType.BASIC,
                model_name="APF::Simple",
                deck_name="Test",
                fields={"Back": "Answer only"},  # Missing Front
                tags=[],
                language=Language.EN,
                bilingual_mode=BilingualMode.NONE,
                slug="test",
                origin=NoteContextOrigin(
                    note_path="/test.md", source_note_lang=Language.EN
                ),
            )


class TestQAReport:
    """Tests for QAReport model."""

    def test_create_qa_report(self):
        """Test creating a QA report."""
        report = QAReport(
            qa_score=0.85,
            issues=[
                QAIssue(
                    type=IssueType.STYLE_ISSUE,
                    severity=IssueSeverity.LOW,
                    message="Minor style issue",
                )
            ],
        )

        assert report.qa_score == 0.85
        assert len(report.issues) == 1

    def test_qa_report_high_severity_check(self):
        """Test high severity issue detection."""
        report = QAReport(
            qa_score=0.6,
            issues=[
                QAIssue(
                    type=IssueType.ANSWER_MISMATCH,
                    severity=IssueSeverity.HIGH,
                    message="Answer doesn't match question",
                )
            ],
        )

        assert report.has_high_severity_issues is True

    def test_qa_report_no_high_severity(self):
        """Test when no high severity issues."""
        report = QAReport(
            qa_score=0.9,
            issues=[
                QAIssue(
                    type=IssueType.STYLE_ISSUE,
                    severity=IssueSeverity.LOW,
                    message="Minor issue",
                )
            ],
        )

        assert report.has_high_severity_issues is False


class TestSchemaValidationResult:
    """Tests for SchemaValidationResult model."""

    def test_create_valid_schema_result(self):
        """Test creating a valid schema result."""
        result = SchemaValidationResult(valid=True)

        assert result.valid is True
        assert result.has_errors is False

    def test_create_invalid_schema_result(self):
        """Test creating an invalid schema result."""
        result = SchemaValidationResult(
            valid=False,
            errors=[
                SchemaValidationError(
                    code=ValidationErrorCode.MISSING_REQUIRED_FIELD,
                    field="Front",
                    message="Front field is missing",
                )
            ],
        )

        assert result.valid is False
        assert result.has_errors is True


class TestCardDiffResult:
    """Tests for CardDiffResult model."""

    def test_create_card_diff_result(self):
        """Test creating a card diff result."""
        diff = CardDiffResult(
            changes=[
                CardFieldChange(
                    field="Back",
                    old_value="Old answer",
                    new_value="New answer",
                    severity=ChangeSeverity.CONTENT,
                    message="Content changed",
                )
            ],
            should_update=True,
            reason="Update approved",
            risk_level=RiskLevel.MEDIUM,
        )

        assert diff.should_update is True
        assert diff.has_content_changes is True

    def test_diff_structural_changes(self):
        """Test structural change detection."""
        diff = CardDiffResult(
            changes=[
                CardFieldChange(
                    field="model_name",
                    old_value="APF::Simple",
                    new_value="APF::Draw",
                    severity=ChangeSeverity.STRUCTURAL,
                    message="Model changed",
                )
            ],
            should_update=False,
            reason="Structural changes not allowed",
            risk_level=RiskLevel.HIGH,
        )

        assert diff.has_structural_changes is True
        assert diff.risk_level == RiskLevel.HIGH


class TestCardDecision:
    """Tests for CardDecision model."""

    def test_create_card_decision(self):
        """Test creating a card decision."""
        card = ProposedCard(
            card_type=CardType.BASIC,
            model_name="APF::Simple",
            deck_name="Test",
            fields={"Front": "Q", "Back": "A"},
            tags=[],
            language=Language.EN,
            bilingual_mode=BilingualMode.NONE,
            slug="test",
            origin=NoteContextOrigin(
                note_path="/test.md", source_note_lang=Language.EN
            ),
        )

        decision = CardDecision(
            action=SyncAction.CREATE,
            proposed_card=card,
            qa_report=QAReport(qa_score=0.9),
            schema_validation=SchemaValidationResult(valid=True),
            messages=["Card approved"],
            slug="test",
        )

        assert decision.action == SyncAction.CREATE
        assert decision.is_approved is True
        assert decision.needs_manual_review is False

    def test_card_decision_manual_review(self):
        """Test manual review decision."""
        card = ProposedCard(
            card_type=CardType.BASIC,
            model_name="APF::Simple",
            deck_name="Test",
            fields={"Front": "Q", "Back": "A"},
            tags=[],
            language=Language.EN,
            bilingual_mode=BilingualMode.NONE,
            slug="test",
            origin=NoteContextOrigin(
                note_path="/test.md", source_note_lang=Language.EN
            ),
        )

        decision = CardDecision(
            action=SyncAction.MANUAL_REVIEW,
            proposed_card=card,
            qa_report=QAReport(qa_score=0.5),
            schema_validation=SchemaValidationResult(valid=True),
            messages=["QA score too low"],
            slug="test",
        )

        assert decision.is_approved is False
        assert decision.needs_manual_review is True
