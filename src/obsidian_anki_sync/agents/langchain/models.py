"""Pydantic models for LangChain agent system data structures."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, cast

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enums
# ============================================================================


class CardType(str, Enum):
    """Anki card types."""

    BASIC = "Basic"
    CLOZE = "Cloze"
    CUSTOM = "Custom"


class Language(str, Enum):
    """Supported languages."""

    EN = "en"
    RU = "ru"


class BilingualMode(str, Enum):
    """Bilingual card generation modes."""

    NONE = "none"
    FRONT_BACK = "front_back"  # RU on Front, EN on Back (or vice versa)
    SEPARATE_CARDS = "separate_cards"  # Generate two separate cards


class IssueType(str, Enum):
    """Types of QA issues."""

    ANSWER_MISMATCH = "answer_mismatch"
    FRONT_LEAKS_ANSWER = "front_leaks_answer"
    LANGUAGE_MISMATCH = "language_mismatch"
    STYLE_ISSUE = "style_issue"
    MISSING_CRITICAL_DETAIL = "missing_critical_detail"


class IssueSeverity(str, Enum):
    """Severity levels for QA issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ValidationErrorCode(str, Enum):
    """Schema validation error codes."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_CLOZE_FORMAT = "invalid_cloze_format"
    LENGTH_EXCEEDED = "length_exceeded"


class ValidationWarningCode(str, Enum):
    """Schema validation warning codes."""

    BACK_TOO_LONG = "back_too_long"
    TOO_MANY_TAGS = "too_many_tags"
    STYLE_WARNING = "style_warning"


class ChangeSeverity(str, Enum):
    """Severity of card changes."""

    COSMETIC = "cosmetic"  # whitespace, punctuation
    CONTENT = "content"  # added/removed statements, changed meaning
    STRUCTURAL = "structural"  # model or deck changes


class RiskLevel(str, Enum):
    """Risk level for card updates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SyncAction(str, Enum):
    """Final sync action for a card."""

    CREATE = "create"
    UPDATE = "update"
    SKIP = "skip"
    MANUAL_REVIEW = "manual_review"


class Difficulty(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ============================================================================
# NoteContext and sub-structures
# ============================================================================


class Subquestion(BaseModel):
    """A subquestion within a note."""

    id: str
    question: str
    answer: str

    class Config:
        frozen = True


class NoteContextSections(BaseModel):
    """Content fragments from note body."""

    question: str = Field(..., description="Primary question text")
    answer: str = Field(..., description="Primary answer text")
    extra: Optional[str] = Field(None, description="Optional extra/explanation")
    examples: Optional[str] = Field(None, description="Optional code examples")
    subquestions: list[Subquestion] = Field(
        default_factory=list, description="Optional array of subquestions"
    )
    raw_markdown: Optional[str] = Field(
        None, description="Original markdown (optional but useful)"
    )

    class Config:
        frozen = True


class ExistingAnkiNote(BaseModel):
    """Existing Anki note data (if note already synced)."""

    note_id: int
    model_name: str
    deck_name: str
    fields: dict[str, str] = Field(..., description="Field name † field value mapping")
    tags: list[str]
    last_sync_ts: str = Field(..., description="ISO8601 timestamp")
    slug: str = Field(..., description="For consistency")

    class Config:
        frozen = True


class NoteContextFrontmatter(BaseModel):
    """YAML front-matter parsed into dict."""

    title: str
    lang: Language
    topic: str
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    card_type_hint: Optional[CardType] = None
    deck_hint: Optional[str] = None
    bilingual: Optional[bool] = None

    # Additional metadata fields
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Other metadata fields as needed"
    )

    class Config:
        frozen = True
        use_enum_values = True


class NoteContext(BaseModel):
    """Canonical representation of a parsed Obsidian note."""

    slug: str = Field(..., description="Stable identifier derived from note")
    note_path: str = Field(..., description="Relative/absolute path in vault")
    vault_root: str = Field(..., description="Vault root for reference")
    frontmatter: NoteContextFrontmatter
    sections: NoteContextSections
    existing_anki_note: Optional[ExistingAnkiNote] = Field(
        None, description="May be null if not synced yet"
    )
    config_profile: Optional[str] = Field(
        None, description="Which mapping/validation profile to use"
    )

    class Config:
        frozen = True

    def model_dump_json(self, **kwargs: Any) -> str:
        """Override to ensure proper JSON serialization."""
        return cast(
            str, super().model_dump_json(by_alias=True, exclude_none=False, **kwargs)
        )


# ============================================================================
# ProposedCard
# ============================================================================


class NoteContextOrigin(BaseModel):
    """Origin information for a card."""

    note_path: str
    source_note_lang: Language

    class Config:
        frozen = True
        use_enum_values = True


class ProposedCard(BaseModel):
    """Direct output of Card Mapping Agent."""

    card_type: CardType
    model_name: str = Field(..., description="Must match an Anki model")
    deck_name: str = Field(..., description="Resolved deck name")
    fields: dict[str, str] = Field(
        ...,
        description="Field name † value mapping (Front, Back, Extra, Hint, etc.)",
    )
    tags: list[str]
    language: Language
    bilingual_mode: BilingualMode
    slug: str = Field(..., description="Copied from NoteContext.slug")
    origin: NoteContextOrigin
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Mapping confidence score"
    )
    notes: str = Field("", description="Explanation from the agent")

    class Config:
        frozen = True
        use_enum_values = True

    @field_validator("fields")
    @classmethod
    def validate_required_fields(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure at minimum Front field is present."""
        if "Front" not in v:
            raise ValueError("ProposedCard must have at least 'Front' field")
        return v


# ============================================================================
# QAReport
# ============================================================================


class QAIssue(BaseModel):
    """A single QA issue."""

    type: IssueType
    severity: IssueSeverity
    message: str
    suggested_change: Optional[str] = None

    class Config:
        frozen = True
        use_enum_values = True


class QAAutoFix(BaseModel):
    """A single auto-fix applied to a card."""

    field: Literal["Front", "Back", "Extra", "Hint"]
    before: str
    after: str
    reason: str

    class Config:
        frozen = True


class QAReport(BaseModel):
    """Semantic QA and other checks."""

    qa_score: float = Field(..., ge=0.0, le=1.0, description="QA score from 0 to 1")
    issues: list[QAIssue] = Field(default_factory=list)
    auto_fixed: list[QAAutoFix] = Field(default_factory=list)

    class Config:
        frozen = True

    @property
    def has_high_severity_issues(self) -> bool:
        """Check if any high severity issues exist."""
        return any(issue.severity == IssueSeverity.HIGH for issue in self.issues)

    @property
    def has_medium_or_high_issues(self) -> bool:
        """Check if any medium or high severity issues exist."""
        return any(
            issue.severity in (IssueSeverity.MEDIUM, IssueSeverity.HIGH)
            for issue in self.issues
        )


# ============================================================================
# SchemaValidationResult
# ============================================================================


class SchemaValidationError(BaseModel):
    """A schema validation error."""

    code: ValidationErrorCode
    field: Optional[Literal["Front", "Back", "Extra", "Hint"]] = None
    message: str

    class Config:
        frozen = True
        use_enum_values = True


class SchemaValidationWarning(BaseModel):
    """A schema validation warning."""

    code: ValidationWarningCode
    field: Optional[Literal["Front", "Back", "Extra", "Hint"]] = None
    message: str

    class Config:
        frozen = True
        use_enum_values = True


class SchemaValidationResult(BaseModel):
    """Result of Anki model schema validation."""

    valid: bool
    errors: list[SchemaValidationError] = Field(default_factory=list)
    warnings: list[SchemaValidationWarning] = Field(default_factory=list)

    class Config:
        frozen = True

    @property
    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return len(self.errors) > 0


# ============================================================================
# CardDiffResult
# ============================================================================


class CardFieldChange(BaseModel):
    """A single field change in a card."""

    field: str = Field(..., description="Field name (e.g., Front, Back, tags)")
    old_value: Any
    new_value: Any
    severity: ChangeSeverity
    message: str

    class Config:
        frozen = True
        use_enum_values = True


class CardDiffResult(BaseModel):
    """Comparison result between existing and proposed card."""

    changes: list[CardFieldChange] = Field(default_factory=list)
    should_update: bool
    reason: str
    risk_level: RiskLevel

    class Config:
        frozen = True
        use_enum_values = True

    @property
    def has_structural_changes(self) -> bool:
        """Check if any structural changes exist."""
        return any(
            change.severity == ChangeSeverity.STRUCTURAL for change in self.changes
        )

    @property
    def has_content_changes(self) -> bool:
        """Check if any content changes exist."""
        return any(change.severity == ChangeSeverity.CONTENT for change in self.changes)


# ============================================================================
# CardDecision
# ============================================================================


class NoteSections(BaseModel):
    """Note sections for display in CardDecision."""

    question: str
    answer: str
    extra: Optional[str] = None

    class Config:
        frozen = True


class CardDecision(BaseModel):
    """Final decision for sync layer."""

    action: SyncAction
    proposed_card: ProposedCard
    qa_report: QAReport
    schema_validation: SchemaValidationResult
    diff: Optional[CardDiffResult] = None
    messages: list[str] = Field(
        default_factory=list, description="Human-readable summary"
    )
    slug: str
    note_sections: Optional[NoteSections] = Field(
        None, description="Original note sections for reference"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True
        use_enum_values = True

    @property
    def is_approved(self) -> bool:
        """Check if the card is approved for syncing."""
        return self.action in (SyncAction.CREATE, SyncAction.UPDATE)

    @property
    def needs_manual_review(self) -> bool:
        """Check if the card needs manual review."""
        return self.action == SyncAction.MANUAL_REVIEW

    def summary(self) -> str:
        """Generate a human-readable summary of the decision."""
        lines = [
            f"Action: {self.action.value}",
            f"Slug: {self.slug}",
            f"QA Score: {self.qa_report.qa_score:.2f}",
            f"Schema Valid: {self.schema_validation.valid}",
        ]

        if self.diff:
            lines.append(
                f"Changes: {len(self.diff.changes)} (Risk: {self.diff.risk_level.value})"
            )

        if self.messages:
            lines.append("Messages:")
            for msg in self.messages:
                lines.append(f"  - {msg}")

        return "\n".join(lines)
