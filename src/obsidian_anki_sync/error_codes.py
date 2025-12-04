"""Structured error codes for machine-readable error handling.

This module provides a comprehensive set of error codes for categorizing
and tracking errors throughout the sync pipeline. Error codes follow the
format: {DOMAIN}-{CATEGORY}-{NUMBER}

Error Domains:
    GEN - Generation errors (card generation, LLM calls)
    VAL - Validation errors (pre/post validation, linting)
    ANK - Anki errors (connection, creation, orphans)
    PRV - Provider errors (RAG, memory, LLM providers)
    STA - State errors (database, consistency)
    CFG - Configuration errors

Usage:
    from obsidian_anki_sync.error_codes import ErrorCode

    logger.error(
        "generation_timeout",
        error_code=ErrorCode.GEN_TIMEOUT_PRIMARY.value,
        timeout_seconds=300,
    )
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Structured error codes for machine-readable handling.

    All error codes inherit from str for JSON serialization compatibility.
    Format: {DOMAIN}-{CATEGORY}-{NUMBER}
    """

    # =========================================================================
    # Generation Errors (GEN-xxx-xxx)
    # =========================================================================
    GEN_TIMEOUT_PRIMARY = "GEN-TIMEOUT-001"
    """Primary model generation timed out."""

    GEN_TIMEOUT_FALLBACK = "GEN-TIMEOUT-002"
    """Fallback model generation timed out."""

    GEN_MODEL_UNAVAILABLE = "GEN-MODEL-001"
    """Generation model is unavailable or failed to initialize."""

    GEN_EMPTY_RESULT = "GEN-EMPTY-001"
    """Generation returned empty result with no cards."""

    GEN_PARTIAL_RESULT = "GEN-PARTIAL-001"
    """Parallel generation had partial failures (some chunks failed)."""

    GEN_INPUT_INVALID = "GEN-INPUT-001"
    """Generation input validation failed (missing required fields)."""

    GEN_CHUNK_FAILED = "GEN-CHUNK-001"
    """Individual chunk failed during parallel generation."""

    # =========================================================================
    # Validation Errors (VAL-xxx-xxx)
    # =========================================================================
    VAL_PRE_FAILED = "VAL-PRE-001"
    """Pre-validation rejected the note structure."""

    VAL_PRE_TIMEOUT = "VAL-PRE-002"
    """Pre-validation timed out."""

    VAL_POST_FAILED = "VAL-POST-001"
    """Post-validation rejected generated cards."""

    VAL_POST_TIMEOUT = "VAL-POST-002"
    """Post-validation timed out."""

    VAL_LINTER_FAILED = "VAL-LINT-001"
    """APF linter validation failed."""

    VAL_AUTOFIX_FAILED = "VAL-FIX-001"
    """Auto-fix attempts exhausted without success."""

    # =========================================================================
    # Anki Errors (ANK-xxx-xxx)
    # =========================================================================
    ANK_CONNECTION_FAILED = "ANK-CONN-001"
    """Failed to connect to AnkiConnect."""

    ANK_CREATE_FAILED = "ANK-CREATE-001"
    """Failed to create note in Anki."""

    ANK_UPDATE_FAILED = "ANK-UPDATE-001"
    """Failed to update note in Anki."""

    ANK_DELETE_FAILED = "ANK-DELETE-001"
    """Failed to delete note from Anki."""

    ANK_ORPHAN_DETECTED = "ANK-ORPHAN-001"
    """Orphaned cards detected (in Anki but not in DB, or vice versa)."""

    ANK_VERIFICATION_FAILED = "ANK-VERIFY-001"
    """Card verification failed after creation."""

    ANK_ROLLBACK_FAILED = "ANK-ROLLBACK-001"
    """Failed to rollback Anki operation."""

    ANK_EMPTY_NOTE = "ANK-EMPTY-001"
    """Attempted to create empty note in Anki."""

    # =========================================================================
    # Provider Errors (PRV-xxx-xxx)
    # =========================================================================
    PRV_RAG_FAILED = "PRV-RAG-001"
    """RAG context enrichment failed."""

    PRV_MEMORY_FAILED = "PRV-MEM-001"
    """Memory store operation failed."""

    PRV_EMPTY_COMPLETION = "PRV-EMPTY-001"
    """LLM provider returned empty completion."""

    PRV_RATE_LIMITED = "PRV-RATE-001"
    """LLM provider rate limited the request."""

    PRV_QUOTA_EXCEEDED = "PRV-QUOTA-001"
    """LLM provider quota exceeded."""

    PRV_AUTH_FAILED = "PRV-AUTH-001"
    """LLM provider authentication failed."""

    PRV_FALLBACK_USED = "PRV-FALLBACK-001"
    """Primary model failed, fallback model used."""

    # =========================================================================
    # State Errors (STA-xxx-xxx)
    # =========================================================================
    STA_DB_WRITE_FAILED = "STA-DB-001"
    """Failed to write to state database."""

    STA_DB_READ_FAILED = "STA-DB-002"
    """Failed to read from state database."""

    STA_INCONSISTENT = "STA-INCON-001"
    """Inconsistent state detected between Anki and database."""

    STA_TRANSACTION_FAILED = "STA-TXN-001"
    """Transaction failed and was rolled back."""

    STA_CHECKPOINT_FAILED = "STA-CKPT-001"
    """Failed to save checkpoint for resumable sync."""

    # =========================================================================
    # Configuration Errors (CFG-xxx-xxx)
    # =========================================================================
    CFG_INVALID = "CFG-INVALID-001"
    """Configuration validation failed."""

    CFG_MISSING_KEY = "CFG-KEY-001"
    """Required configuration key is missing."""

    CFG_PATH_INVALID = "CFG-PATH-001"
    """Configuration path is invalid or inaccessible."""


def get_error_domain(code: ErrorCode) -> str:
    """Extract the domain from an error code.

    Args:
        code: The error code

    Returns:
        The domain prefix (e.g., "GEN", "VAL", "ANK")
    """
    return code.value.split("-")[0]


def is_retriable_error_code(code: ErrorCode) -> bool:
    """Check if an error code represents a retriable error.

    Args:
        code: The error code to check

    Returns:
        True if the error is typically retriable
    """
    retriable_codes = {
        ErrorCode.GEN_TIMEOUT_PRIMARY,
        ErrorCode.GEN_TIMEOUT_FALLBACK,
        ErrorCode.VAL_PRE_TIMEOUT,
        ErrorCode.VAL_POST_TIMEOUT,
        ErrorCode.ANK_CONNECTION_FAILED,
        ErrorCode.PRV_RATE_LIMITED,
        ErrorCode.PRV_EMPTY_COMPLETION,
        ErrorCode.STA_DB_WRITE_FAILED,
    }
    return code in retriable_codes


def get_error_severity(code: ErrorCode) -> str:
    """Get the severity level for an error code.

    Args:
        code: The error code

    Returns:
        Severity level: "critical", "error", "warning"
    """
    critical_codes = {
        ErrorCode.CFG_INVALID,
        ErrorCode.CFG_MISSING_KEY,
        ErrorCode.PRV_AUTH_FAILED,
        ErrorCode.PRV_QUOTA_EXCEEDED,
        ErrorCode.STA_INCONSISTENT,
    }
    warning_codes = {
        ErrorCode.PRV_FALLBACK_USED,
        ErrorCode.GEN_PARTIAL_RESULT,
    }

    if code in critical_codes:
        return "critical"
    if code in warning_codes:
        return "warning"
    return "error"
