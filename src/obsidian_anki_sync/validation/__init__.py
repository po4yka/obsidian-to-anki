"""Vault validation package for Obsidian Q&A notes.

This package provides comprehensive validation of Q&A notes including:
- YAML frontmatter validation
- Content structure validation
- Wikilink validation
- Format and naming validation
- Android-specific validation rules
- AI-powered fixes and translation
"""

from .ai_fixer import AIFixer, AIFixerValidator
from .android_validator import AndroidValidator
from .base import AutoFix, BaseValidator, Severity, ValidationIssue
from .content_validator import ContentValidator
from .format_validator import FormatValidator
from .hash_tracker import HashTracker
from .link_validator import LinkValidator
from .orchestrator import NoteValidator
from .parallel_validator import ParallelConfig, ParallelValidator, validate_directory_parallel
from .report_generator import ReportGenerator
from .taxonomy_loader import TaxonomyLoader
from .yaml_validator import YAMLValidator

__all__ = [
    # Base types
    "Severity",
    "ValidationIssue",
    "AutoFix",
    "BaseValidator",
    # Validators
    "YAMLValidator",
    "ContentValidator",
    "FormatValidator",
    "LinkValidator",
    "AndroidValidator",
    "AIFixerValidator",
    # AI System
    "AIFixer",
    # Orchestrator
    "NoteValidator",
    # Parallel Processing
    "ParallelConfig",
    "ParallelValidator",
    "validate_directory_parallel",
    # Utilities
    "TaxonomyLoader",
    "ReportGenerator",
    "HashTracker",
]
