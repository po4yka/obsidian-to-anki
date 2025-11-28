"""JSON schemas for structured LLM outputs.

Provides JSON Schema definitions for all agents to ensure reliable,
type-safe responses from OpenRouter and other compatible providers.
"""

from typing import Any


def get_qa_extraction_schema() -> dict[str, Any]:
    """Get JSON schema for Q&A extraction responses.

    Returns:
        JSON schema dictionary for structured Q&A extraction
    """
    return {
        "name": "qa_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "qa_pairs": {
                    "type": "array",
                    "description": "List of extracted question-answer pairs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "card_index": {
                                "type": "integer",
                                "description": "Sequential card number starting from 1",
                                "minimum": 1,
                            },
                            "question_en": {
                                "type": "string",
                                "description": "Question in English (if applicable)",
                            },
                            "question_ru": {
                                "type": "string",
                                "description": "Question in Russian (if applicable)",
                            },
                            "answer_en": {
                                "type": "string",
                                "description": "Answer in English (if applicable)",
                                "maxLength": 10000,
                            },
                            "answer_ru": {
                                "type": "string",
                                "description": "Answer in Russian (if applicable)",
                                "maxLength": 10000,
                            },
                            "context": {
                                "type": "string",
                                "description": "Contextual information before the Q&A",
                                "maxLength": 2000,
                            },
                            "followups": {
                                "type": "string",
                                "description": "Follow-up questions or related queries",
                                "maxLength": 1000,
                            },
                            "references": {
                                "type": "string",
                                "description": "References or citations",
                                "maxLength": 1000,
                            },
                            "related": {
                                "type": "string",
                                "description": "Related topics or questions",
                                "maxLength": 1000,
                            },
                        },
                        "required": [
                            "card_index",
                            "question_en",
                            "question_ru",
                            "answer_en",
                            "answer_ru",
                            "context",
                            "followups",
                            "references",
                            "related",
                        ],
                        "additionalProperties": False,
                    },
                },
                "extraction_notes": {
                    "type": "string",
                    "description": "Brief notes about the extraction process (keep concise)",
                    "maxLength": 500,
                },
                "total_pairs": {
                    "type": "integer",
                    "description": "Total number of Q&A pairs extracted",
                    "minimum": 0,
                },
            },
            "required": ["qa_pairs", "extraction_notes", "total_pairs"],
            "additionalProperties": False,
        },
    }


def get_quality_assessment_schema() -> dict[str, Any]:
    """Get JSON schema for quality assessment responses.

    Returns:
        JSON schema dictionary for quality assessment results
    """
    return {
        "name": "quality_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall quality score",
                },
                "dimensions": {
                    "type": "object",
                    "description": "Quality scores by dimension",
                    "properties": {
                        "content": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "learning_science": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "technical": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "accessibility": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actionable improvement suggestions",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the assessment",
                },
            },
            "required": ["overall_score", "dimensions", "suggestions", "confidence"],
        },
    }


def get_pre_validation_schema() -> dict[str, Any]:
    """Get JSON schema for pre-validation responses.

    Returns:
        JSON schema dictionary for pre-validation results
    """
    return {
        "name": "pre_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_valid": {
                    "type": "boolean",
                    "description": "Whether the note passes pre-validation",
                },
                "error_type": {
                    "type": "string",
                    "enum": ["format", "structure", "frontmatter", "content", "none"],
                    "description": "Type of error detected",
                },
                "error_details": {
                    "type": "string",
                    "description": "Detailed error message",
                },
                "auto_fix_applied": {
                    "type": "boolean",
                    "description": "Whether auto-fix was applied",
                },
                "fixed_content": {
                    "type": ["string", "null"],
                    "description": "Fixed content if auto-fix was applied",
                },
                "validation_time": {
                    "type": "number",
                    "description": "Time taken for validation in seconds",
                },
            },
            "required": [
                "is_valid",
                "error_type",
                "error_details",
                "auto_fix_applied",
                "validation_time",
            ],
            "additionalProperties": False,
        },
    }


def get_post_validation_schema() -> dict[str, Any]:
    """Get JSON schema for post-validation responses.

    Returns:
        JSON schema dictionary for post-validation results
    """
    return {
        "name": "post_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_valid": {
                    "type": "boolean",
                    "description": "Whether the generated cards pass validation",
                },
                "error_type": {
                    "type": "string",
                    "enum": ["syntax", "factual", "semantic", "template", "none"],
                    "description": "Type of error detected",
                },
                "error_details": {
                    "type": "string",
                    "description": "Detailed error message",
                },
                "validation_time": {
                    "type": "number",
                    "description": "Time taken for validation in seconds",
                },
            },
            "required": ["is_valid", "error_type", "error_details", "validation_time"],
            "additionalProperties": False,
        },
    }


def get_generation_schema() -> dict[str, Any]:
    """Get JSON schema for card generation responses.

    Returns:
        JSON schema dictionary for card generation results
    """
    return {
        "name": "card_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "description": "List of generated cards",
                    "items": {
                        "type": "object",
                        "properties": {
                            "card_index": {
                                "type": "integer",
                                "description": "Card index starting from 1",
                                "minimum": 1,
                            },
                            "slug": {
                                "type": "string",
                                "description": "Unique card identifier",
                                "minLength": 1,
                            },
                            "lang": {
                                "type": "string",
                                "enum": ["en", "ru"],
                                "description": "Card language",
                            },
                            "apf_html": {
                                "type": "string",
                                "description": "APF HTML content",
                                "minLength": 1,
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Generation confidence score",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": [
                            "card_index",
                            "slug",
                            "lang",
                            "apf_html",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "total_cards": {
                    "type": "integer",
                    "description": "Total number of cards generated",
                    "minimum": 0,
                },
                "generation_time": {
                    "type": "number",
                    "description": "Time taken for generation in seconds",
                    "minimum": 0.0,
                },
                "model_used": {
                    "type": "string",
                    "description": "Model used for generation",
                },
            },
            "required": ["cards", "total_cards", "generation_time", "model_used"],
            "additionalProperties": False,
        },
    }


def get_parser_repair_schema() -> dict[str, Any]:
    """Get JSON schema for parser repair responses.

    Returns:
        JSON schema dictionary for parser repair results
    """
    return {
        "name": "parser_repair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_repairable": {
                    "type": "boolean",
                    "description": "Whether the note can be repaired",
                },
                "diagnosis": {
                    "type": "string",
                    "description": "Diagnosis of the issue",
                },
                "repairs": {
                    "type": "array",
                    "description": "List of repairs applied",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["type", "description"],
                        "additionalProperties": False,
                    },
                },
                "repaired_content": {
                    "type": ["string", "null"],
                    "description": "Repaired content if successful",
                },
                "content_generation_applied": {
                    "type": "boolean",
                    "description": "Whether content generation was applied (missing sections generated)",
                },
                "generated_sections": {
                    "type": "array",
                    "description": "List of sections that were generated (not just repaired)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_type": {
                                "type": "string",
                                "enum": [
                                    "question_en",
                                    "question_ru",
                                    "answer_en",
                                    "answer_ru",
                                ],
                            },
                            "method": {
                                "type": "string",
                                "enum": ["translation", "inference", "completion"],
                                "description": "How the content was generated",
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["section_type", "method", "description"],
                        "additionalProperties": False,
                    },
                },
                "error_diagnosis": {
                    "type": ["object", "null"],
                    "description": "Structured error diagnosis with categorization and severity",
                    "properties": {
                        "error_category": {
                            "type": "string",
                            "enum": [
                                "syntax",
                                "structure",
                                "content",
                                "quality",
                                "frontmatter",
                                "unknown",
                            ],
                            "description": "Category of the error",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Severity of the error",
                        },
                        "error_description": {
                            "type": "string",
                            "description": "Detailed description of the error",
                        },
                        "repair_priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Repair priority (1=highest, 10=lowest)",
                        },
                        "can_auto_fix": {
                            "type": "boolean",
                            "description": "Whether this error can be auto-fixed",
                        },
                    },
                    "required": [
                        "error_category",
                        "severity",
                        "error_description",
                        "repair_priority",
                        "can_auto_fix",
                    ],
                    "additionalProperties": False,
                },
                "quality_before": {
                    "type": ["object", "null"],
                    "description": "Quality score before repair",
                    "properties": {
                        "completeness_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Content completeness (0-1)",
                        },
                        "structure_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Structure quality (0-1)",
                        },
                        "bilingual_consistency": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Consistency between languages (0-1)",
                        },
                        "technical_accuracy": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Technical accuracy score (0-1)",
                        },
                        "overall_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Overall quality score (0-1)",
                        },
                        "issues_found": {
                            "type": "array",
                            "description": "List of quality issues identified",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "completeness_score",
                        "structure_score",
                        "bilingual_consistency",
                        "technical_accuracy",
                        "overall_score",
                        "issues_found",
                    ],
                    "additionalProperties": False,
                },
                "quality_after": {
                    "type": ["object", "null"],
                    "description": "Quality score after repair",
                    "properties": {
                        "completeness_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Content completeness (0-1)",
                        },
                        "structure_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Structure quality (0-1)",
                        },
                        "bilingual_consistency": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Consistency between languages (0-1)",
                        },
                        "technical_accuracy": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Technical accuracy score (0-1)",
                        },
                        "overall_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Overall quality score (0-1)",
                        },
                        "issues_found": {
                            "type": "array",
                            "description": "List of quality issues identified",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "completeness_score",
                        "structure_score",
                        "bilingual_consistency",
                        "technical_accuracy",
                        "overall_score",
                        "issues_found",
                    ],
                    "additionalProperties": False,
                },
                "repair_time": {
                    "type": "number",
                    "description": "Time taken for repair in seconds",
                },
            },
            "required": [
                "is_repairable",
                "diagnosis",
                "repairs",
                "repaired_content",
                "content_generation_applied",
                "generated_sections",
                "repair_time",
            ],
            "additionalProperties": False,
        },
    }
