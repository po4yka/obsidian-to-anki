"""Post-validator agent for card quality validation.

DEPRECATED: This module is kept for backward compatibility.
Use the post_validation package instead.

This agent validates generated APF cards for:
- APF format syntax compliance
- Factual accuracy vs source content
- Semantic coherence
- Template compliance
"""

# Re-export from new modular structure for backward compatibility
from .post_validation import ErrorCategory, PostValidatorAgent

__all__ = ["PostValidatorAgent", "ErrorCategory"]
