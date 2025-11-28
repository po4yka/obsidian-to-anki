"""Debug artifact saving for LLM operations."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class DebugArtifactSaver:
    """Saves debug artifacts (prompts, responses, errors) for failed LLM operations."""

    def __init__(self, artifacts_dir: Path | None = None, enabled: bool = True):
        """Initialize debug artifact saver.

        Args:
            artifacts_dir: Directory to save artifacts (default: .debug_artifacts/)
            enabled: Whether artifact saving is enabled
        """
        self.enabled = enabled
        if artifacts_dir is None:
            # Use a secure default path that resolves to current working directory
            self.artifacts_dir = Path.cwd() / ".debug_artifacts"
        else:
            # Resolve the provided path to prevent path traversal
            self.artifacts_dir = artifacts_dir.resolve()

        if self.enabled:
            # Create parent directories if they don't exist
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "debug_artifacts_enabled", artifacts_dir=str(self.artifacts_dir)
            )

    def save_llm_failure(
        self,
        operation: str,
        model: str,
        prompt: str,
        system_prompt: str,
        response: str | None,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save debug artifacts for a failed LLM operation.

        Args:
            operation: Operation name (e.g., "card_generation", "validation")
            model: Model being used
            prompt: User prompt sent to LLM
            system_prompt: System prompt
            response: Response received (if any)
            error: Exception that occurred
            context: Additional context (slug, card_index, etc.)

        Returns:
            Path to saved artifact file, or None if disabled
        """
        if not self.enabled:
            return None

        try:
            # Create timestamp-based filename
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            safe_operation = operation.replace("/", "_").replace(" ", "_")
            filename = f"{timestamp}_{safe_operation}_{model.replace(':', '_')}.json"
            filepath = self.artifacts_dir / filename

            # Build artifact data
            artifact = {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation": operation,
                "model": model,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "prompts": {
                    "system": system_prompt,
                    "user": prompt,
                    "user_length": len(prompt),
                    "system_length": len(system_prompt),
                    "total_length": len(prompt) + len(system_prompt),
                },
                "response": {
                    "text": response,
                    "length": len(response) if response else 0,
                },
                "context": context or {},
            }

            # Save to file
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)

            logger.info(
                "debug_artifact_saved",
                operation=operation,
                filepath=str(filepath),
                prompt_length=len(prompt),
                error_type=type(error).__name__,
            )

            return filepath

        except Exception as save_error:
            logger.warning(
                "failed_to_save_debug_artifact",
                error=str(save_error),
                original_error=str(error),
            )
            return None

    def cleanup_old_artifacts(self, max_age_days: int = 7) -> int:
        """Clean up old debug artifacts.

        Args:
            max_age_days: Maximum age of artifacts to keep (in days)

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.artifacts_dir.exists():
            return 0

        deleted = 0
        cutoff_timestamp = datetime.now(UTC).timestamp() - (max_age_days * 86400)

        try:
            for filepath in self.artifacts_dir.glob("*.json"):
                if filepath.stat().st_mtime < cutoff_timestamp:
                    filepath.unlink()
                    deleted += 1

            if deleted > 0:
                logger.info(
                    "cleaned_up_old_artifacts",
                    deleted=deleted,
                    max_age_days=max_age_days,
                )

        except Exception as e:
            logger.warning("artifact_cleanup_failed", error=str(e))

        return deleted


# Global instance for easy access
_global_saver: DebugArtifactSaver | None = None


def get_artifact_saver() -> DebugArtifactSaver:
    """Get the global debug artifact saver instance.

    Returns:
        Global DebugArtifactSaver instance
    """
    global _global_saver
    if _global_saver is None:
        _global_saver = DebugArtifactSaver(enabled=True)
    return _global_saver


def save_failed_llm_call(
    operation: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    response: str | None = None,
    error: Exception | None = None,
    **context: Any,
) -> Path | None:
    """Convenience function to save a failed LLM call.

    Args:
        operation: Operation name
        model: Model being used
        prompt: User prompt
        system_prompt: System prompt
        response: Response received (if any)
        error: Exception that occurred
        **context: Additional context

    Returns:
        Path to saved artifact, or None if disabled/failed
    """
    saver = get_artifact_saver()
    if error is None:
        error = Exception("Unknown error")
    return saver.save_llm_failure(
        operation=operation,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        response=response,
        error=error,
        context=context,
    )
