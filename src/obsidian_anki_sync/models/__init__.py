"""Model configuration and selection utilities."""

from .config import (
    ModelConfig,
    ModelPreset,
    ModelTask,
    get_model_config,
    get_model_for_task,
    list_available_models,
    validate_model_name,
)
from .data import (
    Card,
    Manifest,
    ManifestData,
    NoteMetadata,
    QAPair,
    SyncAction,
    ValidationResult,
)

__all__ = [
    # Model configuration
    "ModelConfig",
    "ModelPreset",
    "ModelTask",
    "get_model_config",
    "get_model_for_task",
    "list_available_models",
    "validate_model_name",
    # Data models
    "Card",
    "Manifest",
    "ManifestData",
    "NoteMetadata",
    "QAPair",
    "SyncAction",
    "ValidationResult",
]
