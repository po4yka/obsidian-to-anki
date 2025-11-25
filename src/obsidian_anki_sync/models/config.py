"""Model configuration and selection system.

Provides unified model configuration with presets, capabilities, and per-task customization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelTask(str, Enum):
    """Agent task types."""

    QA_EXTRACTION = "qa_extraction"
    PARSER_REPAIR = "parser_repair"
    PRE_VALIDATION = "pre_validation"
    GENERATION = "generation"
    POST_VALIDATION = "post_validation"
    CONTEXT_ENRICHMENT = "context_enrichment"
    MEMORIZATION_QUALITY = "memorization_quality"
    CARD_SPLITTING = "card_splitting"
    DUPLICATE_DETECTION = "duplicate_detection"


class ModelPreset(str, Enum):
    """Model configuration presets."""

    COST_EFFECTIVE = "cost_effective"  # Lower cost, good quality
    BALANCED = "balanced"  # Best cost/quality balance (default)
    HIGH_QUALITY = "high_quality"  # Maximum quality, higher cost
    FAST = "fast"  # Fastest models, lower quality


@dataclass
class ModelCapabilities:
    """Model capability metadata."""

    supports_structured_outputs: bool = True
    supports_reasoning: bool = False
    max_output_tokens: int = 8192
    context_window: int = 131072
    cost_per_1m_prompt: float = 0.20
    cost_per_1m_completion: float = 0.20
    speed_tier: int = 3  # 1=fastest, 5=slowest
    quality_tier: int = 3  # 1=basic, 5=excellent


@dataclass
class ModelConfig:
    """Model configuration for a specific task."""

    model_name: str
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    reasoning_enabled: bool = False
    capabilities: ModelCapabilities | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"Temperature must be 0-1: {self.temperature}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive: {self.max_tokens}")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be 0-1: {self.top_p}")


# Model capability database
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Qwen 2.5 Series (Recommended)
    "qwen/qwen-2.5-72b-instruct": ModelCapabilities(
        supports_structured_outputs=False,  # Known issues
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.55,
        cost_per_1m_completion=0.55,
        speed_tier=3,
        quality_tier=5,
    ),
    "qwen/qwen-2.5-32b-instruct": ModelCapabilities(
        supports_structured_outputs=False,  # Known issues
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.20,
        cost_per_1m_completion=0.20,
        speed_tier=2,
        quality_tier=4,
    ),
    # DeepSeek Series
    "deepseek/deepseek-chat": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.14,
        cost_per_1m_completion=0.28,
        speed_tier=2,
        quality_tier=5,
    ),
    "deepseek/deepseek-chat-v3.1": ModelCapabilities(
        supports_structured_outputs=False,  # Known issues
        supports_reasoning=True,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.14,
        cost_per_1m_completion=0.28,
        speed_tier=2,
        quality_tier=5,
    ),
    # MiniMax Series
    "minimax/minimax-m2": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.30,
        cost_per_1m_completion=0.30,
        speed_tier=3,
        quality_tier=4,
    ),
    # Moonshot Series
    "moonshotai/kimi-k2": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.25,
        cost_per_1m_completion=0.25,
        speed_tier=3,
        quality_tier=5,
    ),
    "moonshotai/kimi-k2-thinking": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=True,
        max_output_tokens=8192,
        context_window=131072,
        cost_per_1m_prompt=0.50,
        cost_per_1m_completion=0.50,
        speed_tier=4,
        quality_tier=5,
    ),
    # Qwen3 Series (Large models)
    "qwen/qwen3-235b-a22b-2507": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=16384,
        context_window=262144,
        cost_per_1m_prompt=0.08,
        cost_per_1m_completion=0.55,
        speed_tier=5,
        quality_tier=5,
    ),
    "qwen/qwen3-next-80b-a3b-instruct": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=16384,
        context_window=262144,
        cost_per_1m_prompt=0.10,
        cost_per_1m_completion=0.80,
        speed_tier=4,
        quality_tier=5,
    ),
    # xAI Grok Series
    "x-ai/grok-4.1-fast": ModelCapabilities(
        supports_structured_outputs=True,  # OpenRouter normalizes structured outputs
        supports_reasoning=True,  # Supports reasoning but disabled when using JSON schema
        max_output_tokens=32768,  # Higher limit for 2M context model
        context_window=2000000,  # 2M context window
        cost_per_1m_prompt=0.0,  # Free
        cost_per_1m_completion=0.0,  # Free
        speed_tier=2,  # Fast model
        quality_tier=5,  # Best agentic tool calling model
    ),
}

# Model presets - optimized configurations for different use cases
MODEL_PRESETS: dict[ModelPreset, dict[ModelTask, ModelConfig]] = {
    ModelPreset.COST_EFFECTIVE: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.GENERATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.3,
            max_tokens=8192,
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="deepseek/deepseek-chat",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.4,
            max_tokens=8192,
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.2,
            max_tokens=4096,
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
    },
    ModelPreset.BALANCED: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.GENERATION: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.3,
            max_tokens=8192,
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="deepseek/deepseek-chat",
            temperature=0.0,
            max_tokens=8192,
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="minimax/minimax-m2",
            temperature=0.4,
            max_tokens=8192,
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="moonshotai/kimi-k2",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="moonshotai/kimi-k2-thinking",
            temperature=0.2,
            max_tokens=4096,
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
    },
    ModelPreset.HIGH_QUALITY: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.0,
            max_tokens=8192,
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.GENERATION: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.3,
            max_tokens=8192,
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="deepseek/deepseek-chat",
            temperature=0.0,
            max_tokens=8192,
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="moonshotai/kimi-k2",
            temperature=0.4,
            max_tokens=8192,
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="moonshotai/kimi-k2",
            temperature=0.0,
            max_tokens=4096,
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="moonshotai/kimi-k2-thinking",
            temperature=0.2,
            max_tokens=4096,
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-72b-instruct",
            temperature=0.0,
            max_tokens=4096,
        ),
    },
    ModelPreset.FAST: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Lower limit for speed
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=2048,
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=2048,
        ),
        ModelTask.GENERATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.3,
            max_tokens=4096,
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="deepseek/deepseek-chat",
            temperature=0.0,
            max_tokens=2048,
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.4,
            max_tokens=4096,
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=2048,
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.2,
            max_tokens=2048,
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=2048,
        ),
    },
}


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """Get capabilities for a model.

    Args:
        model_name: Model identifier

    Returns:
        Model capabilities (defaults if unknown)
    """
    return MODEL_CAPABILITIES.get(
        model_name,
        ModelCapabilities(),  # Default capabilities
    )


def get_model_config(
    task: ModelTask,
    preset: ModelPreset = ModelPreset.BALANCED,
    overrides: dict[str, Any] | None = None,
) -> ModelConfig:
    """Get model configuration for a task.

    Args:
        task: The task type
        preset: Model preset to use
        overrides: Optional overrides for model settings

    Returns:
        Model configuration
    """
    preset_configs = MODEL_PRESETS.get(preset, MODEL_PRESETS[ModelPreset.BALANCED])
    config = preset_configs.get(task)

    if config is None:
        # Fallback to balanced preset
        logger.warning(
            "model_config_not_found",
            task=task.value,
            preset=preset.value,
            note="Using balanced preset default",
        )
        config = MODEL_PRESETS[ModelPreset.BALANCED][task]

    # Apply overrides
    if overrides:
        config_dict = {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "reasoning_enabled": config.reasoning_enabled,
        }
        config_dict.update(overrides)
        config = ModelConfig(**config_dict)

    # Attach capabilities
    config.capabilities = get_model_capabilities(config.model_name)

    return config


def get_model_for_task(
    task: ModelTask | str,
    preset: ModelPreset | str = ModelPreset.BALANCED,
    **overrides: Any,
) -> str:
    """Get model name for a task (simplified interface).

    Args:
        task: Task type (string or enum)
        preset: Preset to use (string or enum)
        **overrides: Optional model setting overrides

    Returns:
        Model name string
    """
    if isinstance(task, str):
        try:
            task = ModelTask(task)
        except ValueError:
            logger.warning(
                "unknown_task_type",
                task=task,
                note="Using QA_EXTRACTION as default",
            )
            task = ModelTask.QA_EXTRACTION

    if isinstance(preset, str):
        try:
            preset = ModelPreset(preset)
        except ValueError:
            logger.warning(
                "unknown_preset",
                preset=preset,
                note="Using BALANCED as default",
            )
            preset = ModelPreset.BALANCED

    config = get_model_config(task, preset, overrides)
    return config.model_name


def validate_model_name(model_name: str) -> bool:
    """Validate that a model name is known.

    Args:
        model_name: Model identifier

    Returns:
        True if model is known, False otherwise
    """
    return model_name in MODEL_CAPABILITIES


def list_available_models() -> list[str]:
    """List all available models with capabilities.

    Returns:
        List of model names
    """
    return list(MODEL_CAPABILITIES.keys())
