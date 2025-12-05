"""Model configuration and selection system.

Provides unified model configuration with presets, capabilities, and per-task customization.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ModelTask(str, Enum):
    """Agent task types."""

    QA_EXTRACTION = "qa_extraction"
    PARSER_REPAIR = "parser_repair"
    PRE_VALIDATION = "pre_validation"
    HIGHLIGHT = "highlight"
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


class ModelCapabilities(BaseModel):
    """Model capability metadata."""

    model_config = ConfigDict(extra="allow")

    supports_structured_outputs: bool = Field(
        default=True, description="Supports structured JSON outputs"
    )
    supports_reasoning: bool = Field(
        default=False, description="Supports reasoning/thinking mode"
    )
    max_output_tokens: int = Field(
        default=8192, ge=1, description="Maximum output tokens"
    )
    context_window: int = Field(
        default=131072, ge=1, description="Context window size in tokens"
    )
    cost_per_1m_prompt: float = Field(
        default=0.20, ge=0, description="Cost per million prompt tokens"
    )
    cost_per_1m_completion: float = Field(
        default=0.20, ge=0, description="Cost per million completion tokens"
    )
    speed_tier: int = Field(
        default=3, ge=1, le=5, description="Speed tier (1=fastest, 5=slowest)"
    )
    quality_tier: int = Field(
        default=3, ge=1, le=5, description="Quality tier (1=basic, 5=excellent)"
    )


class ModelConfig(BaseModel):
    """Model configuration for a specific task."""

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(min_length=1, description="Model identifier/name")
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    reasoning_enabled: bool = Field(
        default=False, description="Enable reasoning/thinking mode"
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Preferred reasoning effort when supported (auto|minimal|low|medium|high|none)",
    )
    capabilities: ModelCapabilities | None = Field(
        default=None, description="Model capabilities"
    )

    @field_validator("temperature", "top_p", mode="after")
    @classmethod
    def validate_temperature_range(cls, v: float | None) -> float | None:
        """Validate temperature and top_p are in valid range."""
        if v is not None and not (0.0 <= v <= 1.0):
            msg = f"Value must be between 0.0 and 1.0: {v}"
            raise ValueError(msg)
        return v


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
    "deepseek/deepseek-v3.2": ModelCapabilities(
        supports_structured_outputs=True,  # Supports tool calling well
        supports_reasoning=True,  # Controllable via reasoning.enabled
        max_output_tokens=16384,
        context_window=163840,  # 163K context (latest)
        cost_per_1m_prompt=0.14,  # Updated pricing (December 2025)
        cost_per_1m_completion=0.56,
        speed_tier=3,
        quality_tier=5,  # Highest tier - GPT-5 class reasoning
    ),
    # MiniMax Series
    "minimax/minimax-m2": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=204800,  # 204K context (latest)
        cost_per_1m_prompt=0.10,  # Updated pricing (December 2025)
        cost_per_1m_completion=0.40,
        speed_tier=3,
        quality_tier=5,  # Excellent for coding and agentic tasks
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
        max_output_tokens=16384,  # Latest: Supports larger outputs
        context_window=262144,  # 256K context (latest)
        cost_per_1m_prompt=0.12,  # Updated pricing (December 2025)
        cost_per_1m_completion=0.48,
        speed_tier=4,  # Slower due to thinking process
        quality_tier=5,  # Excellent for complex reasoning tasks
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
    # Latest Qwen 2.5 models (December 2025)
    "qwen/qwen-2.5-7b-instruct": ModelCapabilities(
        supports_structured_outputs=True,
        supports_reasoning=False,
        max_output_tokens=8192,
        context_window=32768,  # 33K context
        cost_per_1m_prompt=0.04,  # Very cost-effective
        cost_per_1m_completion=0.10,
        speed_tier=2,  # Fast
        quality_tier=4,  # Good quality for size
    ),
}

# Default model for all tasks - DeepSeek V3.2 (latest, excellent reasoning, 163K context)
DEFAULT_MODEL = "deepseek/deepseek-v3.2"

# Model presets - optimized configurations for different use cases
# All presets use qwen/deepseek/kimi/minimax models
MODEL_PRESETS: dict[ModelPreset, dict[ModelTask, ModelConfig]] = {
    ModelPreset.COST_EFFECTIVE: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            # Latest: Excellent reasoning for complex QA extraction
            model_name="deepseek/deepseek-v3.2",
            temperature=0.0,
            max_tokens=16384,  # Increased: DeepSeek V3.2 supports larger outputs
            reasoning_enabled=True,  # Enable reasoning for complex QA analysis
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            # Latest: Excellent reasoning for error analysis and repair
            model_name="deepseek/deepseek-v3.2",
            temperature=0.0,
            max_tokens=16384,  # Increased: DeepSeek V3.2 supports larger outputs
            reasoning_enabled=True,  # Enable reasoning for error analysis
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            # Latest: Smaller, faster, cheaper ($0.04/$0.10)
            model_name="qwen/qwen-2.5-7b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate: Pre-validation is typically simpler
            reasoning_enabled=False,  # Rule-based validation doesn't need reasoning
        ),
        ModelTask.HIGHLIGHT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.1,
            max_tokens=8192,  # Highlighting may need longer excerpts
            reasoning_enabled=True,  # Enable reasoning for candidate extraction
        ),
        ModelTask.GENERATION: ModelConfig(
            # Cost-effective: Good quality at reasonable cost
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.3,
            max_tokens=24576,  # Increased: Generation benefits from larger outputs
            reasoning_enabled=True,  # Enable reasoning for high-quality generation
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            # Latest: Excellent reasoning for quality assessment
            model_name="deepseek/deepseek-v3.2",
            temperature=0.0,
            max_tokens=16384,  # Increased: DeepSeek V3.2 supports larger outputs
            reasoning_enabled=True,  # Enable reasoning for quality assessment
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            # Latest: Excellent for creative/agentic tasks (204K context)
            model_name="minimax/minimax-m2",
            temperature=0.4,
            max_tokens=16384,  # Increased: Context enrichment can be creative
            reasoning_enabled=True,  # Enable reasoning for creative enhancement
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate: Quality assessment is analytical
            reasoning_enabled=True,  # Enable reasoning for memorization analysis
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            # Latest: 256K context, thinking model for complex decisions
            model_name="moonshotai/kimi-k2-thinking",
            temperature=0.2,
            max_tokens=16384,  # Increased: Kimi K2 Thinking supports larger outputs
            reasoning_enabled=True,  # Enable reasoning for complex decision making
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate: Duplicate detection is comparative
            reasoning_enabled=True,  # Enable reasoning for similarity analysis
        ),
    },
    ModelPreset.BALANCED: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=24576,  # Increased: Leverage 2M context for comprehensive QA
            reasoning_enabled=True,  # Enable reasoning for comprehensive analysis
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=16384,  # Increased: Better error analysis and repair
            reasoning_enabled=True,  # Enable reasoning for repair logic
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,  # Increased: More thorough pre-validation
            reasoning_enabled=False,  # Keep simple for speed
        ),
        ModelTask.HIGHLIGHT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.1,
            max_tokens=12288,  # Allow larger excerpts for summaries
            reasoning_enabled=True,
        ),
        ModelTask.GENERATION: ModelConfig(
            # Latest: Excellent reasoning, 163K context for high-quality generation
            model_name="deepseek/deepseek-v3.2",
            temperature=0.3,
            max_tokens=16384,  # DeepSeek V3.2 max output tokens
            reasoning_enabled=True,  # Enable reasoning for balanced quality generation
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=16384,  # Increased: Detailed validation feedback
            reasoning_enabled=True,  # Enable reasoning for balanced validation
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.4,
            max_tokens=24576,  # Increased: Rich context and examples
            reasoning_enabled=True,  # Enable reasoning for balanced enrichment
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,  # Increased: Comprehensive quality analysis
            reasoning_enabled=True,  # Enable reasoning for quality assessment
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.2,
            max_tokens=16384,  # Increased: Complex decision reasoning
            reasoning_enabled=True,  # Enable reasoning for decision making
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,  # Increased: Better comparison analysis
            reasoning_enabled=True,  # Enable reasoning for similarity analysis
        ),
    },
    ModelPreset.HIGH_QUALITY: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=32768,  # Max: Exhaustive QA extraction with full context
            reasoning_enabled=True,  # Enable reasoning for exhaustive analysis
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=24576,  # Increased: Maximum analysis for complex repairs
            reasoning_enabled=True,  # Enable reasoning for sophisticated repair logic
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=16384,  # Increased: Rigorous pre-validation
            reasoning_enabled=False,  # Keep deterministic for reliability
        ),
        ModelTask.HIGHLIGHT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.1,
            max_tokens=16384,  # Maximum context for deep analysis
            reasoning_enabled=True,
        ),
        ModelTask.GENERATION: ModelConfig(
            # Latest: Premium reasoning model for highest quality
            model_name="deepseek/deepseek-v3.2",
            temperature=0.2,  # Lower temperature for higher quality
            max_tokens=16384,  # DeepSeek V3.2 max output tokens
            reasoning_enabled=True,  # Enable reasoning for premium quality
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=24576,  # Increased: Comprehensive validation with detailed feedback
            reasoning_enabled=True,  # Enable reasoning for deep validation analysis
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=32768,  # Max: Rich, detailed context enrichment
            reasoning_enabled=True,  # Enable reasoning for sophisticated enrichment
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=16384,  # Increased: Deep quality analysis
            reasoning_enabled=True,  # Enable reasoning for expert quality assessment
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.1,  # Very low for consistent decisions
            max_tokens=24576,  # Increased: Sophisticated reasoning for splitting
            reasoning_enabled=True,  # Enable reasoning for expert decision making
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=16384,  # Increased: Precise duplicate analysis
            reasoning_enabled=True,  # Enable reasoning for expert similarity analysis
        ),
    },
    ModelPreset.FAST: {
        ModelTask.QA_EXTRACTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=8192,  # Slightly increased: Still fast but more capable
            # Skip reasoning for speed - QA extraction is often straightforward
            reasoning_enabled=False,
        ),
        ModelTask.PARSER_REPAIR: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate increase for better repairs
            # Enable reasoning for error analysis (critical for repairs)
            reasoning_enabled=True,
        ),
        ModelTask.PRE_VALIDATION: ModelConfig(
            # Latest: Smaller, faster, cheaper for speed preset
            model_name="qwen/qwen-2.5-7b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Increased: Speed-focused but thorough
            reasoning_enabled=False,  # Keep fast and deterministic
        ),
        ModelTask.HIGHLIGHT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.1,
            max_tokens=6144,  # Keep fast while allowing summaries
            reasoning_enabled=True,
        ),
        ModelTask.GENERATION: ModelConfig(
            # Latest: Smaller model for faster generation
            model_name="qwen/qwen-2.5-7b-instruct",
            temperature=0.3,
            max_tokens=8192,  # Increased: Faster generation with good quality
            reasoning_enabled=False,  # Skip reasoning for speed - rely on model training
        ),
        ModelTask.POST_VALIDATION: ModelConfig(
            # Latest: Smaller model for faster validation
            model_name="qwen/qwen-2.5-7b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate increase for validation
            reasoning_enabled=False,  # Keep fast for validation pipeline
        ),
        ModelTask.CONTEXT_ENRICHMENT: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.4,
            max_tokens=8192,  # Increased: Quick but useful enrichment
            # Skip reasoning for speed - enrichment can be creative without it
            reasoning_enabled=False,
        ),
        ModelTask.MEMORIZATION_QUALITY: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate increase for quality checks
            reasoning_enabled=False,  # Keep fast for quality checks
        ),
        ModelTask.CARD_SPLITTING: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.2,
            max_tokens=4096,  # Moderate increase for decision making
            # Enable reasoning for decision making (critical for correctness)
            reasoning_enabled=True,
        ),
        ModelTask.DUPLICATE_DETECTION: ModelConfig(
            model_name="qwen/qwen-2.5-32b-instruct",
            temperature=0.0,
            max_tokens=4096,  # Moderate increase for comparisons
            reasoning_enabled=False,  # Keep fast - similarity can be pattern-based
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
    preset_configs = MODEL_PRESETS.get(
        preset, MODEL_PRESETS[ModelPreset.BALANCED])
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
        config_dict: dict[str, Any] = {
            "model_name": str(config.model_name),
            "temperature": float(config.temperature),
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "reasoning_enabled": bool(config.reasoning_enabled),
        }
        # Update with overrides, ensuring types match
        for key, value in overrides.items():
            if key in config_dict:
                if key == "model_name":
                    config_dict[key] = str(value)
                elif key == "temperature":
                    config_dict[key] = float(value)
                elif key == "max_tokens":
                    config_dict[key] = int(
                        value) if value is not None else None
                elif key == "top_p":
                    config_dict[key] = float(
                        value) if value is not None else None
                elif key == "reasoning_enabled":
                    config_dict[key] = bool(value)
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
