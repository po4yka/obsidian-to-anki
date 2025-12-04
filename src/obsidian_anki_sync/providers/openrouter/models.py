"""Model configuration and constants for OpenRouter provider."""

# Models known to have issues with strict structured outputs
MODELS_WITH_STRUCTURED_OUTPUT_ISSUES = {
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3.1:free",
    "moonshotai/kimi-k2-thinking",
    # Qwen models may have issues with strict structured outputs
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen3-max",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-32b",  # Smaller model, may have structured output quirks
    "qwen/qwen3-30b-a3b",  # Smaller model, may have structured output quirks
    # Note: deepseek/deepseek-v3.2 supports tool calling well, not listed here
}

# Models that support OpenRouter's reasoning-effort contract and their default auto effort
MODELS_WITH_REASONING_SUPPORT: dict[str, str] = {
    "x-ai/grok-4.1-fast:free": "medium",
    "x-ai/grok-4.1": "medium",
    "x-ai/grok-3": "medium",
    "deepseek/deepseek-chat-v3.1": "medium",
    "deepseek/deepseek-v3.2": "medium",  # Supports reasoning via reasoning.enabled boolean
    "deepseek/deepseek-r1": "high",
    "openai/o3-mini": "medium",
    "openai/o3-mini-high": "high",
}

# Model context window sizes (in tokens)
# Default: 131072 (128k) for most modern models
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "moonshotai/kimi-k2": 131072,
    "moonshotai/kimi-k2-thinking": 131072,
    "qwen/qwen-2.5-72b-instruct": 131072,
    "qwen/qwen-2.5-32b-instruct": 131072,
    "deepseek/deepseek-chat-v3.1": 131072,
    "deepseek/deepseek-chat": 131072,
    "deepseek/deepseek-v3.2": 163840,  # 163K context window
    "minimax/minimax-m2": 131072,
    # Qwen3 models support larger context windows
    "qwen/qwen3-235b-a22b-2507": 262144,  # 262K context
    "qwen/qwen3-235b-a22b-thinking-2507": 262144,  # 262K context
    "qwen/qwen3-next-80b-a3b-instruct": 262144,  # 262K context
    "qwen/qwen3-32b": 131072,  # 128K context (standard)
    "qwen/qwen3-30b-a3b": 131072,  # 128K context (standard)
    # xAI Grok Series
    "x-ai/grok-4.1-fast:free": 2000000,  # 2M context window
}
DEFAULT_CONTEXT_WINDOW = 131072  # 128k tokens
CONTEXT_SAFETY_MARGIN = 1000  # Reserve tokens for safety

# Model-specific max output token limits
# Most models have output limits much lower than their context window
# These are conservative limits to avoid 400 Bad Request errors
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "qwen/qwen-2.5-72b-instruct": 8192,  # Conservative limit
    "qwen/qwen-2.5-32b-instruct": 8192,  # Conservative limit
    "deepseek/deepseek-chat": 8192,
    "deepseek/deepseek-chat-v3.1": 8192,
    "deepseek/deepseek-v3.2": 16384,  # Good output capacity with 163K context
    "minimax/minimax-m2": 8192,
    "moonshotai/kimi-k2": 8192,
    "moonshotai/kimi-k2-thinking": 8192,
    # Qwen3 models may support larger outputs
    "qwen/qwen3-235b-a22b-2507": 16384,
    "qwen/qwen3-235b-a22b-thinking-2507": 16384,
    "qwen/qwen3-next-80b-a3b-instruct": 16384,
    "qwen/qwen3-32b": 8192,
    "qwen/qwen3-30b-a3b": 8192,
    # xAI Grok Series - supports larger outputs due to 2M context window
    "x-ai/grok-4.1-fast:free": 32768,  # Conservative limit for 2M context model
}
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Safe default for most models

# Models that support structured outputs well
MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS = {
    "openai/gpt-4",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    "google/gemini-pro",
    "google/gemini-2.0-flash-exp",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "qwen/qwen3-max",
    # xAI Grok 4.1 Fast - excellent structured output support
    # Best practices: temperature=0 for deterministic output, disable reasoning for JSON
    "x-ai/grok-4.1-fast:free",
    # DeepSeek V3.2 - supports tool calling and structured outputs
    # GPT-5 class reasoning, gold-medal performance on IMO/IOI 2025
    "deepseek/deepseek-v3.2",
}

# HTTP status codes that should be retried
HTTP_STATUS_RETRYABLE = {429, 500, 502, 503, 504}
