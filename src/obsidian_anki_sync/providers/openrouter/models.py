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
    # Note: deepseek/deepseek-v3.2 supports tool calling well, but fails strict schema
    "deepseek/deepseek-v3.2",
}

# Models that support OpenRouter's reasoning-effort contract and their default auto effort
MODELS_WITH_REASONING_SUPPORT: dict[str, str] = {
    "deepseek/deepseek-chat-v3.1": "medium",
    # Latest DeepSeek V3.2 - excellent reasoning capabilities
    "deepseek/deepseek-v3.2": "medium",
    "deepseek/deepseek-r1": "high",
    # Latest Kimi K2 Thinking - advanced reasoning model
    "moonshotai/kimi-k2-thinking": "high",
}

# Model context window sizes (in tokens)
# Default: 131072 (128k) for most modern models
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "moonshotai/kimi-k2": 131072,
    "moonshotai/kimi-k2-thinking": 262144,  # 256K context window (latest)
    "qwen/qwen-2.5-72b-instruct": 131072,
    "qwen/qwen-2.5-32b-instruct": 131072,
    "qwen/qwen-2.5-7b-instruct": 32768,  # 33K context window (latest)
    "deepseek/deepseek-chat-v3.1": 131072,
    "deepseek/deepseek-chat": 131072,
    "deepseek/deepseek-v3.2": 163840,  # 163K context window (latest)
    "minimax/minimax-m2": 204800,  # 204K context window (latest)
    # Qwen3 models support larger context windows
    "qwen/qwen3-235b-a22b-2507": 262144,  # 262K context
    "qwen/qwen3-235b-a22b-thinking-2507": 262144,  # 262K context
    "qwen/qwen3-next-80b-a3b-instruct": 262144,  # 262K context
    "qwen/qwen3-32b": 131072,  # 128K context (standard)
    "qwen/qwen3-30b-a3b": 131072,  # 128K context (standard)
}
DEFAULT_CONTEXT_WINDOW = 131072  # 128k tokens
CONTEXT_SAFETY_MARGIN = 1000  # Reserve tokens for safety

# Model-specific max output token limits
# Most models have output limits much lower than their context window
# These are conservative limits to avoid 400 Bad Request errors
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "qwen/qwen-2.5-72b-instruct": 8192,  # Conservative limit
    "qwen/qwen-2.5-32b-instruct": 8192,  # Conservative limit
    "qwen/qwen-2.5-7b-instruct": 8192,  # Latest: Standard output limit
    "deepseek/deepseek-chat": 8192,
    "deepseek/deepseek-chat-v3.1": 8192,
    "deepseek/deepseek-v3.2": 16384,  # Latest: Good output capacity with 163K context
    "minimax/minimax-m2": 8192,  # Latest: Standard output limit
    "moonshotai/kimi-k2": 8192,
    # Latest: Thinking model supports larger outputs
    "moonshotai/kimi-k2-thinking": 16384,
    # Qwen3 models may support larger outputs
    "qwen/qwen3-235b-a22b-2507": 16384,
    "qwen/qwen3-235b-a22b-thinking-2507": 16384,
    "qwen/qwen3-next-80b-a3b-instruct": 16384,
    "qwen/qwen3-32b": 8192,
    "qwen/qwen3-30b-a3b": 8192,
}
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # Safe default for most models

# Models that support structured outputs well
MODELS_WITH_EXCELLENT_STRUCTURED_OUTPUTS = {
    # Qwen models
    "qwen/qwen3-max",
    "qwen/qwen-2.5-7b-instruct",  # Latest: Good structured output support
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v3.2",  # Latest: Excellent structured outputs
    "minimax/minimax-m2",  # Latest: Excellent for structured outputs
    "moonshotai/kimi-k2",
    "moonshotai/kimi-k2-thinking",  # Latest: Thinking model with structured outputs
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
}

# HTTP status codes that should be retried
HTTP_STATUS_RETRYABLE = {429, 500, 502, 503, 504}
