"""Configuration management for the sync service."""

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError
from .utils.path_validator import (
    validate_db_path,
    validate_source_dir,
    validate_vault_path,
)


class Config(BaseSettings):
    """Service configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        validate_assignment=True,
    )

    # Required fields
    # Obsidian paths - vault_path can be empty string from env, will be validated
    vault_path: Path | str = Field(default="", description="Path to Obsidian vault")
    source_dir: Path = Field(
        default=Path("."), description="Source directory within vault"
    )

    @field_validator("vault_path", mode="before")
    @classmethod
    def parse_vault_path(cls, v: Any) -> Path:
        """Convert string to Path for vault_path."""
        if isinstance(v, str):
            if not v:
                # Empty string means not set - will be caught by validation
                return Path("")
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        return v

    @field_validator("source_dir", "db_path", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @field_validator("source_subdirs", mode="before")
    @classmethod
    def parse_source_subdirs(cls, v: Any) -> list[Path] | None:
        """Convert source_subdirs to list of Paths."""
        if v is None:
            return None
        if isinstance(v, str):
            return [Path(v)]
        if isinstance(v, list):
            return [Path(str(d)) for d in v]
        return v

    # Anki settings
    anki_connect_url: str = Field(
        default="http://127.0.0.1:8765", description="AnkiConnect URL"
    )
    anki_deck_name: str = Field(
        default="Interview Questions", description="Anki deck name"
    )
    anki_note_type: str = Field(default="APF::Simple", description="Anki note type")

    # Anki model name mapping (internal -> actual Anki model name)
    # Maps internal note type names to actual Anki model names
    # Can be overridden via environment variables or config.yaml
    model_names: dict[str, str] = Field(
        default_factory=lambda: {
            "APF::Simple": os.getenv("ANKI_MODEL_SIMPLE", "APF: Simple (3.0.0)"),
            "APF::Missing (Cloze)": os.getenv(
                "ANKI_MODEL_MISSING", "APF: Missing! (3.0.0)"
            ),
            "APF::Draw": os.getenv("ANKI_MODEL_DRAW", "APF: Draw! (3.0.0)"),
        },
        description="Mapping from internal note type to actual Anki model name",
    )

    # Runtime settings
    run_mode: str = Field(default="apply", description="Run mode: 'apply' or 'dry-run'")
    delete_mode: str = Field(
        default="delete", description="Delete mode: 'delete' or 'archive'"
    )

    # Database
    db_path: Path = Field(
        default=Path(".sync_state.db"), description="Path to sync state database"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    project_log_dir: Path = Field(
        default=Path("./logs"), description="Directory for project-level logs"
    )
    problematic_notes_dir: Path = Field(
        default=Path("./problematic_notes"),
        description="Directory for archiving problematic notes",
    )
    enable_problematic_notes_archival: bool = Field(
        default=True, description="Enable automatic archival of problematic notes"
    )
    error_log_retention_days: int = Field(
        default=90, description="Days to retain error logs"
    )

    # Optional fields (with defaults)
    # Obsidian source directories (optional - overrides source_dir if provided)
    # List of relative paths from vault_path to search for Q&A notes
    # Example: [".", "Interviews", "CS/Algorithms"]
    source_subdirs: list[Path] | None = Field(
        default=None, description="List of source subdirectories"
    )
    # LLM Provider Configuration
    # Unified provider system - choose one: 'ollama', 'lm_studio', 'openrouter'
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 900.0  # 15 minutes default for large models
    # Reasonable default - models have output token limits separate from context window
    llm_max_tokens: int = 8192
    # Enable reasoning mode for models that support it (e.g., DeepSeek)
    llm_reasoning_enabled: bool = False

    # Ollama provider settings (local or cloud)
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None  # Only for Ollama Cloud

    # LM Studio provider settings
    lm_studio_base_url: str = "http://localhost:1234/v1"

    # OpenAI provider settings
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_organization: str | None = None
    openai_max_retries: int = 3

    # Anthropic provider settings
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_version: str = "2023-06-01"
    anthropic_max_retries: int = 3

    # OpenRouter provider settings
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_site_name: str | None = None

    # Legacy OpenRouter settings (for backward compatibility)
    openrouter_model: str = "x-ai/grok-4.1-fast"

    # Deck export settings (for .apkg generation) - optional with defaults
    export_deck_name: str | None = None
    export_deck_description: str = ""
    export_output_path: Path | None = None

    # Agent system settings (optional, defaults provided)
    use_agent_system: bool = False
    # Default to False - validation done by LLM repair instead
    enforce_bilingual_validation: bool = False

    # Imperfect note processing settings
    enable_content_generation: bool = True  # Allow LLM to generate missing content
    repair_missing_sections: bool = True  # Generate missing language sections
    tolerant_parsing: bool = True  # Allow notes with minor issues to proceed
    # Enable content generation in repair agent
    parser_repair_generate_content: bool = True

    # Parser-Repair Agent (reactive - only runs when parsing fails)
    parser_repair_enabled: bool = True
    # Model config moved to unified section below (parser_repair_model, etc.)

    # Note Correction Agent (optional proactive correction before parsing)
    enable_note_correction: bool = Field(
        default=False,
        description="Enable proactive note correction before parsing (optional pre-processing)",
    )
    # Model config: note_correction_model in unified section below

    # Pre-Validator Agent
    pre_validation_enabled: bool = True

    # Post-Validator Agent
    post_validation_max_retries: int = 3
    post_validation_auto_fix: bool = True
    post_validation_strict_mode: bool = True
    # Configurable retry counts per error type (dict mapping error_type to max_retries)
    # Example: {"syntax": 5, "html": 4, "semantic": 2}
    post_validation_retry_config: dict[str, int] = Field(
        default_factory=dict, description="Custom retry counts per error type"
    )

    # ============================================================================
    # Resilience Configuration for Specialized Agents
    # ============================================================================
    # Circuit breaker configuration per agent domain
    # Example: {"yaml_frontmatter": {"failure_threshold": 5, "timeout": 60}, "default": {"failure_threshold": 5, "timeout": 60}}
    circuit_breaker_config: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Circuit breaker configuration per agent domain",
    )

    # Retry configuration for specialized agents
    retry_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_retries": 3,
            "initial_delay": 1.0,
            "backoff_factor": 2.0,
            "jitter": True,
        },
        description="Retry configuration for specialized agents",
    )

    # Confidence threshold for agent results
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for accepting agent repairs",
    )

    # Rate limiting configuration per agent domain (calls per minute)
    # Example: {"yaml_frontmatter": 10, "content_corruption": 5, "default": 20}
    rate_limit_config: dict[str, int] = Field(
        default_factory=dict, description="Rate limiting configuration per agent domain"
    )

    # Bulkhead configuration per agent domain (max concurrent calls)
    # Example: {"yaml_frontmatter": 2, "content_corruption": 1, "default": 3}
    bulkhead_config: dict[str, int] = Field(
        default_factory=dict, description="Bulkhead configuration per agent domain"
    )

    # Metrics storage configuration
    metrics_storage: str = Field(
        default="memory", description="Metrics storage backend: 'memory' or 'database'"
    )

    # Enable adaptive routing
    enable_adaptive_routing: bool = Field(
        default=True,
        description="Enable adaptive routing based on historical performance",
    )

    # Enable learning system
    enable_learning: bool = Field(
        default=True,
        description="Enable failure pattern learning and routing optimization",
    )

    # ============================================================================
    # Agent Memory Configuration
    # ============================================================================
    # Enable agentic memory system
    enable_agent_memory: bool = Field(
        default=True, description="Enable persistent agentic memory for learning"
    )

    # Memory storage path
    memory_storage_path: Path = Field(
        default=Path(".agent_memory"), description="Path to store agent memory data"
    )

    # Memory backend
    memory_backend: str = Field(
        default="chromadb", description="Memory backend: 'chromadb' or 'sqlite'"
    )

    # Enable semantic search
    enable_semantic_search: bool = Field(
        default=True, description="Enable semantic search using embeddings"
    )

    # Embedding model
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for semantic search",
    )

    # Memory retention
    memory_retention_days: int = Field(
        default=90, description="Number of days to retain memories"
    )

    # Maximum memories per type
    max_memories_per_type: int = Field(
        default=10000, description="Maximum number of memories per type"
    )

    # ============================================================================
    # RAG (Retrieval-Augmented Generation) Configuration
    # ============================================================================
    rag_enabled: bool = Field(
        default=False,
        description="Enable RAG for context enrichment, duplicate detection, and few-shot examples",
    )

    rag_db_path: Path = Field(
        default=Path(".chroma_db"),
        description="Path to ChromaDB persistence directory (relative to vault)",
    )

    rag_embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model for RAG (via OpenRouter or direct provider)",
    )

    rag_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum characters per chunk for RAG indexing",
    )

    rag_chunk_overlap: int = Field(
        default=200, ge=0, le=500, description="Overlap between chunks for RAG indexing"
    )

    rag_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve in RAG searches",
    )

    rag_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for duplicate detection",
    )

    rag_index_on_sync: bool = Field(
        default=True, description="Automatically re-index changed files during sync"
    )

    rag_context_enrichment: bool = Field(
        default=True, description="Use RAG to enrich context during card generation"
    )

    rag_duplicate_detection: bool = Field(
        default=True, description="Use RAG for semantic duplicate detection"
    )

    rag_few_shot_examples: bool = Field(
        default=True, description="Use RAG to retrieve few-shot examples for generation"
    )

    # LLM Performance Monitoring
    llm_slow_request_threshold: float = Field(
        default=60.0, description="Threshold in seconds for logging slow LLM requests"
    )

    # LangGraph + PydanticAI Agent System (new!)
    use_langgraph: bool = False  # Enable LangGraph-based orchestration
    use_pydantic_ai: bool = False  # Enable PydanticAI for structured outputs

    # ============================================================================
    # Unified Model Configuration System
    # ============================================================================
    # Model preset: "cost_effective", "balanced", "high_quality", or "fast"
    # This sets optimized models for all tasks automatically
    model_preset: str = "balanced"

    # Default model used by ALL agents unless specifically overridden
    # Only used if model_preset is not set or for backward compatibility
    default_llm_model: str = "x-ai/grok-4.1-fast"

    # Per-agent model overrides (optional - overrides preset defaults)
    # Set model to empty string ("") to use preset default
    # Set temperature/max_tokens to None to use preset default

    # QA Extraction
    qa_extractor_model: str = ""
    qa_extractor_temperature: float | None = None
    qa_extractor_max_tokens: int | None = None
    # Parser Repair (reactive - only runs when parsing fails)
    parser_repair_model: str = ""
    parser_repair_temperature: float | None = None
    parser_repair_max_tokens: int | None = None
    # Note Correction (proactive - runs before parsing if enabled)
    note_correction_model: str = ""
    note_correction_temperature: float | None = None
    note_correction_max_tokens: int | None = None
    # Pre-Validation
    pre_validator_model: str = ""
    pre_validator_temperature: float | None = None
    pre_validator_max_tokens: int | None = None
    # Generation
    generator_model: str = ""
    generator_temperature: float | None = None
    generator_max_tokens: int | None = None
    # Post-Validation
    post_validator_model: str = ""
    post_validator_temperature: float | None = None
    post_validator_max_tokens: int | None = None
    # Context Enrichment
    context_enrichment_model: str = ""
    # Memorization Quality
    memorization_quality_model: str = ""
    # Card Splitting
    card_splitting_model: str = ""
    # Split Validation
    split_validator_model: str = ""
    # Duplicate Detection
    duplicate_detection_model: str = ""

    # LangGraph Workflow Configuration
    langgraph_max_retries: int = 3
    langgraph_auto_fix: bool = True
    langgraph_strict_mode: bool = True
    langgraph_checkpoint_enabled: bool = True  # Enable state persistence
    langgraph_max_steps: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum steps allowed in LangGraph workflow (prevents infinite loops)",
    )

    # ============================================================================
    # Chain of Thought (CoT) Reasoning Configuration
    # ============================================================================
    enable_cot_reasoning: bool = Field(
        default=False,
        description="Enable Chain of Thought reasoning nodes before action nodes",
    )
    store_reasoning_traces: bool = Field(
        default=True,
        description="Store reasoning traces in pipeline state for inspection",
    )
    log_reasoning_traces: bool = Field(
        default=False,
        description="Log reasoning traces to logger (can be verbose)",
    )
    reasoning_model: str = Field(
        default="",
        description="Model for reasoning nodes (uses generator_model if empty)",
    )
    reasoning_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for reasoning model (lower = more deterministic)",
    )
    cot_enabled_stages: list[str] = Field(
        default_factory=lambda: [
            "pre_validation",
            "generation",
            "post_validation",
        ],
        description="Stages where CoT reasoning is applied",
    )

    # ============================================================================
    # Self-Reflection Configuration
    # ============================================================================
    enable_self_reflection: bool = Field(
        default=False,
        description="Enable self-reflection after action nodes to evaluate and revise outputs",
    )
    store_reflection_traces: bool = Field(
        default=True,
        description="Store reflection traces in pipeline state for inspection",
    )
    log_reflection_traces: bool = Field(
        default=False,
        description="Log reflection traces to logger (can be verbose)",
    )
    reflection_model: str = Field(
        default="",
        description="Model for reflection nodes (uses generator_model if empty)",
    )
    reflection_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Temperature for reflection model",
    )
    max_revisions: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum revision passes per stage (0 disables auto-revision)",
    )
    reflection_enabled_stages: list[str] = Field(
        default_factory=lambda: [
            "generation",
            "context_enrichment",
        ],
        description="Stages where self-reflection is applied (after validation passes)",
    )

    # Smart reflection skipping - skip reflection for simple content
    reflection_skip_qa_threshold: int = Field(
        default=2,
        ge=0,
        description="Skip reflection if Q/A pair count <= this value",
    )
    reflection_skip_content_length: int = Field(
        default=500,
        ge=0,
        description="Skip reflection if content length < this value (chars)",
    )
    reflection_skip_confidence_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Skip reflection if validation confidence >= this value",
    )

    # ============================================================================
    # Unified Agent Framework Configuration
    # ============================================================================

    # Primary agent framework selection
    agent_framework: str = Field(
        default="pydantic_ai",
        description="Primary agent framework: 'pydantic_ai' or 'langchain'",
    )

    @field_validator("agent_framework")
    @classmethod
    def validate_agent_framework(cls, v: str) -> str:
        """Validate agent framework selection."""
        valid_frameworks = ["pydantic_ai", "langchain"]
        if v not in valid_frameworks:
            raise ValueError(
                f"agent_framework must be one of {valid_frameworks}, got '{v}'"
            )
        return v

    # LangChain Agent Type Configuration
    # Specify which LangChain agent type to use for each task
    langchain_generator_type: str = Field(
        default="tool_calling",
        description="LangChain agent type for card generation: 'tool_calling', 'react', 'structured_chat', 'json_chat'",
    )
    langchain_pre_validator_type: str = Field(
        default="react",
        description="LangChain agent type for pre-validation: 'tool_calling', 'react', 'structured_chat', 'json_chat'",
    )
    langchain_post_validator_type: str = Field(
        default="tool_calling",
        description="LangChain agent type for post-validation: 'tool_calling', 'react', 'structured_chat', 'json_chat'",
    )
    langchain_enrichment_type: str = Field(
        default="structured_chat",
        description="LangChain agent type for context enrichment: 'tool_calling', 'react', 'structured_chat', 'json_chat'",
    )

    @field_validator(
        "langchain_generator_type",
        "langchain_pre_validator_type",
        "langchain_post_validator_type",
        "langchain_enrichment_type",
    )
    @classmethod
    def validate_langchain_agent_type(cls, v: str) -> str:
        """Validate LangChain agent type selection."""
        valid_types = ["tool_calling", "react", "structured_chat", "json_chat"]
        if v not in valid_types:
            raise ValueError(
                f"LangChain agent type must be one of {valid_types}, got '{v}'"
            )
        return v

    # Agent Framework Fallback Configuration
    agent_fallback_on_error: str = Field(
        default="pydantic_ai",
        description="Fallback agent framework when primary framework fails",
    )
    agent_fallback_on_timeout: str = Field(
        default="react", description="Fallback agent type when primary agent times out"
    )

    # Enhancement Agents (optional quality improvements)
    enable_card_splitting: bool = True  # Analyze if note should be split

    # Card splitting preferences
    card_splitting_preferred_size: Literal["small", "medium", "large"] = Field(
        default="medium",
        description="Preferred card size: small (more splits), medium (balanced), large (fewer splits)",
    )
    card_splitting_prefer_splitting: bool = Field(
        default=True,
        description="Prefer splitting complex notes into multiple cards",
    )
    card_splitting_min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold to apply split decision",
    )
    card_splitting_max_cards_per_note: int = Field(
        default=10,
        ge=1,
        description="Maximum number of cards to generate from a single note (safety limit)",
    )
    enable_context_enrichment: bool = True  # Add examples and mnemonics
    enable_memorization_quality: bool = True  # Check SRS effectiveness
    enable_duplicate_detection: bool = (
        False  # Check against existing cards (requires existing_cards)
    )

    # Performance Optimization Settings
    enable_batch_operations: bool = True  # Enable batch Anki and DB operations
    batch_size: int = 50  # Cards per batch for Anki and DB operations
    max_concurrent_generations: int = 5  # Max parallel card generations
    auto_adjust_workers: bool = Field(
        default=False,
        description="Automatically adjust worker count based on system resources",
    )
    retry_config_parallel: dict[str, Any] = Field(
        default_factory=dict,
        description="Retry configuration for parallel tasks (max_attempts, wait_exponential, etc.)",
    )
    index_use_llm_extraction: bool = False  # Use LLM extraction during indexing
    verify_card_creation: bool = True  # Verify cards exist in Anki after creation

    # Memory Management Settings
    # Maximum note content size to keep in memory (MB)
    max_note_content_size_mb: float = 50.0
    # Enable aggressive memory cleanup after processing
    enable_memory_cleanup: bool = True

    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model name for a specific agent.

        Uses new model configuration system with presets if available,
        falls back to legacy configuration.

        Args:
            agent_type: Agent type (e.g., "pre_validator", "generator", "context_enrichment")

        Returns:
            Model name
        """
        from .models.config import ModelPreset, ModelTask, get_model_for_task

        # Map agent types to ModelTask
        agent_to_task = {
            "pre_validator": ModelTask.PRE_VALIDATION,
            "generator": ModelTask.GENERATION,
            "post_validator": ModelTask.POST_VALIDATION,
            "context_enrichment": ModelTask.CONTEXT_ENRICHMENT,
            "memorization_quality": ModelTask.MEMORIZATION_QUALITY,
            "card_splitting": ModelTask.CARD_SPLITTING,
            "split_validator": ModelTask.CARD_SPLITTING,  # Reuse card splitting task
            "duplicate_detection": ModelTask.DUPLICATE_DETECTION,
            "qa_extractor": ModelTask.QA_EXTRACTION,
            "parser_repair": ModelTask.PARSER_REPAIR,
            "note_correction": ModelTask.PARSER_REPAIR,  # Reuse parser repair task
            "reasoning": ModelTask.GENERATION,  # CoT reasoning uses generation task
            # Self-reflection uses post-validation task
            "reflection": ModelTask.POST_VALIDATION,
        }

        task = agent_to_task.get(agent_type)

        # Check for explicit override first
        # Maps agent type to the override field (empty string = use preset)
        agent_model_map = {
            "pre_validator": self.pre_validator_model,
            "generator": self.generator_model,
            "post_validator": self.post_validator_model,
            "context_enrichment": self.context_enrichment_model,
            "memorization_quality": self.memorization_quality_model,
            "card_splitting": self.card_splitting_model,
            "split_validator": self.split_validator_model,
            "duplicate_detection": self.duplicate_detection_model,
            "qa_extractor": self.qa_extractor_model,
            "parser_repair": self.parser_repair_model,
            "note_correction": self.note_correction_model or self.parser_repair_model,
            "reasoning": self.reasoning_model or self.generator_model,  # CoT reasoning
            "reflection": self.reflection_model
            or self.generator_model,  # Self-reflection
        }

        explicit_model = agent_model_map.get(agent_type, "")
        if explicit_model:
            return explicit_model

        # Use preset system if task is known
        if task:
            try:
                preset = ModelPreset(self.model_preset.lower())
                return get_model_for_task(task, preset)
            except (ValueError, AttributeError):
                # Invalid preset, fall back to default_llm_model
                pass

        # Fallback to default_llm_model
        return self.default_llm_model

    def get_model_config_for_task(self, task: str) -> dict[str, Any]:
        """Get full model configuration for a task including temperature, max_tokens, etc.

        Args:
            task: Task name (e.g., "qa_extraction", "generation")

        Returns:
            Dictionary with model configuration
        """
        from .models.config import ModelPreset, ModelTask, get_model_config

        try:
            model_task = ModelTask(task.lower())
        except ValueError:
            # Unknown task, return minimal config
            return {
                "model_name": self.default_llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens,
            }

        # Get preset
        try:
            preset = ModelPreset(self.model_preset.lower())
        except (ValueError, AttributeError):
            preset = ModelPreset.BALANCED

        # Build overrides from config
        overrides: dict[str, Any] = {}

        # Task-specific overrides
        if task == "qa_extraction":
            if self.qa_extractor_temperature is not None:
                overrides["temperature"] = self.qa_extractor_temperature
            if self.qa_extractor_max_tokens is not None:
                overrides["max_tokens"] = self.qa_extractor_max_tokens
        elif task == "generation":
            if self.generator_temperature is not None:
                overrides["temperature"] = self.generator_temperature
            if self.generator_max_tokens is not None:
                overrides["max_tokens"] = self.generator_max_tokens
        elif task == "pre_validation":
            if self.pre_validator_temperature is not None:
                overrides["temperature"] = self.pre_validator_temperature
            if self.pre_validator_max_tokens is not None:
                overrides["max_tokens"] = self.pre_validator_max_tokens
        elif task == "post_validation":
            if self.post_validator_temperature is not None:
                overrides["temperature"] = self.post_validator_temperature
            if self.post_validator_max_tokens is not None:
                overrides["max_tokens"] = self.post_validator_max_tokens
        elif task == "parser_repair":
            if self.parser_repair_temperature is not None:
                overrides["temperature"] = self.parser_repair_temperature
            if self.parser_repair_max_tokens is not None:
                overrides["max_tokens"] = self.parser_repair_max_tokens

        # Get model config from preset
        config = get_model_config(model_task, preset, overrides if overrides else None)

        # Override model name if explicitly set
        explicit_model = self.get_model_for_agent(task)
        if explicit_model and explicit_model != config.model_name:
            config.model_name = explicit_model
            config.capabilities = None  # Will be recalculated

        return {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens or self.llm_max_tokens,
            "top_p": config.top_p or self.llm_top_p,
            "reasoning_enabled": config.reasoning_enabled or self.llm_reasoning_enabled,
        }

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        """Validate configuration values after initialization.

        Note: This runs during initialization, before the model is frozen.
        Since frozen=True is set, we use object.__setattr__ to update validated paths.
        """
        vault_path = self.vault_path
        if isinstance(vault_path, str):
            if not vault_path:
                raise ConfigurationError(
                    "vault_path is required",
                    suggestion="Set VAULT_PATH environment variable or vault_path in config.yaml",
                )
            vault_path = Path(vault_path).expanduser().resolve()

        validated_vault = validate_vault_path(vault_path, allow_symlinks=False)
        _ = validate_source_dir(validated_vault, self.source_dir)
        validated_db = validate_db_path(self.db_path, vault_path=validated_vault)

        parent_dir = validated_db.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ConfigurationError(
                    f"Cannot create database directory: {parent_dir}",
                    suggestion=f"Ensure you have write permissions to {parent_dir.parent}. Error: {e}",
                )

        if not os.access(parent_dir, os.W_OK):
            raise ConfigurationError(
                f"Database directory is not writable: {parent_dir}",
                suggestion=f"Check directory permissions: chmod 755 {parent_dir}",
            )

        if validated_db.exists():
            if not os.access(validated_db, os.R_OK):
                raise ConfigurationError(
                    f"Database file exists but is not readable: {validated_db}",
                    suggestion=f"Check file permissions: chmod 644 {validated_db}",
                )
            if not os.access(validated_db, os.W_OK):
                raise ConfigurationError(
                    f"Database file exists but is not writable: {validated_db}",
                    suggestion=f"Check file permissions: chmod 644 {validated_db}",
                )

        valid_providers = [
            "ollama",
            "lm_studio",
            "lmstudio",
            "openrouter",
            "openai",
            "anthropic",
            "claude",
        ]
        if self.llm_provider.lower() not in valid_providers:
            raise ConfigurationError(
                f"Invalid llm_provider: {self.llm_provider}. "
                f"Must be one of: {', '.join(valid_providers)}",
                suggestion=f"Set llm_provider to one of: {', '.join(valid_providers)}",
            )

        provider_lower = self.llm_provider.lower()

        if provider_lower == "openrouter" and not self.openrouter_api_key:
            raise ConfigurationError(
                "OpenRouter API key is required when using OpenRouter provider.",
                suggestion="Set OPENROUTER_API_KEY environment variable or openrouter_api_key in config.yaml",
            )

        if provider_lower == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required when using OpenAI provider.",
                suggestion="Set OPENAI_API_KEY environment variable or openai_api_key in config.yaml. "
                "Get your API key from https://platform.openai.com/api-keys",
            )

        if provider_lower in ("anthropic", "claude") and not self.anthropic_api_key:
            raise ConfigurationError(
                "Anthropic API key is required when using Anthropic/Claude provider.",
                suggestion="Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config.yaml. "
                "Get your API key from https://console.anthropic.com/",
            )

        if self.run_mode not in ("apply", "dry-run"):
            raise ConfigurationError(
                f"Invalid run_mode: {self.run_mode}",
                suggestion="Set run_mode to either 'apply' or 'dry-run'",
            )

        if self.delete_mode not in ("delete", "archive"):
            raise ConfigurationError(
                f"Invalid delete_mode: {self.delete_mode}",
                suggestion="Set delete_mode to either 'delete' or 'archive'",
            )

        if not (0 <= self.llm_temperature <= 1):
            raise ConfigurationError(
                f"LLM temperature must be 0-1: {self.llm_temperature}",
                suggestion="Set llm_temperature to a value between 0.0 and 1.0",
            )

        if not (0 <= self.llm_top_p <= 1):
            raise ConfigurationError(
                f"LLM top_p must be 0-1: {self.llm_top_p}",
                suggestion="Set llm_top_p to a value between 0.0 and 1.0",
            )

        if validated_vault != self.vault_path:
            object.__setattr__(self, "vault_path", validated_vault)
        if validated_db != self.db_path:
            object.__setattr__(self, "db_path", validated_db)

        return self


_config: Config | None = None


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from .env and config.yaml files using pydantic-settings."""
    import yaml

    # Find config.yaml file
    candidate_paths: list[Path] = []

    if config_path:
        candidate_paths.append(config_path.expanduser())
    else:
        env_path = os.getenv("OBSIDIAN_ANKI_CONFIG")
        if env_path:
            candidate_paths.append(Path(env_path).expanduser())
        candidate_paths.append(Path.cwd() / "config.yaml")
        default_repo_config = Path(__file__).resolve().parents[2] / "config.yaml"
        candidate_paths.append(default_repo_config)

    resolved_config_path: Path | None = None
    for candidate in candidate_paths:
        if candidate.exists():
            resolved_config_path = candidate
            break

    # Load YAML data if file exists
    yaml_data: dict[str, Any] = {}
    if resolved_config_path:
        try:
            with open(resolved_config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            from ..utils.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "yaml_config_load_failed", path=str(resolved_config_path), error=str(e)
            )

    # Convert YAML data to environment variable format for pydantic-settings
    # pydantic-settings will automatically load from .env file via model_config
    # We'll merge YAML data by setting environment variables temporarily
    import contextlib

    @contextlib.contextmanager
    def yaml_as_env():
        """Temporarily set environment variables from YAML data."""
        # Store original env values
        original_env: dict[str, str] = {}
        try:
            # Set env vars from YAML (YAML takes precedence over .env)
            for key, value in yaml_data.items():
                env_key = key.upper()
                original_env[env_key] = os.environ.get(env_key, "")
                if value is not None:
                    # Handle special types
                    if isinstance(value, (list, dict)):
                        # For complex types, we'll handle them in the Config model
                        continue
                    elif isinstance(value, Path):
                        os.environ[env_key] = str(value)
                    else:
                        os.environ[env_key] = str(value)
            yield
        finally:
            # Restore original env values
            for key, value in original_env.items():
                if value:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    # Load config using pydantic-settings
    # Handle special cases for Path and list types from YAML
    with yaml_as_env():
        # Process source_subdirs from YAML if present
        source_subdirs: list[Path] | None = None
        if "source_subdirs" in yaml_data:
            source_subdirs_raw = yaml_data["source_subdirs"]
            if isinstance(source_subdirs_raw, list):
                source_subdirs = [Path(str(d)) for d in source_subdirs_raw]
            elif isinstance(source_subdirs_raw, str):
                source_subdirs = [Path(source_subdirs_raw)]

        # Process export_output_path from YAML if present
        export_output_path: Path | None = None
        if "export_output_path" in yaml_data:
            export_output_str = yaml_data["export_output_path"]
            if export_output_str:
                export_output_path = Path(str(export_output_str))

        # Create config instance - pydantic-settings will load from env vars
        # We pass YAML-specific values directly
        config_kwargs: dict[str, Any] = {}
        if source_subdirs is not None:
            config_kwargs["source_subdirs"] = source_subdirs
        if export_output_path is not None:
            config_kwargs["export_output_path"] = export_output_path
        if "vault_path" in yaml_data:
            config_kwargs["vault_path"] = (
                Path(str(yaml_data["vault_path"])).expanduser().resolve()
            )
        if "source_dir" in yaml_data:
            config_kwargs["source_dir"] = Path(str(yaml_data["source_dir"]))
        if "db_path" in yaml_data:
            config_kwargs["db_path"] = Path(str(yaml_data["db_path"]))

        config = Config(**config_kwargs)

    return config


def get_config() -> Config:
    """Get singleton config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set singleton config instance (for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global config instance (for testing only).

    Warning:
        This should only be used in tests. In production, config
        should be loaded once and never changed.
    """
    global _config
    _config = None
