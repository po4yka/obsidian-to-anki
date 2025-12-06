"""Settings model for the sync service (split from config.py)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config_models import (
    BulkheadDomainConfig,
    CircuitBreakerDomainConfig,
    RateLimitDomainConfig,
    RetryConfig,
)
from .exceptions import ConfigurationError
from .utils.path_validator import (
    validate_db_path,
    validate_source_dir,
    validate_vault_path,
)
from .utils.reasoning import normalize_reasoning_effort


class Config(BaseSettings):
    """Service configuration using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    # Required fields
    # Obsidian paths - vault_path can be empty string from env, will be validated
    vault_path: Path | str = Field(default="", description="Path to Obsidian vault")
    source_dir: Path = Field(
        default=Path(), description="Source directory within vault"
    )

    # Data storage directory - all processing data stored here (not in vault)
    # Default: repo root directory (where config.yaml lives)
    data_dir: Path = Field(
        default=Path(),
        description=(
            "Directory for all data files (chroma_db, logs, cache, sync_state). "
            "Keeps vault clean of processing artifacts."
        ),
    )

    @field_validator("vault_path", mode="before")
    @classmethod
    def parse_vault_path(cls, v: Any) -> Path:
        """Convert string to Path for vault_path."""
        if isinstance(v, str):
            if not v:
                # Empty string means not set - will be caught by validation
                return Path()
            return Path(v).expanduser().resolve()
        if isinstance(v, Path):
            return v.expanduser().resolve()
        if v is None:
            return Path()
        msg = f"vault_path must be string or Path, got {type(v).__name__}"
        raise ValueError(msg)

    @field_validator("source_dir", "db_path", "data_dir", mode="before")
    @classmethod
    def parse_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v
        if v is None:
            return Path()
        msg = f"Path field must be string or Path, got {type(v).__name__}"
        raise ValueError(msg)

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
        msg = f"source_subdirs must be string, list, or None, got {type(v).__name__}"
        raise ValueError(msg)

    # Anki settings
    anki_connect_url: str = Field(
        default="http://127.0.0.1:8765", description="AnkiConnect URL"
    )
    anki_deck_name: str = Field(
        default="Interview Questions", description="Anki deck name"
    )
    anki_note_type: str = Field(default="APF::Simple", description="Anki note type")

    # Anki model name mapping (internal -> actual Anki model name)
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

    # Database (relative to data_dir)
    db_path: Path = Field(
        default=Path(".sync_state.db"),
        description="Path to sync state database (relative to data_dir)",
    )

    # Logging (relative to data_dir)
    log_level: str = Field(default="INFO", description="Log level")
    project_log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for project-level logs (relative to data_dir)",
    )
    problematic_notes_dir: Path = Field(
        default=Path("problematic_notes"),
        description="Directory for archiving problematic notes (relative to data_dir)",
    )
    enable_problematic_notes_archival: bool = Field(
        default=True, description="Enable automatic archival of problematic notes"
    )
    error_log_retention_days: int = Field(
        default=90, description="Days to retain error logs"
    )
    compress_error_logs: bool = Field(
        default=True,
        description="Compress rotated error logs within retention window",
    )
    archiver_batch_size: int = Field(
        default=64,
        ge=1,
        description="Number of deferred archives to flush per batch",
    )
    archiver_min_fd_headroom: int = Field(
        default=32,
        ge=1,
        description="Minimum file descriptor headroom required before flushing another batch",
    )
    archiver_fd_poll_interval: float = Field(
        default=0.05,
        ge=0.01,
        description="Delay (seconds) between FD headroom checks when deferred archiving pauses",
    )

    # Queue Configuration
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for task queue",
    )
    enable_queue: bool = Field(
        default=False,
        description="Enable Redis-backed task queue for card generation",
    )
    queue_max_retries: int = Field(
        default=3,
        description="Maximum retries for queued tasks",
    )
    redis_socket_connect_timeout: float = Field(
        default=5.0,
        description="Redis socket connect timeout in seconds",
    )

    # Queue Stability Configuration
    queue_max_wait_time_seconds: int = Field(
        default=18000,
        description="Overall timeout for queue polling (5 hours)",
    )
    queue_job_timeout_seconds: int = Field(
        default=10800,
        description="Per-job timeout (3 hours)",
    )
    queue_poll_interval: float = Field(
        default=0.5,
        description="Initial poll interval for job status (seconds)",
    )
    queue_poll_max_interval: float = Field(
        default=5.0,
        description="Maximum poll interval with adaptive backoff (seconds)",
    )
    queue_circuit_breaker_threshold: int = Field(
        default=3,
        description="Consecutive failures before circuit breaker opens",
    )
    queue_circuit_breaker_timeout: int = Field(
        default=60,
        description="Seconds to wait before retrying after circuit breaker opens",
    )
    result_queue_ttl_seconds: int = Field(
        default=3600,
        description="TTL for per-run result queues (seconds)",
    )
    result_dead_letter_ttl_seconds: int = Field(
        default=3600,
        description="TTL for dead-letter queues when result push fails (seconds)",
    )
    result_dead_letter_max_length: int = Field(
        default=100,
        description="Maximum retained entries in dead-letter queues",
    )
    worker_generation_timeout_seconds: float = Field(
        default=2700.0,
        ge=60.0,
        description="SLA (seconds) for generation stage before worker flags a timeout",
    )
    worker_validation_timeout_seconds: float = Field(
        default=2700.0,
        ge=30.0,
        description="SLA (seconds) for post-validation stage before worker flags a timeout",
    )

    # Optional fields (with defaults)
    source_subdirs: list[Path] | None = Field(
        default=None, description="List of source subdirectories"
    )
    # LLM Provider Configuration
    llm_provider: str = "ollama"

    # Common LLM settings
    llm_temperature: float = 0.2
    llm_top_p: float = 0.3
    llm_timeout: float = 3600.0  # 60 minutes default for large models
    llm_max_tokens: int = 8192
    llm_streaming_enabled: bool = Field(
        default=False,
        description="Enable SSE streaming for providers that support it",
    )
    llm_reasoning_enabled: bool = False
    llm_reasoning_effort: str = Field(
        default="auto",
        description=(
            "Reasoning effort for providers that support it "
            "(auto|minimal|low|medium|high|none)"
        ),
    )
    reasoning_effort_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Per-agent reasoning effort overrides (e.g., {'generation': 'high'})",
    )

    @field_validator("llm_reasoning_effort", mode="before")
    @classmethod
    def _normalize_llm_reasoning_effort(cls, value: str | None) -> str:
        return normalize_reasoning_effort(value)

    @field_validator("reasoning_effort_overrides", mode="before")
    @classmethod
    def _normalize_reasoning_effort_overrides(
        cls, value: dict[str, str] | None
    ) -> dict[str, str]:
        if not value:
            return {}
        return {k.lower(): normalize_reasoning_effort(v) for k, v in value.items()}

    # Ollama provider settings (local or cloud)
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str | None = None  # Only for Ollama Cloud

    # LM Studio provider settings
    lm_studio_base_url: str = "http://localhost:1234/v1"

    # OpenRouter provider settings
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_site_name: str | None = None

    # Fallback model for when primary model fails (e.g., empty completions)
    fallback_llm_model: str = Field(
        default="qwen/qwen3-max",
        description="Fallback model to use when primary model fails with empty completions",
    )

    # Stability Configuration
    generation_timeout_seconds: float = Field(
        default=300.0,
        ge=30.0,
        description="Timeout for card generation stage in seconds. "
        "Prevents indefinite hangs on LLM calls.",
    )

    detect_orphans_on_sync: bool = Field(
        default=True,
        description="Check for orphaned cards at end of each sync. "
        "Orphans are cards in Anki without DB records or vice versa.",
    )

    log_state_transitions: bool = Field(
        default=True,
        description="Log pipeline state transitions for debugging. "
        "Useful for tracing card generation failures.",
    )

    include_retry_stats_in_summary: bool = Field(
        default=True,
        description="Include retry statistics in sync summary log.",
    )

    # APF Generation retry configuration
    generation_max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Max retries for failed APF generation with sentinel validation.",
    )

    generation_retry_prompt_mode: str = Field(
        default="constrained",
        description="Prompt mode for retries: 'constrained' or 'verbose'.",
    )

    # Deck export settings (for .apkg generation)
    export_deck_name: str | None = None
    export_deck_description: str = ""
    export_output_path: Path | None = None

    # Agent system settings
    enforce_bilingual_validation: bool = False
    enable_content_generation: bool = True
    repair_missing_sections: bool = True
    tolerant_parsing: bool = True
    parser_repair_generate_content: bool = True
    parser_repair_enabled: bool = True
    enable_note_correction: bool = Field(
        default=False,
        description="Enable proactive note correction before parsing (optional pre-processing)",
    )
    pre_validation_enabled: bool = True
    post_validation_max_retries: int = 3
    post_validation_auto_fix: bool = True
    post_validation_strict_mode: bool = True
    post_validation_retry_config: dict[str, int] = Field(
        default_factory=dict, description="Custom retry counts per error type"
    )

    # Resilience Configuration for Specialized Agents
    circuit_breaker_config: dict[str, CircuitBreakerDomainConfig] = Field(
        default_factory=dict,
        description="Circuit breaker configuration per agent domain",
    )
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration for specialized agents",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for accepting agent repairs",
    )
    rate_limit_config: dict[str, RateLimitDomainConfig] = Field(
        default_factory=dict, description="Rate limiting configuration per agent domain"
    )
    bulkhead_config: dict[str, BulkheadDomainConfig] = Field(
        default_factory=dict, description="Bulkhead configuration per agent domain"
    )
    metrics_storage: str = Field(
        default="memory", description="Metrics storage backend: 'memory' or 'database'"
    )
    enable_adaptive_routing: bool = Field(
        default=True,
        description="Enable adaptive routing based on historical performance",
    )
    enable_learning: bool = Field(
        default=True,
        description="Enable failure pattern learning and routing optimization",
    )

    # Agent Memory Configuration
    enable_agent_memory: bool = Field(
        default=True, description="Enable persistent agentic memory for learning"
    )
    memory_storage_path: Path = Field(
        default=Path(".agent_memory"),
        description="Path to store agent memory data (relative to data_dir)",
    )
    memory_backend: str = Field(
        default="chromadb", description="Memory backend: 'chromadb' or 'sqlite'"
    )
    enable_semantic_search: bool = Field(
        default=True, description="Enable semantic search using embeddings"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for semantic search",
    )
    memory_retention_days: int = Field(
        default=90, description="Number of days to retain memories"
    )
    max_memories_per_type: int = Field(
        default=10000, description="Maximum number of memories per type"
    )
    max_agent_memory_size_mb: int = Field(
        default=500,
        description="Maximum size of agent memory directory in MB (0 = unlimited)",
    )

    # RAG (Retrieval-Augmented Generation) Configuration
    rag_enabled: bool = Field(
        default=False,
        description="Enable RAG for context enrichment, duplicate detection, and few-shot examples",
    )
    rag_db_path: Path = Field(
        default=Path(".chroma_db"),
        description="Path to ChromaDB persistence directory (relative to data_dir)",
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

    # Unified Model Configuration System
    model_preset: str = "balanced"
    default_llm_model: str = "deepseek/deepseek-v3.2"
    model_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Model overrides per task. Format: "
            "{'task_name': {'model_name': '...', 'temperature': 0.3, ...}}"
        ),
    )
    post_validator_timeout_seconds: float = Field(
        default=2700.0,
        ge=5.0,
        description="Per-attempt timeout for post-validation agent calls (seconds)",
    )
    post_validator_retry_backoff_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Base delay (seconds) before retrying post-validation",
    )
    post_validator_retry_jitter_seconds: float = Field(
        default=1.5,
        ge=0.0,
        description="Maximum jitter (seconds) added to the backoff delay",
    )

    # LangGraph Workflow Configuration
    langgraph_max_retries: int = 3
    langgraph_auto_fix: bool = True
    langgraph_strict_mode: bool = True
    langgraph_max_steps: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum steps allowed in LangGraph workflow (prevents infinite loops)",
    )

    # Auto-Fix Configuration
    autofix_write_back: bool = Field(
        default=False,
        description="Write auto-fixes back to source files (modifies original notes)",
    )
    autofix_handlers: list[str] | None = Field(
        default=None,
        description=(
            "List of enabled auto-fix handler types (None = all handlers enabled). "
            "Available: trailing_whitespace, empty_references, title_format, "
            "moc_mismatch, section_order, missing_related_questions, "
            "broken_wikilink, broken_related_entry"
        ),
    )

    # Chain of Thought (CoT) Reasoning Configuration
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
    cot_enabled_stages: list[str] = Field(
        default_factory=lambda: [
            "pre_validation",
            "generation",
            "post_validation",
        ],
        description="Stages where CoT reasoning is applied",
    )

    # Self-Reflection Configuration
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

    # Unified Agent Framework Configuration
    agent_framework: str = Field(
        default="pydantic_ai",
        description="Primary agent framework: 'pydantic_ai' or 'memory_enhanced'",
    )

    @field_validator("agent_framework")
    @classmethod
    def validate_agent_framework(cls, v: str) -> str:
        """Validate agent framework selection."""
        valid_frameworks = ["pydantic_ai", "memory_enhanced"]
        if v not in valid_frameworks:
            msg = f"agent_framework must be one of {valid_frameworks}, got '{v}'"
            raise ValueError(msg)
        return v

    # Enhancement Agents
    enable_card_splitting: bool = True
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
    enable_context_enrichment: bool = True
    enable_memorization_quality: bool = True
    enable_duplicate_detection: bool = False
    enable_highlight_agent: bool = Field(
        default=True,
        description="Run highlight agent to extract candidate Q&A when validation fails",
    )
    highlight_max_candidates: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of QA candidates suggested by highlight agent",
    )

    # Performance Optimization Settings
    enable_batch_operations: bool = True
    batch_size: int = 50
    max_concurrent_generations: int = 5
    auto_adjust_workers: bool = Field(
        default=False,
        description="Automatically adjust worker count based on system resources",
    )
    retry_config_parallel: dict[str, Any] = Field(
        default_factory=dict,
        description="Retry configuration for parallel tasks (max_attempts, wait_exponential, etc.)",
    )
    index_use_llm_extraction: bool = False
    verify_card_creation: bool = True

    # Memory Management Settings
    max_note_content_size_mb: float = 50.0
    enable_memory_cleanup: bool = True

    # Fail-Fast Configuration
    strict_mode: bool = Field(
        default=True,
        description=(
            "Enable strict validation and fail-fast behavior. "
            "When True, errors halt execution immediately. "
            "Set to False for lenient mode."
        ),
    )
    fail_on_card_error: bool = Field(
        default=True,
        description=(
            "Stop sync on first card operation error. "
            "Set to False to continue processing remaining cards on error."
        ),
    )
    verify_connectivity_at_startup: bool = Field(
        default=True,
        description=(
            "Verify external services (Anki, LLM provider) connectivity at startup. "
            "Catches configuration issues early before processing begins."
        ),
    )

    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the model name for a specific agent."""
        from .models.config import ModelPreset, ModelTask, get_model_for_task

        agent_to_task = {
            "pre_validator": ModelTask.PRE_VALIDATION,
            "generator": ModelTask.GENERATION,
            "post_validator": ModelTask.POST_VALIDATION,
            "context_enrichment": ModelTask.CONTEXT_ENRICHMENT,
            "memorization_quality": ModelTask.MEMORIZATION_QUALITY,
            "card_splitting": ModelTask.CARD_SPLITTING,
            "split_validator": ModelTask.CARD_SPLITTING,
            "duplicate_detection": ModelTask.DUPLICATE_DETECTION,
            "qa_extractor": ModelTask.QA_EXTRACTION,
            "parser_repair": ModelTask.PARSER_REPAIR,
            "note_correction": ModelTask.PARSER_REPAIR,
            "reasoning": ModelTask.GENERATION,
            "reflection": ModelTask.POST_VALIDATION,
            "highlight": ModelTask.HIGHLIGHT,
        }

        task = agent_to_task.get(agent_type)

        agent_to_task_name = {
            "pre_validator": "pre_validation",
            "generator": "generation",
            "post_validator": "post_validation",
            "context_enrichment": "context_enrichment",
            "memorization_quality": "memorization_quality",
            "card_splitting": "card_splitting",
            "split_validator": "card_splitting",
            "duplicate_detection": "duplicate_detection",
            "qa_extractor": "qa_extraction",
            "parser_repair": "parser_repair",
            "note_correction": "parser_repair",
            "reasoning": "reasoning",
            "reflection": "reflection",
            "highlight": "highlight",
        }

        task_name = agent_to_task_name.get(agent_type)

        if task_name and task_name in self.model_overrides:
            override = self.model_overrides[task_name]
            if override.get("model_name"):
                return override["model_name"]

        if agent_type == "note_correction" and "parser_repair" in self.model_overrides:
            override = self.model_overrides["parser_repair"]
            if override.get("model_name"):
                return override["model_name"]
        if agent_type == "reasoning" and "generation" in self.model_overrides:
            override = self.model_overrides["generation"]
            if override.get("model_name"):
                return override["model_name"]
        if agent_type == "reflection" and "generation" in self.model_overrides:
            override = self.model_overrides["generation"]
            if override.get("model_name"):
                return override["model_name"]
        if agent_type == "highlight" and "pre_validation" in self.model_overrides:
            override = self.model_overrides["pre_validation"]
            if override.get("model_name"):
                return override["model_name"]

        if task:
            try:
                preset = ModelPreset(self.model_preset.lower())
                return get_model_for_task(task, preset)
            except (ValueError, AttributeError):
                pass

        return self.default_llm_model

    def get_model_config_for_task(self, task: str) -> dict[str, Any]:
        """Get full model configuration for a task including temperature and tokens."""
        from .models.config import ModelPreset, ModelTask, get_model_config

        try:
            model_task = ModelTask(task.lower())
        except ValueError:
            return {
                "model_name": self.default_llm_model,
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens,
            }

        try:
            preset = ModelPreset(self.model_preset.lower())
        except (ValueError, AttributeError):
            preset = ModelPreset.BALANCED

        task_overrides = self.model_overrides.get(task.lower(), {})

        config = get_model_config(
            model_task, preset, task_overrides if task_overrides else None
        )

        return {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens or self.llm_max_tokens,
            "top_p": config.top_p or self.llm_top_p,
            "reasoning_enabled": config.reasoning_enabled or self.llm_reasoning_enabled,
            "reasoning_effort": self._resolve_reasoning_effort(
                task, config.reasoning_effort
            ),
        }

    def _resolve_reasoning_effort(
        self, task: str | None, model_effort: str | None = None
    ) -> str:
        """Resolve reasoning effort precedence: overrides -> model config -> global."""
        if task:
            key = task.lower()
            if key in self.reasoning_effort_overrides:
                return self.reasoning_effort_overrides[key]

        if model_effort:
            return normalize_reasoning_effort(model_effort)

        return self.llm_reasoning_effort

    def get_reasoning_effort(self, agent_type: str | None = None) -> str:
        """Get reasoning effort for an agent or task."""
        if not agent_type:
            return self.llm_reasoning_effort

        key = agent_type.lower()
        if key in self.reasoning_effort_overrides:
            return self.reasoning_effort_overrides[key]

        agent_to_task = {
            "pre_validator": "pre_validation",
            "post_validator": "post_validation",
            "qa_extractor": "qa_extraction",
            "context_enrichment": "context_enrichment",
            "memorization_quality": "memorization_quality",
            "card_splitting": "card_splitting",
            "split_validator": "card_splitting",
            "duplicate_detection": "duplicate_detection",
            "parser_repair": "parser_repair",
            "note_correction": "parser_repair",
            "highlight": "pre_validation",
            "reasoning": "generation",
            "reflection": "post_validation",
        }

        task_key = agent_to_task.get(key, key)
        return self.reasoning_effort_overrides.get(task_key, self.llm_reasoning_effort)

    @model_validator(mode="after")
    def validate_config(self) -> Config:
        """Validate configuration values after initialization."""
        vault_path = self.vault_path
        if isinstance(vault_path, str):
            if not vault_path:
                msg = "vault_path is required"
                raise ConfigurationError(
                    msg,
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
                msg = f"Cannot create database directory: {parent_dir}"
                raise ConfigurationError(
                    msg,
                    suggestion=f"Ensure you have write permissions to {parent_dir.parent}. Error: {e}",
                )

        if not os.access(parent_dir, os.W_OK):
            msg = f"Database directory is not writable: {parent_dir}"
            raise ConfigurationError(
                msg,
                suggestion=f"Check directory permissions: chmod 755 {parent_dir}",
            )

        if validated_db.exists():
            if not os.access(validated_db, os.R_OK):
                msg = f"Database file exists but is not readable: {validated_db}"
                raise ConfigurationError(
                    msg,
                    suggestion=f"Check file permissions: chmod 644 {validated_db}",
                )
            if not os.access(validated_db, os.W_OK):
                msg = f"Database file exists but is not writable: {validated_db}"
                raise ConfigurationError(
                    msg,
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
            msg = (
                f"Invalid llm_provider: {self.llm_provider}. "
                f"Must be one of: {', '.join(valid_providers)}"
            )
            raise ConfigurationError(
                msg,
                suggestion=f"Set llm_provider to one of: {', '.join(valid_providers)}",
            )

        provider_lower = self.llm_provider.lower()

        if provider_lower == "openrouter" and not self.openrouter_api_key:
            msg = "OpenRouter API key is required when using OpenRouter provider."
            raise ConfigurationError(
                msg,
                suggestion="Set OPENROUTER_API_KEY environment variable or openrouter_api_key in config.yaml",
            )

        if self.run_mode not in ("apply", "dry-run"):
            msg = f"Invalid run_mode: {self.run_mode}"
            raise ConfigurationError(
                msg,
                suggestion="Set run_mode to either 'apply' or 'dry-run'",
            )

        if self.delete_mode not in ("delete", "archive"):
            msg = f"Invalid delete_mode: {self.delete_mode}"
            raise ConfigurationError(
                msg,
                suggestion="Set delete_mode to either 'delete' or 'archive'",
            )

        if not (0 <= self.llm_temperature <= 1):
            msg = f"LLM temperature must be 0-1: {self.llm_temperature}"
            raise ConfigurationError(
                msg,
                suggestion="Set llm_temperature to a value between 0.0 and 1.0",
            )

        if not (0 <= self.llm_top_p <= 1):
            msg = f"LLM top_p must be 0-1: {self.llm_top_p}"
            raise ConfigurationError(
                msg,
                suggestion="Set llm_top_p to a value between 0.0 and 1.0",
            )

        if self.enable_queue:
            if self.queue_max_wait_time_seconds < 60:
                msg = (
                    f"queue_max_wait_time_seconds must be >= 60: "
                    f"{self.queue_max_wait_time_seconds}"
                )
                raise ConfigurationError(
                    msg,
                    suggestion="Set queue_max_wait_time_seconds to at least 60 seconds.",
                )
            if self.queue_job_timeout_seconds < 30:
                msg = (
                    f"queue_job_timeout_seconds must be >= 30: "
                    f"{self.queue_job_timeout_seconds}"
                )
                raise ConfigurationError(
                    msg,
                    suggestion="Set queue_job_timeout_seconds to at least 30 seconds.",
                )
            if self.queue_circuit_breaker_threshold < 1:
                msg = (
                    "queue_circuit_breaker_threshold must be >= 1: "
                    f"{self.queue_circuit_breaker_threshold}"
                )
                raise ConfigurationError(
                    msg,
                    suggestion="Set queue_circuit_breaker_threshold to at least 1.",
                )
            if not (0.1 <= self.queue_poll_interval <= 10.0):
                msg = (
                    f"queue_poll_interval must be 0.1-10.0: {self.queue_poll_interval}"
                )
                raise ConfigurationError(
                    msg,
                    suggestion="Set queue_poll_interval between 0.1 and 10.0 seconds.",
                )

        retry_attrs = [
            ("queue_max_retries", self.queue_max_retries if self.enable_queue else 3),
            ("post_validation_max_retries", self.post_validation_max_retries),
            ("langgraph_max_retries", self.langgraph_max_retries),
        ]
        for attr_name, value in retry_attrs:
            if value < 0 or value > 10:
                msg = f"{attr_name} must be 0-10: {value}"
                raise ConfigurationError(
                    msg,
                    suggestion=f"Set {attr_name} to a value between 0 and 10.",
                )

        if (
            self.max_concurrent_generations < 1
            or self.max_concurrent_generations > 500
        ):
            msg = (
                "max_concurrent_generations must be 1-500: "
                f"{self.max_concurrent_generations}"
            )
            raise ConfigurationError(
                msg,
                suggestion="Set max_concurrent_generations between 1 and 500.",
            )

        if validated_vault != self.vault_path:
            object.__setattr__(self, "vault_path", validated_vault)
        if validated_db != self.db_path:
            object.__setattr__(self, "db_path", validated_db)

        return self

    def get_data_path(self, relative_path: Path | str | None = None) -> Path:
        """Get absolute path within data_dir."""
        data_dir = self.data_dir
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        data_dir = data_dir.resolve()

        if relative_path is None:
            return data_dir
        return data_dir / relative_path

    def get_chroma_db_path(self) -> Path:
        """Get absolute path to ChromaDB directory."""
        return self.get_data_path(self.rag_db_path)

    def get_sync_db_path(self) -> Path:
        """Get absolute path to sync state database."""
        return self.get_data_path(self.db_path)

    def get_log_dir(self) -> Path:
        """Get absolute path to log directory."""
        return self.get_data_path(self.project_log_dir)

    def get_validation_log_dir(self) -> Path:
        """Get absolute path to validation log directory."""
        return self.get_data_path("validation_logs")

    def get_validation_cache_path(self) -> Path:
        """Get absolute path to validation cache file."""
        return self.get_data_path(".validation_cache.json")

    def get_memory_storage_path(self) -> Path:
        """Get absolute path to agent memory storage."""
        return self.get_data_path(self.memory_storage_path)

    def get_cache_dir(self) -> Path:
        """Get absolute path to cache directory."""
        return self.get_data_path(".cache")


__all__ = ["Config"]

