"""Agent orchestrator for coordinating the three-agent pipeline.

This module coordinates:
1. Pre-Validator Agent - structure and format validation
2. Generator Agent - card generation
3. Post-Validator Agent - quality validation

Implements retry logic and auto-fix capabilities.
"""

import hashlib
import time
from pathlib import Path

from ..config import Config
from ..models import Card, Manifest, NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..providers.factory import ProviderFactory
from ..providers.ollama import OllamaProvider
from ..utils.content_hash import compute_content_hash
from ..utils.logging import get_logger
from .generator import GeneratorAgent
from .models import AgentPipelineResult, GeneratedCard
from .post_validator import PostValidatorAgent
from .pre_validator import PreValidatorAgent
from .slug_utils import generate_agent_slug_base

logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates the three-agent pipeline for card generation.

    Manages:
    - Pre-validation of note structure
    - Card generation with powerful LLM
    - Post-validation with retry and auto-fix
    """

    def __init__(self, config: Config, provider: BaseLLMProvider | None = None):
        """Initialize agent orchestrator.

        Args:
            config: Service configuration
            provider: Optional LLM provider instance. If not provided, will create
                     one using ProviderFactory based on config settings.
        """
        self.config = config

        # Initialize LLM provider (use provided or create from config)
        if provider is not None:
            self.provider = provider
        else:
            # Create provider based on config (supports Ollama, LM Studio, OpenRouter)
            try:
                self.provider = ProviderFactory.create_from_config(config)
            except (ConnectionError, ValueError, RuntimeError, OSError) as e:
                # Fallback to OllamaProvider with default settings
                logger.warning(
                    "provider_creation_failed_fallback_to_ollama",
                    error=str(e),
                    llm_provider=getattr(config, "llm_provider", "ollama"),
                )
                ollama_base_url = getattr(
                    config, "ollama_base_url", "http://localhost:11434"
                )
                self.provider = OllamaProvider(base_url=ollama_base_url)

        # Check provider connection
        provider_name = self.provider.get_provider_name()
        if not self.provider.check_connection():
            logger.error("provider_connection_failed", provider=provider_name)
            raise ConnectionError(
                f"Cannot connect to {provider_name} provider. "
                "Please check your provider configuration and ensure the service is running."
            )

        # Initialize agents
        pre_val_model = getattr(config, "pre_validator_model", "qwen3:8b")
        gen_model = getattr(config, "generator_model", "qwen3:32b")
        post_val_model = getattr(config, "post_validator_model", "qwen3:14b")

        self.pre_validator = PreValidatorAgent(
            ollama_client=self.provider,  # Provider is compatible with OllamaClient interface
            model=pre_val_model,
            temperature=getattr(config, "pre_validator_temperature", 0.0),
            enable_content_generation=getattr(
                config, "enable_content_generation", True),
            repair_missing_sections=getattr(
                config, "repair_missing_sections", True),
        )

        self.generator = GeneratorAgent(
            ollama_client=self.provider,  # Provider is compatible with OllamaClient interface
            model=gen_model,
            temperature=getattr(config, "generator_temperature", 0.3),
        )

        self.post_validator = PostValidatorAgent(
            ollama_client=self.provider,  # Provider is compatible with OllamaClient interface
            model=post_val_model,
            temperature=getattr(config, "post_validator_temperature", 0.0),
        )

        logger.info(
            "orchestrator_initialized",
            provider=provider_name,
            pre_val_model=pre_val_model,
            gen_model=gen_model,
            post_val_model=post_val_model,
            pre_val_temp=getattr(config, "pre_validator_temperature", 0.0),
            gen_temp=getattr(config, "generator_temperature", 0.3),
            post_val_temp=getattr(config, "post_validator_temperature", 0.0),
            post_val_max_retries=getattr(
                config, "post_validation_max_retries", 3),
            post_val_auto_fix=getattr(
                config, "post_validation_auto_fix", True),
            post_val_strict=getattr(
                config, "post_validation_strict_mode", True),
        )

    def process_note(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        file_path: Path | None = None,
    ) -> AgentPipelineResult:
        """Process a note through the complete agent pipeline.

        Args:
            note_content: Full note content
            metadata: Parsed metadata
            qa_pairs: Parsed Q/A pairs
            file_path: Optional file path for validation

        Returns:
            AgentPipelineResult with all stages
        """
        start_time = time.time()
        stage_times: dict[str, float] = {}

        # Generate correlation ID for tracking this note through the pipeline
        correlation_id = hashlib.sha256(
            f"{metadata.id}:{int(start_time * 1000)}".encode()
        ).hexdigest()[:12]

        logger.info(
            "pipeline_start",
            correlation_id=correlation_id,
            note_id=metadata.id,
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
            languages=metadata.language_tags,
            file=str(file_path) if file_path else "unknown",
        )

        # Stage 1: Pre-validation
        pre_validation_enabled = getattr(
            self.config, "pre_validation_enabled", True)
        pre_val_start = time.time()

        if pre_validation_enabled:
            pre_result = self.pre_validator.validate(
                note_content=note_content,
                metadata=metadata,
                qa_pairs=qa_pairs,
                file_path=file_path,
            )

            if not pre_result.is_valid and not pre_result.auto_fix_applied:
                total_time = time.time() - start_time
                logger.warning(
                    "pipeline_failed_pre_validation",
                    correlation_id=correlation_id,
                    note_id=metadata.id,
                    title=metadata.title,
                    error_type=pre_result.error_type,
                    error_details=pre_result.error_details,
                    time=total_time,
                )

                return AgentPipelineResult(
                    success=False,
                    pre_validation=pre_result,
                    generation=None,
                    post_validation=None,
                    total_time=total_time,
                    retry_count=0,
                )

            # Use fixed content if auto-fix was applied
            if pre_result.auto_fix_applied and pre_result.fixed_content:
                note_content = pre_result.fixed_content
                logger.info(
                    "using_auto_fixed_content",
                    correlation_id=correlation_id,
                    note_id=metadata.id,
                )
        else:
            # Skip pre-validation, create dummy result
            from .models import PreValidationResult

            pre_result = PreValidationResult(
                is_valid=True,
                error_type="none",
                error_details="Pre-validation skipped",
                auto_fix_applied=False,
                fixed_content=None,
                validation_time=0.0,
            )

        stage_times["pre_validation"] = time.time() - pre_val_start

        # Stage 2: Card Generation
        gen_start = time.time()
        try:
            # Generate slug base from note title
            slug_base = self._generate_slug_base(metadata)

            gen_result = self.generator.generate_cards(
                note_content=note_content,
                metadata=metadata,
                qa_pairs=qa_pairs,
                slug_base=slug_base,
            )
            stage_times["generation"] = time.time() - gen_start

        except Exception as e:
            stage_times["generation"] = time.time() - gen_start
            total_time = time.time() - start_time
            logger.error(
                "pipeline_failed_generation",
                correlation_id=correlation_id,
                note_id=metadata.id,
                title=metadata.title,
                error=str(e),
                time=total_time,
                stage_times=stage_times,
            )

            return AgentPipelineResult(
                success=False,
                pre_validation=pre_result,
                generation=None,
                post_validation=None,
                total_time=total_time,
                retry_count=0,
            )

        # Stage 3: Post-validation with retry
        post_val_start = time.time()
        max_retries = getattr(self.config, "post_validation_max_retries", 3)
        auto_fix = getattr(self.config, "post_validation_auto_fix", True)
        strict_mode = getattr(self.config, "post_validation_strict_mode", True)

        current_cards = gen_result.cards
        retry_count = 0

        for attempt in range(max_retries):
            post_result = self.post_validator.validate(
                cards=current_cards, metadata=metadata, strict_mode=strict_mode
            )

            if post_result.is_valid:
                # Success!
                stage_times["post_validation"] = time.time() - post_val_start
                total_time = time.time() - start_time
                logger.info(
                    "pipeline_success",
                    correlation_id=correlation_id,
                    note_id=metadata.id,
                    title=metadata.title,
                    cards_generated=len(current_cards),
                    retry_count=retry_count,
                    time=total_time,
                    stage_times=stage_times,
                )

                # Create updated generation result with final cards (avoid mutation)
                from .models import GenerationResult

                final_gen_result = GenerationResult(
                    cards=current_cards,
                    total_cards=len(current_cards),
                    generation_time=gen_result.generation_time,
                    model_used=gen_result.model_used,
                )

                return AgentPipelineResult(
                    success=True,
                    pre_validation=pre_result,
                    generation=final_gen_result,
                    post_validation=post_result,
                    total_time=total_time,
                    retry_count=retry_count,
                )

            # Validation failed
            logger.warning(
                "post_validation_failed",
                correlation_id=correlation_id,
                note_id=metadata.id,
                attempt=attempt + 1,
                error_type=post_result.error_type,
                error_details=post_result.error_details,
            )

            # Try auto-fix if enabled
            if auto_fix and post_result.corrected_cards:
                logger.info(
                    "applying_auto_fix",
                    correlation_id=correlation_id,
                    note_id=metadata.id,
                    cards_count=len(post_result.corrected_cards),
                )
                current_cards = post_result.corrected_cards
                retry_count = attempt + 1
            elif auto_fix:
                # Attempt auto-fix manually
                logger.info(
                    "attempting_manual_auto_fix",
                    correlation_id=correlation_id,
                    note_id=metadata.id,
                )
                fixed_cards = self.post_validator.attempt_auto_fix(
                    cards=current_cards, error_details=post_result.error_details
                )

                if fixed_cards:
                    current_cards = fixed_cards
                    retry_count = attempt + 1
                else:
                    # Cannot fix, abort
                    break
            else:
                # Auto-fix disabled, abort
                break

        # Failed after all retries
        stage_times["post_validation"] = time.time() - post_val_start
        total_time = time.time() - start_time
        logger.error(
            "pipeline_failed_post_validation",
            correlation_id=correlation_id,
            note_id=metadata.id,
            title=metadata.title,
            retry_count=retry_count,
            time=total_time,
            stage_times=stage_times,
        )

        return AgentPipelineResult(
            success=False,
            pre_validation=pre_result,
            generation=gen_result,
            post_validation=post_result,
            total_time=total_time,
            retry_count=retry_count,
        )

    def _generate_slug_base(self, metadata: NoteMetadata) -> str:
        """Generate base slug from note metadata using collision-safe helper."""

        return generate_agent_slug_base(metadata)

    def convert_to_cards(
        self,
        generated_cards: list[GeneratedCard],
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
    ) -> list[Card]:
        """Convert GeneratedCard instances to Card instances."""

        cards: list[Card] = []
        qa_lookup = {qa.card_index: qa for qa in qa_pairs}

        for gen_card in generated_cards:
            # Create manifest
            # Safely extract slug_base by removing -index-lang suffix
            parts = gen_card.slug.rsplit("-", 2)
            slug_base = parts[0] if len(parts) >= 3 else gen_card.slug

            manifest = Manifest(
                slug=gen_card.slug,
                slug_base=slug_base,
                lang=gen_card.lang,
                source_path="",  # Will be set by sync engine
                source_anchor=f"qa-{gen_card.card_index}",
                note_id="",  # Will be set by sync engine
                note_title="",  # Will be set by sync engine
                card_index=gen_card.card_index,
                guid=gen_card.slug,  # Use slug as GUID for now
                hash6=None,
            )

            qa_pair = qa_lookup.get(gen_card.card_index)
            content_hash = gen_card.content_hash
            if not content_hash and qa_pair:
                content_hash = compute_content_hash(
                    qa_pair, metadata, gen_card.lang)
            elif not content_hash:
                content_hash = hashlib.sha256(
                    gen_card.apf_html.encode("utf-8")
                ).hexdigest()

            cards.append(
                Card(
                    slug=gen_card.slug,
                    lang=gen_card.lang,
                    apf_html=gen_card.apf_html,
                    manifest=manifest,
                    content_hash=content_hash,
                    note_type="APF::Simple",  # Default, can be detected from HTML
                    tags=[],  # Extract from manifest in HTML
                    guid=gen_card.slug,
                )
            )

        return cards
