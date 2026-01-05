"""Infrastructure service for LLM-based card generation.

This module handles the low-level LLM integration for card generation,
following Clean Architecture principles.
"""

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from obsidian_anki_sync.agents.debug_artifacts import save_failed_llm_call

if TYPE_CHECKING:
    from obsidian_anki_sync.agents.models import GeneratedCard
from obsidian_anki_sync.agents.llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
    should_retry_llm_error,
)
from obsidian_anki_sync.agents.metrics import record_operation_metric
from obsidian_anki_sync.domain.interfaces.card_generation import (
    ICardDataExtractor,
    ISingleCardGenerator,
    ITranslatedCardGenerator,
    ParsedCardStructure,
)
from obsidian_anki_sync.domain.interfaces.tag_generation import (
    ICodeDetector,
    ITagGenerator,
)
from obsidian_anki_sync.infrastructure.html_post_processor import APFHTMLPostProcessor
from obsidian_anki_sync.infrastructure.manifest.manifest_generator import (
    ManifestGenerator,
)
from obsidian_anki_sync.models import Manifest, NoteMetadata, QAPair
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.llm_logging import log_slow_llm_request
from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class LLMCardGenerator(
    ISingleCardGenerator, ITranslatedCardGenerator, ICardDataExtractor
):
    """Infrastructure service for LLM-based card generation.

    Handles low-level LLM integration and card generation logic.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:32b",
        temperature: float = 0.3,
        slow_request_threshold: float = 60.0,
        tag_generator: ITagGenerator | None = None,
        code_detector: ICodeDetector | None = None,
        html_post_processor: APFHTMLPostProcessor | None = None,
        manifest_generator: ManifestGenerator | None = None,
    ):
        """Initialize LLM card generator.

        Args:
            ollama_client: LLM provider instance
            model: Model to use for generation
            temperature: Sampling temperature
            slow_request_threshold: Threshold for slow request logging
            tag_generator: Tag generation service
            code_detector: Code detection service
            html_post_processor: HTML post-processing service
            manifest_generator: Manifest generation service
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.slow_request_threshold = slow_request_threshold

        # Dependencies (with defaults for backward compatibility)
        self.tag_generator = tag_generator
        self.code_detector = code_detector
        self.html_post_processor = html_post_processor or APFHTMLPostProcessor()
        self.manifest_generator = manifest_generator or ManifestGenerator()

        # Load system prompt from CARDS_PROMPT.md
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = "Generate APF cards following strict APF v2.1 format."

        logger.info("llm_card_generator_initialized", model=model)

    def generate_single_card(
        self, qa_pair: QAPair, metadata: NoteMetadata, manifest: Manifest, lang: str
    ) -> "GeneratedCard":
        """Generate a single APF card.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            manifest: Card manifest
            lang: Language code

        Returns:
            GeneratedCard instance
        """
        from obsidian_anki_sync.agents.models import GeneratedCard

        card_start_time = time.time()

        # Select language-specific content
        question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

        # Validate that content exists for this language
        if not question or not question.strip():
            logger.error(
                "empty_question_for_lang",
                lang=lang,
                slug=manifest.slug,
                card_index=qa_pair.card_index,
            )
            msg = f"Empty question for language '{lang}' in card {qa_pair.card_index}"
            raise ValueError(msg)

        if not answer or not answer.strip():
            logger.error(
                "empty_answer_for_lang",
                lang=lang,
                slug=manifest.slug,
                card_index=qa_pair.card_index,
            )
            msg = f"Empty answer for language '{lang}' in card {qa_pair.card_index}"
            raise ValueError(msg)

        # Build user prompt for card generation
        user_prompt = self._build_user_prompt(
            question=question,
            answer=answer,
            qa_pair=qa_pair,
            metadata=metadata,
            manifest=manifest,
            lang=lang,
        )

        # Estimate token count (rough: 4 chars per token)
        total_input_chars = len(user_prompt) + len(self.system_prompt)
        estimated_tokens = total_input_chars // 4

        # Context window warnings for common models
        context_limits = {
            "qwen3:8b": 8192,
            "qwen3:14b": 8192,
            "qwen3:32b": 32768,
        }

        model_limit = context_limits.get(self.model, 8192)
        utilization_pct = (estimated_tokens / model_limit) * 100

        logger.info(
            "generating_single_card",
            model=self.model,
            slug=manifest.slug,
            card_index=qa_pair.card_index,
            lang=lang,
            prompt_length=len(user_prompt),
            system_length=len(self.system_prompt),
            estimated_tokens=estimated_tokens,
            context_limit=model_limit,
            context_utilization_pct=round(utilization_pct, 1),
        )

        # Warn if approaching context limit
        if utilization_pct > 80:
            logger.warning(
                "high_context_utilization",
                slug=manifest.slug,
                utilization_pct=round(utilization_pct, 1),
                estimated_tokens=estimated_tokens,
                context_limit=model_limit,
                recommendation="Consider reducing prompt size or using a model with larger context window",
            )
        elif utilization_pct > 90:
            logger.error(
                "critical_context_utilization",
                slug=manifest.slug,
                utilization_pct=round(utilization_pct, 1),
                estimated_tokens=estimated_tokens,
                context_limit=model_limit,
                risk="Very high risk of context truncation",
            )

        # Retry logic for LLM calls
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Call Ollama LLM
                llm_start_time = time.time()

                logger.info(
                    "llm_generation_attempt",
                    slug=manifest.slug,
                    attempt=attempt,
                    max_attempts=max_retries,
                    model=self.model,
                )

                result = self.ollama_client.generate(
                    model=self.model,
                    prompt=user_prompt,
                    system=self.system_prompt,
                    temperature=self.temperature,
                )
                llm_duration = time.time() - llm_start_time

                log_slow_llm_request(
                    duration_seconds=llm_duration,
                    threshold_seconds=self.slow_request_threshold,
                    model=self.model,
                    operation="card_generation",
                )

                apf_html = result.get("response", "")

                if not apf_html or not apf_html.strip():
                    msg = "LLM returned empty response"
                    raise ValueError(msg)

                # Check for truncation indicators
                truncation_indicators = self._detect_truncation(apf_html)
                if truncation_indicators:
                    logger.warning(
                        "possible_response_truncation",
                        slug=manifest.slug,
                        indicators=truncation_indicators,
                        response_length=len(apf_html),
                        attempt=attempt,
                    )
                    # If severely truncated and we have retries left, retry
                    if attempt < max_retries and len(truncation_indicators) > 1:
                        msg = f"Response appears truncated: {', '.join(truncation_indicators)}"
                        raise ValueError(msg)

                # Extract token usage if available
                token_usage = result.get("_token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)

                # Post-process APF HTML
                post_process_start = time.time()
                apf_html = self.html_post_processor.post_process_apf(
                    apf_html, metadata, manifest
                )
                post_process_duration = time.time() - post_process_start

                # Extract confidence from LLM response (if available)
                confidence = 0.9

                card_duration = time.time() - card_start_time

                logger.info(
                    "single_card_generated",
                    slug=manifest.slug,
                    card_index=qa_pair.card_index,
                    lang=lang,
                    response_length=len(apf_html),
                    llm_duration=round(llm_duration, 2),
                    post_process_duration=round(post_process_duration, 3),
                    total_duration=round(card_duration, 2),
                    attempts_needed=attempt,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                # Record success metrics
                record_operation_metric(
                    operation="card_generation",
                    success=True,
                    duration=card_duration,
                    llm_duration=llm_duration,
                    tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retried=(attempt > 1),
                )

                content_hash = compute_content_hash(qa_pair, metadata, lang)

                return GeneratedCard(
                    card_index=qa_pair.card_index,
                    slug=manifest.slug,
                    lang=lang,
                    apf_html=apf_html,
                    confidence=confidence,
                    content_hash=content_hash,
                )

            except Exception as e:
                llm_duration = time.time() - llm_start_time

                # Try to extract response if available (for debug artifacts)
                response_text = None
                if hasattr(e, "__context__") and hasattr(e.__context__, "response"):
                    try:
                        response_text = getattr(e.__context__, "response", None)
                    except (AttributeError, TypeError) as attr_err:
                        logger.debug(
                            "response_extraction_failed",
                            error=str(attr_err),
                        )

                # Categorize the error
                llm_error = categorize_llm_error(
                    error=e,
                    model=self.model,
                    operation=f"card generation ({manifest.slug})",
                    duration=llm_duration,
                )

                # Log the error with context
                log_llm_error(
                    llm_error,
                    slug=manifest.slug,
                    card_index=qa_pair.card_index,
                    lang=lang,
                    attempt=attempt,
                    max_attempts=max_retries,
                )

                # Check if we should retry
                if should_retry_llm_error(llm_error, attempt, max_retries):
                    # Calculate exponential backoff delay
                    delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.info(
                        "retrying_after_delay",
                        slug=manifest.slug,
                        attempt=attempt,
                        delay_seconds=delay,
                    )
                    time.sleep(delay)
                    continue

                # No more retries or non-retryable error - save debug artifacts
                card_duration = time.time() - card_start_time

                # Save debug artifact for this failure
                artifact_path = save_failed_llm_call(
                    operation=f"card_generation_{manifest.slug}",
                    model=self.model,
                    prompt=user_prompt,
                    system_prompt=self.system_prompt,
                    response=response_text,
                    error=llm_error,
                    slug=manifest.slug,
                    card_index=qa_pair.card_index,
                    lang=lang,
                    attempts_made=attempt,
                )

                logger.error(
                    "card_generation_failed",
                    slug=manifest.slug,
                    error_type=llm_error.error_type.value,
                    error=str(llm_error),
                    user_message=format_llm_error_for_user(llm_error),
                    duration=round(card_duration, 2),
                    attempts_made=attempt,
                    debug_artifact=str(artifact_path) if artifact_path else None,
                )

                # Record failure metrics
                record_operation_metric(
                    operation="card_generation",
                    success=False,
                    duration=card_duration,
                    llm_duration=llm_duration,
                    retried=(attempt > 1),
                    error_type=llm_error.error_type.value,
                )

                raise llm_error from e

        # Should never reach here, but just in case
        msg = f"Failed to generate card after {max_retries} attempts"
        raise RuntimeError(msg)

    def generate_translated_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        english_structure: "ParsedCardStructure",
        lang: str,
    ) -> "GeneratedCard":
        """Generate a translated card using the canonical English structure.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            manifest: Card manifest for the target language
            english_structure: Parsed structure from the English card
            lang: Target language code

        Returns:
            GeneratedCard with translated content
        """
        from obsidian_anki_sync.agents.models import GeneratedCard

        card_start_time = time.time()

        # Get the localized question and answer
        question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

        # Validate content exists
        if not question or not question.strip():
            msg = f"Empty question for language '{lang}' in card {qa_pair.card_index}"
            raise ValueError(msg)
        if not answer or not answer.strip():
            msg = f"Empty answer for language '{lang}' in card {qa_pair.card_index}"
            raise ValueError(msg)

        # Build translation prompt using English structure
        user_prompt = self._build_translation_prompt(
            question=question,
            answer=answer,
            english_structure=english_structure,
            metadata=metadata,
            manifest=manifest,
            lang=lang,
        )

        # Estimate token count
        total_input_chars = len(user_prompt) + len(self.system_prompt)
        estimated_tokens = total_input_chars // 4

        context_limits = {
            "qwen3:8b": 8192,
            "qwen3:14b": 8192,
            "qwen3:32b": 32768,
        }
        model_limit = context_limits.get(self.model, 8192)
        utilization_pct = (estimated_tokens / model_limit) * 100

        logger.info(
            "generating_translated_card",
            model=self.model,
            slug=manifest.slug,
            card_index=qa_pair.card_index,
            lang=lang,
            prompt_length=len(user_prompt),
            system_length=len(self.system_prompt),
            estimated_tokens=estimated_tokens,
            context_limit=model_limit,
            context_utilization_pct=round(utilization_pct, 1),
        )

        if utilization_pct > 80:
            logger.warning(
                "high_context_utilization_translation",
                slug=manifest.slug,
                utilization_pct=round(utilization_pct, 1),
                estimated_tokens=estimated_tokens,
                context_limit=model_limit,
            )

        # Generate translated card with retry logic
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                llm_start_time = time.time()

                logger.info(
                    "llm_translation_attempt",
                    slug=manifest.slug,
                    attempt=attempt,
                    max_attempts=max_retries,
                    model=self.model,
                )

                result = self.ollama_client.generate(
                    model=self.model,
                    prompt=user_prompt,
                    system=self.system_prompt,
                    temperature=self.temperature,
                )

                llm_duration = time.time() - llm_start_time
                log_slow_llm_request(
                    duration_seconds=llm_duration,
                    threshold_seconds=self.slow_request_threshold,
                    model=self.model,
                    operation="card_translation",
                )
                translated_html = result.get("response", "")

                if not translated_html or not translated_html.strip():
                    msg = "LLM returned empty translation response"
                    raise ValueError(msg)

                # Assemble final APF HTML using English structure + translated text
                final_apf_html = self._assemble_translated_card_html(
                    english_structure=english_structure,
                    translated_html=translated_html,
                    metadata=metadata,
                    manifest=manifest,
                )

                # Post-process the assembled HTML
                post_process_start = time.time()
                final_apf_html = self.html_post_processor.post_process_apf(
                    final_apf_html, metadata, manifest
                )
                post_process_duration = time.time() - post_process_start

                confidence = 0.9
                card_duration = time.time() - card_start_time

                logger.info(
                    "translated_card_generated",
                    slug=manifest.slug,
                    card_index=qa_pair.card_index,
                    lang=lang,
                    response_length=len(final_apf_html),
                    llm_duration=round(llm_duration, 2),
                    post_process_duration=round(post_process_duration, 3),
                    total_duration=round(card_duration, 2),
                    attempts_needed=attempt,
                )

                # Record metrics
                token_usage = result.get("_token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)

                record_operation_metric(
                    operation="card_translation",
                    success=True,
                    duration=card_duration,
                    llm_duration=llm_duration,
                    tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    retried=(attempt > 1),
                )

                content_hash = compute_content_hash(qa_pair, metadata, lang)

                return GeneratedCard(
                    card_index=qa_pair.card_index,
                    slug=manifest.slug,
                    lang=lang,
                    apf_html=final_apf_html,
                    confidence=confidence,
                    content_hash=content_hash,
                )

            except Exception as e:
                llm_duration = time.time() - llm_start_time

                # Categorize and handle errors
                llm_error = categorize_llm_error(
                    error=e,
                    model=self.model,
                    operation=f"card translation ({manifest.slug})",
                    duration=llm_duration,
                )
                log_llm_error(llm_error, slug=manifest.slug, lang=lang, attempt=attempt)

                if attempt == max_retries:
                    # Record failure metrics
                    record_operation_metric(
                        operation="card_translation",
                        success=False,
                        duration=time.time() - card_start_time,
                        retried=(attempt > 1),
                        error_type=llm_error.error_type.value,
                    )
                    raise llm_error from e

                # Continue to next attempt

        # Should never reach here
        msg = f"Failed to generate translated card after {max_retries} attempts"
        raise RuntimeError(msg)

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> str:
        """Build user prompt for LLM.

        Uses deterministic tag generation and provides clear examples.
        """
        ref_link = f"[[{manifest.source_path}#{manifest.source_anchor}]]"

        # Detect code language from answer
        detected_lang = (
            self.code_detector.detect_code_language(answer)
            if self.code_detector
            else "text"
        )
        code_lang = (
            detected_lang
            if detected_lang != "text"
            else (
                self.code_detector.get_code_language_hint(metadata)
                if self.code_detector
                else "text"
            )
        )

        # Generate tags deterministically
        suggested_tags = (
            self.tag_generator.generate_tags(metadata, lang)
            if self.tag_generator
            else []
        )

        prompt = f"""Generate an APF card in HTML format following APF v2.1 specification.

Metadata:
- Topic: {metadata.topic}
- Subtopics: {", ".join(metadata.subtopics) if metadata.subtopics else "None"}
- Difficulty: {metadata.difficulty or "Not specified"}
- Language: {lang}
- Slug: {manifest.slug}
- Code Language: {code_lang}

Question:
{question}

Answer:
{answer}"""

        # Add optional context
        if qa_pair.followups:
            prompt += f"\nFollow-ups:\n{qa_pair.followups}\n"

        if qa_pair.references:
            prompt += f"\nReferences:\n{qa_pair.references}\n"

        if qa_pair.related:
            prompt += f"\nRelated Questions:\n{qa_pair.related}\n"

        if qa_pair.context:
            prompt += f"\nAdditional Context:\n{qa_pair.context}\n"

        # Add requirements with few-shot example
        prompt += f"""
Requirements:
- Use EXACTLY these tags: {" ".join(suggested_tags)}
- CardType: Simple (or Missing if cloze {{{{c1::...}}}}, Draw if diagram)
- CRITICAL: ALL code blocks MUST use <pre><code> format - NEVER use standalone <code> tags
- Preserve code blocks with <pre><code class="language-{code_lang}">...</code></pre>
- Add "Ref: {ref_link}" in Other notes section
- Follow exact structure below including ALL wrapper sentinels
- ALWAYS include <!-- END_CARDS --> before END_OF_CARDS

CRITICAL: Output ONLY the complete APF v2.1 format below. NO explanations before or after.

CRITICAL FORMAT REQUIREMENTS:
- Card headers MUST use 'CardType:' (capital C and T) - NOT 'type:'
- END_OF_CARDS must be the absolute last line with NO content after it
- Manifest uses "type" but headers use "CardType:" - this is correct per APF v2.1

Example Structure (follow exactly):

<!-- PROMPT_VERSION: apf-v2.1 -->
<!-- BEGIN_CARDS -->

<!-- Card 1 | slug: {manifest.slug} | CardType: Simple | Tags: {" ".join(suggested_tags)} -->

<!-- Title -->
[Your question here - max 80 chars]

<!-- Key point (code block) -->
<pre><code class="language-{code_lang}">
[Your answer code here - preserve indentation]
</code></pre>

<!-- Key point notes -->
<ul>
  <li>Mechanism or rule explanation (max 20 words)</li>
  <li>Key constraint or important detail</li>
  <li>Common pitfall or edge case</li>
</ul>

<!-- Other notes (optional) -->
<ul>
  <li>Ref: {ref_link}</li>
</ul>

<!-- manifest: {{"slug":"{manifest.slug}","lang":"{lang}","type":"Simple","tags":{json.dumps(suggested_tags)}}} -->

<!-- END_CARDS -->
END_OF_CARDS

Now generate the card following this structure:
"""

        return prompt

    def _build_translation_prompt(
        self,
        question: str,
        answer: str,
        english_structure: "ParsedCardStructure",
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> str:
        """Build prompt for translating card content while preserving structure.

        Args:
            question: Localized question text
            answer: Localized answer text
            english_structure: Parsed English card structure
            manifest: Card manifest
            lang: Target language

        Returns:
            Translation prompt
        """
        import json

        prompt = f"""Translate the TEXT CONTENT of an English APF card to {lang.upper()}, preserving the EXACT structure and code blocks.

English Card Structure:
- Title: "{english_structure.title}"
- Key Point Notes: {json.dumps(english_structure.key_point_notes, ensure_ascii=False)}
{f"- Other Notes: {json.dumps(english_structure.other_notes, ensure_ascii=False)}" if english_structure.other_notes else ""}

Localized Content:
- Question: {question}
- Answer: {answer}

CRITICAL REQUIREMENTS:
- PRESERVE the exact same card structure and logic
- TRANSLATE ONLY the text content (titles, bullet points)
- KEEP ALL code blocks IDENTICAL (do not translate code)
- Maintain the same number of bullet points with same logical meaning
- Do not add or remove information - only translate existing content

Output ONLY the translated text sections in JSON format:

{{
  "title": "translated title",
  "key_point_notes": ["translated bullet 1", "translated bullet 2", ...],
  "other_notes": ["translated other note 1", ...]  // or null if no other notes
}}

Translated content:"""

        return prompt

    def _assemble_translated_card_html(
        self,
        english_structure: "ParsedCardStructure",
        translated_html: str,
        metadata: NoteMetadata,
        manifest: Manifest,
    ) -> str:
        """Assemble the final APF HTML using English structure + translated text.

        Args:
            english_structure: Parsed English card structure
            translated_html: JSON with translated text sections
            manifest: Card manifest

        Returns:
            Complete APF HTML
        """
        import json

        try:
            # Parse the translated JSON
            translated_data = json.loads(translated_html.strip())

            title = translated_data.get("title", english_structure.title)
            key_point_notes = translated_data.get(
                "key_point_notes", english_structure.key_point_notes
            )
            other_notes = translated_data.get(
                "other_notes", english_structure.other_notes
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "translation_json_parse_failed",
                error=str(e),
                slug=manifest.slug,
                lang=manifest.lang,
                falling_back_to_english=True,
            )
            # Fallback to English structure if JSON parsing fails
            title = english_structure.title
            key_point_notes = english_structure.key_point_notes
            other_notes = english_structure.other_notes

        # Build the APF HTML using the standard template structure
        tags_str = " ".join(
            self.tag_generator.generate_tags(metadata, manifest.lang)
            if self.tag_generator
            else []
        )

        html_parts = [
            f"<!-- Card {manifest.card_index} | slug: {manifest.slug} | CardType: Simple | Tags: {tags_str} -->",
            "",
            "<!-- Title -->",
            title,
            "",
            "<!-- Key point (code block / image) -->",
        ]

        if english_structure.key_point_code:
            html_parts.append(english_structure.key_point_code)
        html_parts.append("")

        # Key point notes
        html_parts.extend(
            [
                "<!-- Key point notes -->",
                "<ul>",
            ]
        )
        for note in key_point_notes:
            html_parts.append(f"  <li>{note}</li>")
        html_parts.extend(
            [
                "</ul>",
                "",
            ]
        )

        # Other notes (if present)
        if other_notes:
            html_parts.extend(
                [
                    "<!-- Other notes -->",
                    "<ul>",
                ]
            )
            for note in other_notes:
                html_parts.append(f"  <li>{note}</li>")
            html_parts.extend(
                [
                    "</ul>",
                    "",
                ]
            )

        # Manifest
        import json

        manifest_dict = {
            "slug": manifest.slug,
            "lang": manifest.lang,
            "type": "Simple",
            "tags": self.tag_generator.generate_tags(metadata, manifest.lang)
            if self.tag_generator
            else [],
        }
        html_parts.append(f"<!-- manifest: {json.dumps(manifest_dict)} -->")

        return "\n".join(html_parts)

    def _detect_truncation(self, html: str) -> list[str]:
        """Detect if a response appears to be truncated.

        Checks for common indicators of truncation:
        - Unclosed HTML tags (code, pre, ul, li, etc.)
        - Missing required APF sections
        - Response ends mid-word or mid-tag

        Args:
            html: The generated HTML content

        Returns:
            List of truncation indicators found (empty if none)
        """
        indicators = []

        # Check for unclosed code blocks
        code_opens = html.count("<code")
        code_closes = html.count("</code>")
        if code_opens > code_closes:
            indicators.append(
                f"unclosed_code_tags ({code_opens} opens, {code_closes} closes)"
            )

        pre_opens = html.count("<pre")
        pre_closes = html.count("</pre>")
        if pre_opens > pre_closes:
            indicators.append(
                f"unclosed_pre_tags ({pre_opens} opens, {pre_closes} closes)"
            )

        # Check for unclosed lists
        ul_opens = html.count("<ul")
        ul_closes = html.count("</ul>")
        if ul_opens > ul_closes:
            indicators.append(
                f"unclosed_ul_tags ({ul_opens} opens, {ul_closes} closes)"
            )

        li_opens = html.count("<li")
        li_closes = html.count("</li>")
        if li_opens > li_closes:
            indicators.append(
                f"unclosed_li_tags ({li_opens} opens, {li_closes} closes)"
            )

        # Check for truncated SVG (common issue)
        if "<svg" in html.lower() and "</svg>" not in html.lower():
            indicators.append("unclosed_svg_tag")

        # Check if response ends in incomplete syntax
        stripped = html.rstrip()
        if stripped:
            # Ends with incomplete HTML tag
            if stripped.endswith("<") or re.search(
                r"<[a-z]+[^>]*$", stripped, re.IGNORECASE
            ):
                indicators.append("ends_with_incomplete_tag")

            # Ends with incomplete attribute
            if re.search(r'="[^"]*$', stripped):
                indicators.append("ends_with_incomplete_attribute")

            # Very short response (less than 500 chars) for card generation
            if len(stripped) < 500:
                indicators.append(f"very_short_response ({len(stripped)} chars)")

        return indicators

    def extract_card_data_from_html(
        self, apf_html: str, manifest: Manifest
    ) -> dict | None:
        """Extract card data from existing HTML for regeneration."""
        # Safe access to optional attributes with sensible defaults
        manifest_tags = getattr(manifest, "tags", [])

        card_data = {
            "card_index": 1,
            "slug": manifest.slug,
            "tags": manifest_tags,
            "title": "Generated Card",
            "question": "",
            "answer": "",
            "code_sample": None,
            "key_points": [],
            "other_notes": "",
            "references": "",
        }

        try:
            # Extract title
            title_match = re.search(
                r"<!-- Title -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if title_match:
                card_data["title"] = title_match.group(1).strip()

            # Extract question
            question_match = re.search(
                r"<!-- Question -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if question_match:
                card_data["question"] = question_match.group(1).strip()

            # Extract answer
            answer_match = re.search(
                r"<!-- Answer -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if answer_match:
                card_data["answer"] = answer_match.group(1).strip()

            # Extract key points
            key_points_match = re.search(
                r"<!-- Key point -->\s*\n(.*?)(?=\n<!--|\n*$)", apf_html, re.DOTALL
            )
            if key_points_match:
                key_points_html = key_points_match.group(1).strip()
                # Extract list items
                li_matches = re.findall(
                    r"<li>(.*?)</li>", key_points_html, re.IGNORECASE
                )
                if li_matches:
                    card_data["key_points"] = [li.strip() for li in li_matches]

            # Extract code samples
            code_match = re.search(
                r"<!-- Sample \(code block\) -->\s*\n(.*?)(?=\n<!--|\n*$)",
                apf_html,
                re.DOTALL,
            )
            if code_match:
                code_html = code_match.group(1).strip()
                # Extract code content
                code_content_match = re.search(
                    r"<pre><code[^>]*>(.*?)</code></pre>",
                    code_html,
                    re.DOTALL | re.IGNORECASE,
                )
                if code_content_match:
                    card_data["code_sample"] = code_content_match.group(1).strip()

        except Exception as e:
            logger.debug(
                "card_data_extraction_failed", slug=manifest.slug, error=str(e)
            )
            return None

        return card_data
