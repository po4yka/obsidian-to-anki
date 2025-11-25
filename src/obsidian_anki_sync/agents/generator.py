"""Generator agent for APF card generation using local LLM.

This agent generates APF cards from Q/A pairs using Ollama with powerful models
like qwen3:32b for high-quality card generation.
"""

import hashlib
import json
import re
import time
from pathlib import Path

from ..apf.html_generator import HTMLTemplateGenerator
from ..models import Manifest, NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.code_detection import (
    detect_code_language_from_content,
    detect_code_language_from_metadata,
)
from ..utils.content_hash import compute_content_hash
from ..utils.logging import get_logger
from .debug_artifacts import save_failed_llm_call
from .llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
    should_retry_llm_error,
)
from .metrics import record_operation_metric
from .models import GeneratedCard, GenerationResult

logger = get_logger(__name__)


class GeneratorAgent:
    """Agent for generating APF cards using local LLM.

    Uses powerful model (qwen3:32b) for high-quality card generation.
    Reuses existing APF generation logic from APFGenerator.
    """

    def __init__(
        self,
        ollama_client: BaseLLMProvider,
        model: str = "qwen3:32b",
        temperature: float = 0.3,
    ):
        """Initialize generator agent.

        Args:
            ollama_client: LLM provider instance (BaseLLMProvider)
            model: Model to use for generation
            temperature: Sampling temperature
        """
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature

        # Load system prompt from CARDS_PROMPT.md
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = "Generate APF cards following strict APF v2.1 format."

        logger.info("generator_agent_initialized", model=model)

    def generate_cards(
        self,
        note_content: str,
        metadata: NoteMetadata,
        qa_pairs: list[QAPair],
        slug_base: str,
    ) -> GenerationResult:
        """Generate APF cards for all Q/A pairs in a note.

        Args:
            note_content: Full note content (for context)
            metadata: Note metadata
            qa_pairs: List of Q/A pairs to convert to cards
            slug_base: Base slug for card generation

        Returns:
            GenerationResult with all generated cards
        """
        start_time = time.time()

        logger.info(
            "card_generation_start",
            title=metadata.title,
            qa_pairs_count=len(qa_pairs),
            languages=metadata.language_tags,
        )

        generated_cards: list[GeneratedCard] = []

        # Generate cards for each Q/A pair in each language
        for qa_pair in qa_pairs:
            for lang in metadata.language_tags:
                # Create manifest for this card
                manifest = self._create_manifest(
                    qa_pair=qa_pair,
                    metadata=metadata,
                    slug_base=slug_base,
                    lang=lang,
                )

                # Generate APF card
                card = self._generate_single_card(
                    qa_pair=qa_pair, metadata=metadata, manifest=manifest, lang=lang
                )

                generated_cards.append(card)

        generation_time = time.time() - start_time

        # Calculate aggregate statistics for future metrics
        total_confidence = sum(card.confidence for card in generated_cards)
        avg_confidence = (
            total_confidence / len(generated_cards) if generated_cards else 0
        )
        logger.debug(
            "generation_confidence_stats",
            avg_confidence=avg_confidence,
            total=total_confidence,
        )

        logger.info(
            "card_generation_complete",
            cards_generated=len(generated_cards),
            time=generation_time,
            avg_time_per_card=(
                round(generation_time / len(generated_cards), 2)
                if generated_cards
                else 0
            ),
        )

        return GenerationResult(
            cards=generated_cards,
            total_cards=len(generated_cards),
            generation_time=generation_time,
            model_used=self.model,
        )

    def _create_manifest(
        self, qa_pair: QAPair, metadata: NoteMetadata, slug_base: str, lang: str
    ) -> Manifest:
        """Create card manifest.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            slug_base: Base slug
            lang: Language code

        Returns:
            Manifest instance
        """
        # Generate slug
        slug = f"{slug_base}-{qa_pair.card_index}-{lang}"

        # Generate GUID (deterministic based on slug using SHA256)
        guid = hashlib.sha256(slug.encode()).hexdigest()[:16]

        return Manifest(
            slug=slug,
            slug_base=slug_base,
            lang=lang,
            source_path=metadata.title,  # Simplified for agent system
            source_anchor=f"qa-{qa_pair.card_index}",
            note_id=metadata.id,
            note_title=metadata.title,
            card_index=qa_pair.card_index,
            guid=guid,
            hash6=None,
        )

    def _generate_single_card(
        self, qa_pair: QAPair, metadata: NoteMetadata, manifest: Manifest, lang: str
    ) -> GeneratedCard:
        """Generate a single APF card.

        Args:
            qa_pair: Q/A pair
            metadata: Note metadata
            manifest: Card manifest
            lang: Language code

        Returns:
            GeneratedCard instance
        """
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
            raise ValueError(
                f"Empty question for language '{lang}' in card {qa_pair.card_index}"
            )

        if not answer or not answer.strip():
            logger.error(
                "empty_answer_for_lang",
                lang=lang,
                slug=manifest.slug,
                card_index=qa_pair.card_index,
            )
            raise ValueError(
                f"Empty answer for language '{lang}' in card {qa_pair.card_index}"
            )

        # Build user prompt (reuse logic from APFGenerator)
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
            "llama3:8b": 8192,
            "llama3:70b": 8192,
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

                apf_html = result.get("response", "")

                if not apf_html or not apf_html.strip():
                    raise ValueError("LLM returned empty response")

                # Extract token usage if available
                token_usage = result.get("_token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)

                # Post-process APF HTML (normalize code blocks, ensure manifest)
                post_process_start = time.time()
                apf_html = self._post_process_apf(apf_html, metadata, manifest)
                post_process_duration = time.time() - post_process_start

                # Extract confidence from LLM response (if available)
                # For now, use a default confidence
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
                        # Defensive: Ignore if response extraction fails
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
        raise RuntimeError(f"Failed to generate card after {max_retries} attempts")

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
        detected_lang = self._detect_code_language(answer)
        code_lang = (
            detected_lang
            if detected_lang != "text"
            else self._code_language_hint(metadata)
        )

        # Generate tags deterministically
        suggested_tags = self._generate_tags(metadata, lang)

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
{answer}
"""

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

    def _code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks."""
        return detect_code_language_from_metadata(metadata)

    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code content.

        Uses syntax patterns to automatically detect the language.

        Args:
            code: Code snippet to analyze

        Returns:
            Detected language name or 'text' if unknown
        """
        if not code or not code.strip():
            return "text"

        detected = detect_code_language_from_content(code)
        return detected if detected else "text"

    def _generate_tags(self, metadata: NoteMetadata, lang: str) -> list[str]:
        """Generate deterministic tags from metadata.

        This ensures tag taxonomy compliance and consistency.

        Args:
            metadata: Note metadata
            lang: Language code

        Returns:
            List of 3-6 snake_case tags
        """
        tags = []

        # 1. Primary language/tech (required first)
        primary_tech = self._code_language_hint(metadata)
        if primary_tech != "plaintext":
            tags.append(primary_tech)

        # 2. Platform (if available)
        platforms = {
            "android",
            "ios",
            "kmp",
            "jvm",
            "nodejs",
            "browser",
            "linux",
            "macos",
            "windows",
        }
        for tag in metadata.tags:
            tag_lower = tag.lower().replace("-", "_")
            if tag_lower in platforms and tag_lower not in tags:
                tags.append(tag_lower)
                break

        # 3. Topic-based tag
        topic_tag = metadata.topic.lower().replace(" ", "_").replace("-", "_")
        if topic_tag not in tags:
            tags.append(topic_tag)

        # 4. Subtopic tags (up to 3 more)
        for subtopic in metadata.subtopics:
            if len(tags) >= 6:
                break
            tag = subtopic.lower().replace(" ", "_").replace("-", "_")
            if tag not in tags:
                tags.append(tag)

        # 5. Difficulty (if less than 6 tags and specified)
        if len(tags) < 6 and metadata.difficulty:
            difficulty_tag = f"difficulty_{metadata.difficulty.lower()}"
            if difficulty_tag not in tags:
                tags.append(difficulty_tag)

        # Ensure at least 3 tags
        while len(tags) < 3:
            if "programming" not in tags:
                tags.append("programming")
            elif "conceptual" not in tags:
                tags.append("conceptual")
            else:
                tags.append("general")

        return tags[:6]  # Max 6 tags

    def _generate_manifest(
        self, manifest: Manifest, card_type: str, tags: list[str]
    ) -> str:
        """Generate manifest JSON string.

        Args:
            manifest: Card manifest
            card_type: Card type (Simple, Missing, Draw)
            tags: List of tags

        Returns:
            Manifest comment string
        """
        manifest_dict = {
            "slug": manifest.slug,
            "lang": manifest.lang,
            "type": card_type,
            "tags": tags,
        }
        return f"<!-- manifest: {json.dumps(manifest_dict, ensure_ascii=False)} -->"

    def _post_process_apf(
        self, apf_html: str, metadata: NoteMetadata, manifest: Manifest
    ) -> str:
        """Post-process APF HTML to ensure correctness.

        This method:
        1. Strips markdown code fences
        2. Ensures APF v2.1 wrapper sentinels are present
        3. Removes explanatory text before/after card
        4. Detects card type
        5. Generates and injects correct manifest
        6. Ensures proper formatting

        Args:
            apf_html: Raw APF HTML from LLM
            metadata: Note metadata
            manifest: Card manifest

        Returns:
            Post-processed APF HTML
        """
        # 1. Strip markdown code fences if present
        apf_html = re.sub(r"^```html\s*\n", "", apf_html, flags=re.MULTILINE)
        apf_html = re.sub(r"\n```\s*$", "", apf_html, flags=re.MULTILINE)

        # 2. Strip any text before PROMPT_VERSION or Card comment
        version_start = apf_html.find("<!-- PROMPT_VERSION:")
        card_start = apf_html.find("<!-- Card")

        if version_start >= 0:
            # Has PROMPT_VERSION, strip everything before it
            if version_start > 0:
                logger.debug(
                    "stripped_text_before_version",
                    slug=manifest.slug,
                    chars_removed=version_start,
                )
                apf_html = apf_html[version_start:]
        elif card_start > 0:
            # No PROMPT_VERSION, strip everything before Card
            logger.debug(
                "stripped_text_before_card",
                slug=manifest.slug,
                chars_removed=card_start,
            )
            apf_html = apf_html[card_start:]

        # 3. Strip any text after END_OF_CARDS or manifest comment
        end_of_cards_pos = apf_html.find("END_OF_CARDS")
        if end_of_cards_pos >= 0:
            # Keep until end of END_OF_CARDS line
            apf_html = apf_html[: end_of_cards_pos + len("END_OF_CARDS")]
        else:
            # Fallback: Strip after manifest comment
            manifest_match = re.search(r"(<!-- manifest:.*?-->)", apf_html, re.DOTALL)
            if manifest_match:
                end_pos = manifest_match.end()
                apf_html = apf_html[:end_pos]

        # 4. Extract tags from card header
        tags_match = re.search(r"Tags:\s*([^\]]+?)\s*-->", apf_html)
        if tags_match:
            # Use tags from model output
            tags = tags_match.group(1).strip().split()
            logger.debug("extracted_tags_from_output", slug=manifest.slug, tags=tags)
        else:
            # Generate tags deterministically
            tags = self._generate_tags(metadata, manifest.lang)
            logger.debug("generated_tags", slug=manifest.slug, tags=tags)

        # 5. Detect card type
        if "{{c" in apf_html:
            card_type = "Missing"
        elif "<img " in apf_html and "svg" in apf_html.lower():
            card_type = "Draw"
        else:
            card_type = "Simple"

        logger.debug("detected_card_type", slug=manifest.slug, type=card_type)

        # 6. Generate correct manifest
        correct_manifest = self._generate_manifest(manifest, card_type, tags)

        # 7. Replace existing manifest or append
        if "<!-- manifest:" in apf_html:
            apf_html = re.sub(
                r"<!-- manifest:.*?-->", correct_manifest, apf_html, flags=re.DOTALL
            )
            logger.debug("replaced_manifest", slug=manifest.slug)
        else:
            apf_html += "\n\n" + correct_manifest
            logger.debug("appended_manifest", slug=manifest.slug)

        # 8. Ensure APF v2.1 wrapper sentinels are present
        has_prompt_version = "<!-- PROMPT_VERSION: apf-v2.1 -->" in apf_html
        has_begin_cards = "<!-- BEGIN_CARDS -->" in apf_html
        has_end_cards = "<!-- END_CARDS -->" in apf_html
        has_end_of_cards = "END_OF_CARDS" in apf_html

        if not (
            has_prompt_version
            and has_begin_cards
            and has_end_cards
            and has_end_of_cards
        ):
            # Missing wrapper sentinels, add them
            logger.debug("adding_missing_wrapper_sentinels", slug=manifest.slug)

            # Wrap the card content
            lines = []
            if not has_prompt_version:
                lines.append("<!-- PROMPT_VERSION: apf-v2.1 -->")
            if not has_begin_cards:
                lines.append("<!-- BEGIN_CARDS -->")
                lines.append("")

            lines.append(apf_html.strip())

            if not has_end_cards:
                lines.append("")
                lines.append("<!-- END_CARDS -->")
            if not has_end_of_cards:
                lines.append("END_OF_CARDS")

            apf_html = "\n".join(lines)

        # 9. Normalize card header to match validator expectations
        apf_html = self._normalize_card_header(apf_html, manifest, card_type, tags)

        # 10. Fix HTML validation issues: wrap standalone <code> in <pre><code>
        apf_html = self._fix_code_tags(apf_html)

        # 11. Final HTML validation and auto-fix using structured templates
        apf_html = self._validate_and_fix_html(apf_html, manifest)

        # 12. Ensure proper formatting
        apf_html = apf_html.strip()

        return apf_html

    def _fix_card_headers(self, html: str, manifest: Manifest) -> str:
        """Fix card headers to ensure correct format.

        Args:
            html: APF HTML
            manifest: Card manifest

        Returns:
            HTML with fixed card headers
        """
        import re

        # Find all card headers
        header_pattern = r"(<!--\s*Card\s+(\d+)\s*\|.*?-->)"
        matches = list(re.finditer(header_pattern, html))

        for match in reversed(matches):
            header = match.group(1)
            card_num = match.group(2)

            # Check if header is valid
            valid_pattern = r"<!--\s*Card\s+\d+\s*\|\s*slug:\s*[a-z0-9-]+\s*\|\s*CardType:\s*(Simple|Missing|Draw)\s*\|\s*Tags:\s*[^>]+\s*-->"
            if not re.match(valid_pattern, header):
                # Fix the header
                # Extract components
                slug_match = re.search(r"slug:\s*([^\s|]+)", header)
                slug = slug_match.group(1) if slug_match else manifest.slug

                card_type_match = re.search(r"CardType:\s*(\w+)", header, re.IGNORECASE)
                card_type = (
                    card_type_match.group(1).capitalize()
                    if card_type_match
                    else "Simple"
                )
                if card_type not in ["Simple", "Missing", "Draw"]:
                    card_type = "Simple"

                tags_match = re.search(r"Tags:\s*([^>]+)", header)
                if tags_match:
                    tags_str = tags_match.group(1).strip()
                    # Fix tag format: space-separated, lowercase
                    tags = tags_str.split()
                    fixed_tags = []
                    for tag in tags:
                        # Convert to snake_case
                        fixed_tag = re.sub(r"[^a-z0-9_]", "_", tag.lower())
                        fixed_tag = re.sub(r"_+", "_", fixed_tag).strip("_")
                        if fixed_tag:
                            fixed_tags.append(fixed_tag)
                    tags_str = " ".join(fixed_tags[:6])  # Limit to 6 tags
                else:
                    tags_str = " ".join(manifest.tags[:6])

                # Build correct header
                fixed_header = f"<!-- Card {card_num} | slug: {slug} | CardType: {card_type} | Tags: {tags_str} -->"
                html = html[: match.start()] + fixed_header + html[match.end() :]
                logger.debug(
                    "pre_validation_fix_card_header", card_num=card_num, slug=slug
                )

        return html

    def _ensure_manifest(self, html: str, manifest: Manifest) -> str:
        """Ensure manifest is present and valid.

        Args:
            html: APF HTML
            manifest: Card manifest

        Returns:
            HTML with valid manifest
        """
        import json
        import re

        # Check if manifest exists
        manifest_pattern = r"<!--\s*manifest:\s*({.*?})\s*-->"
        manifest_match = re.search(manifest_pattern, html, re.DOTALL)

        if not manifest_match:
            # Add manifest before END_CARDS
            manifest_data = {
                "slug": manifest.slug,
                "lang": manifest.lang,
                "type": manifest.type,
                "tags": manifest.tags,
            }
            manifest_json = json.dumps(manifest_data, separators=(",", ":"))
            manifest_comment = f"<!-- manifest: {manifest_json} -->"

            if "<!-- END_CARDS -->" in html:
                html = html.replace(
                    "<!-- END_CARDS -->", f"{manifest_comment}\n<!-- END_CARDS -->"
                )
            else:
                html += f"\n{manifest_comment}"
            logger.debug("pre_validation_fix_added_manifest", slug=manifest.slug)
        else:
            # Validate manifest JSON
            try:
                manifest_json_str = manifest_match.group(1)
                manifest_data = json.loads(manifest_json_str)
                # Check if slug matches
                if manifest_data.get("slug") != manifest.slug:
                    # Fix manifest slug
                    manifest_data["slug"] = manifest.slug
                    manifest_data["lang"] = manifest.lang
                    manifest_data["type"] = manifest.type
                    manifest_data["tags"] = manifest.tags
                    new_manifest_json = json.dumps(manifest_data, separators=(",", ":"))
                    new_manifest = f"<!-- manifest: {new_manifest_json} -->"
                    html = re.sub(manifest_pattern, new_manifest, html, flags=re.DOTALL)
                    logger.debug("pre_validation_fix_manifest_slug", slug=manifest.slug)
            except json.JSONDecodeError:
                # Invalid JSON, replace it
                manifest_data = {
                    "slug": manifest.slug,
                    "lang": manifest.lang,
                    "type": manifest.type,
                    "tags": manifest.tags,
                }
                new_manifest_json = json.dumps(manifest_data, separators=(",", ":"))
                new_manifest = f"<!-- manifest: {new_manifest_json} -->"
                html = re.sub(manifest_pattern, new_manifest, html, flags=re.DOTALL)
                logger.debug("pre_validation_fix_manifest_json", slug=manifest.slug)

        return html

    def _fix_code_tags(self, html: str) -> str:
        """Fix HTML validation issues by wrapping standalone <code> in <pre><code>.

        Args:
            html: HTML content to fix

        Returns:
            Fixed HTML with all code blocks properly wrapped
        """
        import re

        # Match <code>...</code> tags (non-greedy to match individual tags)
        fixed_html = html
        code_pattern = r"<code(?:\s[^>]*)?>.*?</code>"
        matches = list(re.finditer(code_pattern, fixed_html, re.DOTALL | re.IGNORECASE))

        # Process matches in reverse order to avoid offset issues
        for match in reversed(matches):
            code_tag = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            # Check if this code tag is already inside a <pre> tag
            # Look backwards for <pre> tag
            context_before = fixed_html[max(0, start_pos - 500) : start_pos]
            # Find the last <pre> tag before this code
            pre_matches = list(
                re.finditer(r"<pre(?:\s[^>]*)?>", context_before, re.IGNORECASE)
            )
            if pre_matches:
                last_pre = pre_matches[-1]
                pre_start = last_pre.start() + (start_pos - 500)
                # Check if there's a closing </pre> between <pre> and our <code>
                between = fixed_html[pre_start:start_pos]
                if "</pre>" not in between and "</PRE>" not in between:
                    # <pre> is still open, this code is already wrapped
                    continue

            # This is a standalone <code> tag, wrap it
            wrapped = f"<pre>{code_tag}</pre>"
            fixed_html = fixed_html[:start_pos] + wrapped + fixed_html[end_pos:]

        return fixed_html

    def _validate_and_fix_html(self, apf_html: str, manifest: Manifest) -> str:
        """Validate and auto-fix HTML using structured generation approach.

        Args:
            apf_html: APF HTML content to validate and fix
            manifest: Card manifest for context

        Returns:
            Validated and fixed HTML
        """
        from ..apf.html_validator import validate_card_html

        # Validate the HTML
        validation_errors = validate_card_html(apf_html)

        if not validation_errors:
            # HTML is already valid
            return apf_html

        logger.debug(
            "html_validation_errors_found", slug=manifest.slug, errors=validation_errors
        )

        # Attempt auto-fix using the HTML generator
        html_generator = HTMLTemplateGenerator()

        # Try to extract card data and regenerate
        try:
            # Extract basic card data from the HTML
            card_data = self._extract_card_data_from_html(apf_html, manifest)

            if card_data:
                # Regenerate HTML using structured templates
                result = html_generator.generate_card_html(card_data)

                if result.is_valid:
                    logger.info(
                        "html_auto_fixed_with_templates",
                        slug=manifest.slug,
                        original_errors=len(validation_errors),
                        remaining_warnings=len(result.warnings),
                    )

                    if result.warnings:
                        for warning in result.warnings:
                            logger.warning(
                                "html_fix_warning", warning=warning, slug=manifest.slug
                            )

                    return result.html
                else:
                    logger.warning(
                        "html_template_regeneration_failed",
                        slug=manifest.slug,
                        errors=result.validation_errors,
                    )

        except Exception as e:
            logger.warning(
                "html_auto_fix_extraction_failed", slug=manifest.slug, error=str(e)
            )

        # Fallback: apply basic fixes
        fixed_html = self._apply_basic_html_fixes(apf_html, validation_errors)

        logger.debug(
            "applied_basic_html_fixes",
            slug=manifest.slug,
            original_errors=len(validation_errors),
        )

        return fixed_html

    def _extract_card_data_from_html(self, apf_html: str, manifest: Manifest) -> dict:
        """Extract card data from existing HTML for regeneration."""
        import re

        card_data = {
            "card_index": 1,
            "slug": manifest.slug,
            "tags": manifest.tags,
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

    def _apply_basic_html_fixes(self, html: str, errors: list[str]) -> str:
        """Apply basic HTML fixes for common validation errors."""
        fixed_html = html

        for error in errors:
            if "language- class" in error:
                # Add default language class to code elements without one
                fixed_html = re.sub(
                    r"<code(?![^>]*class=)",
                    r'<code class="language-text"',
                    fixed_html,
                    flags=re.IGNORECASE,
                )
            elif "wrap in <pre><code>" in error:
                # This is complex to fix automatically, log for manual review
                logger.debug("complex_html_fix_needed", error=error)

        return fixed_html

    def _normalize_card_header(
        self, apf_html: str, manifest: Manifest, card_type: str, tags: list[str]
    ) -> str:
        """Normalize card header to match validator's expected format.

        The validator expects exactly:
        <!-- Card N | slug: slug-name | CardType: Simple | Tags: tag1 tag2 tag3 -->

        Args:
            apf_html: APF HTML content
            manifest: Card manifest
            card_type: Card type (Simple, Missing, Draw)
            tags: List of tags

        Returns:
            APF HTML with normalized card header
        """
        # Build the correct header format
        correct_header = f"<!-- Card {manifest.card_index} | slug: {manifest.slug} | CardType: {card_type} | Tags: {' '.join(tags)} -->"

        # Find and replace existing card header
        # Pattern matches various card header formats the LLM might produce
        card_header_pattern = r"<!--\s*Card\s+\d+[^\]]*?-->"

        match = re.search(card_header_pattern, apf_html)
        if match:
            apf_html = (
                apf_html[: match.start()] + correct_header + apf_html[match.end() :]
            )
            logger.debug(
                "normalized_card_header",
                slug=manifest.slug,
                old_header=match.group(0)[:100],
                new_header=correct_header,
            )
        else:
            logger.warning(
                "no_card_header_found_to_normalize",
                slug=manifest.slug,
                inserting_header=True,
            )
            # If no header found, insert it after BEGIN_CARDS
            begin_cards_pos = apf_html.find("<!-- BEGIN_CARDS -->")
            if begin_cards_pos >= 0:
                insert_pos = begin_cards_pos + len("<!-- BEGIN_CARDS -->")
                apf_html = (
                    apf_html[:insert_pos]
                    + "\n\n"
                    + correct_header
                    + apf_html[insert_pos:]
                )

        return apf_html
