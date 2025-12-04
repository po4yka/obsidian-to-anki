"""APF card generation via OpenRouter LLM."""  # pragma: allowlist secret

import hashlib
import html
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from obsidian_anki_sync.config import Config
from obsidian_anki_sync.exceptions import APFValidationError  # pragma: allowlist secret
from obsidian_anki_sync.models import Card, Manifest, NoteMetadata, QAPair
from obsidian_anki_sync.providers.openrouter import OpenRouterProvider
from obsidian_anki_sync.utils.code_detection import detect_code_language_from_metadata
from obsidian_anki_sync.utils.content_hash import compute_content_hash
from obsidian_anki_sync.utils.llm_logging import (
    log_llm_stream_chunk,
    log_slow_llm_request,
)
from obsidian_anki_sync.utils.logging import get_logger
from obsidian_anki_sync.utils.retry import retry

if TYPE_CHECKING:
    from obsidian_anki_sync.providers.openrouter.provider import OpenRouterStreamResult

logger = get_logger(__name__)


class APFGenerator:
    """Generate APF cards using OpenRouter LLM."""

    MAX_VALIDATION_RETRIES = 2  # Number of times to ask LLM to fix validation errors

    def __init__(self, config: Config):
        """Initialize the generator."""
        self.config = config
        self.provider = OpenRouterProvider(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            site_url=config.openrouter_site_url,
            site_name=config.openrouter_site_name,
            timeout=config.llm_timeout,
            max_tokens=config.llm_max_tokens,
        )
        self._prompt_fingerprints: dict[str, int] = {}
        self._slow_request_threshold = config.llm_slow_request_threshold

        # Load CARDS_PROMPT template
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = "Generate APF cards following strict APF format."

    def _get_generator_model(self) -> str:
        """Resolve the generator model using config overrides with fallbacks."""
        model_name = getattr(self.config, "generator_model", "") or ""
        if model_name:
            return model_name

        resolve_model = getattr(self.config, "get_model_for_agent", None)
        if callable(resolve_model):
            resolved = resolve_model("generator")
            if resolved:
                return resolved

        legacy = getattr(self.config, "openrouter_model", "") or ""
        if legacy:
            return legacy

        default_model = getattr(self.config, "default_llm_model", "") or ""
        if default_model:
            return default_model

        msg = "No generator model configured for APF generation"
        raise ValueError(msg)

    @retry(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(httpx.HTTPError, TimeoutError, ConnectionError),
    )
    def generate_card(
        self,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> Card:
        """
        Generate an APF card for a Q/A pair in a specific language.

        Args:
            qa_pair: Q/A pair content
            metadata: Note metadata
            manifest: Card manifest
            lang: Language code (en, ru)

        Returns:
            Generated card
        """
        from .html_validator import validate_card_html
        from .linter import validate_apf

        # Select language-specific content
        question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

        # Build user prompt
        user_prompt = self._build_user_prompt(
            question, answer, qa_pair, metadata, manifest, lang
        )

        # Conversation transcript (system + user, with optional feedback turns)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        apf_html = None
        validation_errors = []

        # Generation loop with validation retries
        for attempt in range(1 + self.MAX_VALIDATION_RETRIES):
            # Call LLM
            model_name = self._get_generator_model()
            logger.debug(
                "calling_llm",
                model=model_name,
                temp=self.config.llm_temperature,
                slug=manifest.slug,
                attempt=attempt + 1,
            )

            apf_html = self._invoke_llm(messages, manifest, model_name)

            # Normalize Markdown code fences (if any) into HTML blocks
            default_lang = self._code_language_hint(metadata)
            apf_html = self._normalize_code_blocks(apf_html, default_lang)

            # Validate the generated HTML
            html_errors = validate_card_html(apf_html)
            linter_result = validate_apf(apf_html, manifest.slug)
            validation_errors = html_errors + linter_result.errors

            if not validation_errors:
                # Validation passed
                logger.debug(
                    "validation_passed", slug=manifest.slug, attempt=attempt + 1
                )
                break

            # If this was the last attempt, fail fast
            if attempt >= self.MAX_VALIDATION_RETRIES:
                logger.error(
                    "validation_failed_after_retries",
                    slug=manifest.slug,
                    errors=validation_errors,
                    attempts=attempt + 1,
                )
                msg = (
                    f"APF validation failed for {manifest.slug} after "
                    f"{attempt + 1} attempts: {validation_errors[0]}"
                )
                raise APFValidationError(
                    msg,
                    slug=manifest.slug,
                    validation_errors=validation_errors,
                    attempts=attempt + 1,
                    suggestion=(
                        "Review the Q&A content for formatting issues, "
                        "ensure code blocks are properly fenced, "
                        "and check for balanced HTML tags."
                    ),
                )

            # Ask LLM to fix the errors
            logger.info(
                "validation_retry",
                slug=manifest.slug,
                errors=validation_errors,
                attempt=attempt + 1,
            )

            # Add the assistant's response and error feedback to messages
            messages.append({"role": "assistant", "content": apf_html})
            messages.append(
                {
                    "role": "user",
                    "content": self._build_fix_prompt(validation_errors),
                }
            )

        # Compute content hash
        content_hash = compute_content_hash(qa_pair, metadata, lang)

        # Determine note type (Simple by default, could be detected from metadata)
        note_type = self._determine_note_type(metadata, apf_html)

        # Extract tags from metadata
        tags = self._extract_tags(metadata, lang)

        # Ensure manifest comment is accurate
        apf_html = self._ensure_manifest(apf_html, manifest, tags, note_type)

        return Card(
            slug=manifest.slug,
            lang=lang,
            apf_html=apf_html,
            manifest=manifest,
            content_hash=content_hash,
            note_type=note_type,
            tags=tags,
            guid=manifest.guid,
        )

    def generate_cards(
        self,
        qa_pairs: list[QAPair],
        metadata: NoteMetadata,
        manifests: list[Manifest],
        lang: str,
    ) -> list[Card]:
        """
        Generate multiple APF cards in a single batch request.

        This method groups cards by note and language, generating them
        together to reduce API calls. Falls back to individual generation
        if batch fails.

        Args:
            qa_pairs: List of Q/A pairs to generate cards for
            metadata: Note metadata (shared across all cards)
            manifests: List of manifests (one per Q/A pair)
            lang: Language code (en, ru)

        Returns:
            List of generated cards
        """
        if len(qa_pairs) == 1:
            # Single card - use individual method
            return [self.generate_card(qa_pairs[0], metadata, manifests[0], lang)]

        if len(qa_pairs) != len(manifests):
            msg = "qa_pairs and manifests must have same length"
            raise ValueError(msg)

        logger.info(
            "batch_generation_start",
            count=len(qa_pairs),
            lang=lang,
            note=metadata.title,
        )

        # Try batch generation
        try:
            return self._generate_cards_batch(qa_pairs, metadata, manifests, lang)
        except Exception as e:
            logger.warning(
                "batch_generation_failed_fallback",
                error=str(e),
                falling_back_to_individual=True,
            )
            # Fall back to individual generation
            cards = []
            for qa_pair, manifest in zip(qa_pairs, manifests):
                try:
                    card = self.generate_card(qa_pair, metadata, manifest, lang)
                    cards.append(card)
                except Exception as card_error:
                    logger.error(
                        "individual_card_generation_failed",
                        slug=manifest.slug,
                        error=str(card_error),
                    )
                    # Continue with other cards
            return cards

    def _generate_cards_batch(
        self,
        qa_pairs: list[QAPair],
        metadata: NoteMetadata,
        manifests: list[Manifest],
        lang: str,
    ) -> list[Card]:
        """Generate multiple cards in a single LLM call."""
        # Build batch prompt
        user_prompt = self._build_batch_prompt(qa_pairs, metadata, manifests, lang)

        model_name = self._get_generator_model()
        logger.debug(
            "calling_llm_batch",
            model=model_name,
            temp=self.config.llm_temperature,
            card_count=len(qa_pairs),
        )

        try:
            result = self.provider.generate(
                model=model_name,
                prompt=user_prompt,
                system=self.system_prompt,
                temperature=self.config.llm_temperature,
                stream=False,
                reasoning_enabled=self.config.llm_reasoning_enabled,
                reasoning_effort=self.config.get_reasoning_effort("generation"),
            )

            apf_html_batch = result.get("response")

            if not apf_html_batch:
                msg = "LLM returned empty response"
                raise ValueError(msg)

            # Parse batch response (expects multiple card blocks)
            cards = self._parse_batch_response(
                apf_html_batch, qa_pairs, metadata, manifests, lang
            )

            logger.info("batch_generation_success", cards_generated=len(cards))
            return cards

        except Exception as e:
            logger.error("batch_llm_call_failed", error=str(e))
            raise

    def _build_batch_prompt(
        self,
        qa_pairs: list[QAPair],
        metadata: NoteMetadata,
        manifests: list[Manifest],
        lang: str,
    ) -> str:
        """Build user prompt for batch card generation."""
        prompt = f"""Generate {len(qa_pairs)} APF cards in HTML format.

Metadata (shared for all cards):
- Topic: {metadata.topic}
- Subtopics: {", ".join(metadata.subtopics) if metadata.subtopics else "None"}
- Difficulty: {metadata.difficulty or "Not specified"}
- Language: {lang}

Cards to generate:
"""

        for i, (qa_pair, manifest) in enumerate(zip(qa_pairs, manifests), 1):
            question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
            answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

            prompt += f"""
Card {i} (Slug: {manifest.slug}):
Question: {question[:200]}{"..." if len(question) > 200 else ""}
Answer: {answer[:300]}{"..." if len(answer) > 300 else ""}
"""

        prompt += f"""
Requirements:
- Generate {len(qa_pairs)} complete APF card HTML blocks
- Each card must be separated by <!-- CARD_SEPARATOR -->
- CardType: Simple (or Missing if {{{{c}}}} detected, or Draw if diagram marker present)
- Tags: Derive from topic/subtopics, 3-6 snake_case tags, include primary language/tech
- Primary language tag: {lang}
- Topic-based tags: {metadata.topic.lower().replace(" ", "_")}
- Include manifest at end of each card with correct slug
- Follow APF v2.1 format strictly
- Output ONLY the card HTML blocks separated by <!-- CARD_SEPARATOR -->, no explanations
"""

        return prompt

    def _parse_batch_response(
        self,
        apf_html_batch: str,
        qa_pairs: list[QAPair],
        metadata: NoteMetadata,
        manifests: list[Manifest],
        lang: str,
    ) -> list[Card]:
        """Parse batch response into individual cards."""
        # Split by separator
        card_blocks = apf_html_batch.split("<!-- CARD_SEPARATOR -->")

        # If no separator, try to split by BEGIN_CARDS markers
        if len(card_blocks) == 1:
            card_blocks = re.split(r"<!-- BEGIN_CARDS -->", apf_html_batch)
            # Remove first empty block if present
            if card_blocks and not card_blocks[0].strip():
                card_blocks = card_blocks[1:]

        cards = []
        default_lang = self._code_language_hint(metadata)

        for i, (card_html, qa_pair, manifest) in enumerate(
            zip(card_blocks, qa_pairs, manifests)
        ):
            try:
                # Clean up the HTML block
                card_html = card_html.strip()

                # Normalize code blocks
                card_html = self._normalize_code_blocks(card_html, default_lang)

                # Compute content hash
                content_hash = compute_content_hash(qa_pair, metadata, lang)

                # Determine note type
                note_type = self._determine_note_type(metadata, card_html)

                # Extract tags
                tags = self._extract_tags(metadata, lang)

                # Ensure manifest is correct
                card_html = self._ensure_manifest(card_html, manifest, tags, note_type)

                cards.append(
                    Card(
                        slug=manifest.slug,
                        lang=lang,
                        apf_html=card_html,
                        manifest=manifest,
                        content_hash=content_hash,
                        note_type=note_type,
                        tags=tags,
                        guid=manifest.guid,
                    )
                )

            except Exception as e:
                logger.error(
                    "batch_card_parse_failed",
                    index=i,
                    slug=manifest.slug,
                    error=str(e),
                )
                # Skip this card, will be regenerated individually if needed
                continue

        if len(cards) != len(qa_pairs):
            msg = (
                f"Expected {len(qa_pairs)} cards, got {len(cards)} from batch response"
            )
            raise ValueError(msg)

        return cards

    def _invoke_llm(
        self, messages: list[dict[str, str]], manifest: Manifest, model: str
    ) -> str:
        """Call OpenRouter (streaming if enabled) and return HTML."""
        stream_enabled = self.config.llm_streaming_enabled
        system_prompt = (
            messages[0]["content"]
            if messages and messages[0]["role"] == "system"
            else self.system_prompt
        )
        conversation_prompt = self._serialize_messages_for_prompt(messages[1:])
        reasoning_effort = self.config.get_reasoning_effort("generation")

        fingerprint, fingerprint_count = self._record_prompt_signature(
            conversation_prompt
        )
        if fingerprint_count > 1:
            logger.info(
                "apf_prompt_cache_hit",
                slug=manifest.slug,
                fingerprint=fingerprint[:12],
                seen=fingerprint_count,
            )

        request_start = time.perf_counter()

        result = self.provider.generate(
            model=model,
            prompt=conversation_prompt,
            system=system_prompt,
            temperature=self.config.llm_temperature,
            stream=stream_enabled,
            reasoning_enabled=self.config.llm_reasoning_enabled,
            reasoning_effort=reasoning_effort,
        )

        if stream_enabled:
            stream = result.get("stream")
            if stream is None:
                msg = "Streaming was requested but provider returned no stream handle"
                raise RuntimeError(msg)
            apf_html = self._consume_stream(
                stream=stream,
                slug=manifest.slug,
                model=model,
                start_time=request_start,
            )

            duration = time.perf_counter() - request_start
            log_slow_llm_request(
                duration_seconds=duration,
                threshold_seconds=self._slow_request_threshold,
                model=model,
                operation="apf_card_generation_stream",
            )
            return apf_html

        apf_html = result.get("response")
        if not apf_html:
            msg = "OpenRouter returned empty response"
            raise ValueError(msg)

        duration = time.perf_counter() - request_start
        log_slow_llm_request(
            duration_seconds=duration,
            threshold_seconds=self._slow_request_threshold,
            model=model,
            operation="apf_card_generation",
        )
        return apf_html

    def _consume_stream(
        self,
        stream: "OpenRouterStreamResult",
        slug: str,
        model: str,
        start_time: float | None = None,
    ) -> str:
        """Iterate over streaming chunks while logging telemetry."""
        start_time = start_time or time.time()
        for chunk_index, chunk in enumerate(stream, start=1):
            elapsed = time.time() - start_time
            chunk_text = chunk if isinstance(chunk, str) else str(chunk)
            log_llm_stream_chunk(
                model=model,
                operation="apf_card_generation",
                chunk_index=chunk_index,
                elapsed_seconds=elapsed,
                chunk_chars=len(chunk_text),
                chunk_preview=chunk_text,
                slug=slug,
            )
        return stream.collect()

    @staticmethod
    def _serialize_messages_for_prompt(messages: list[dict[str, str]]) -> str:
        """Flatten a chat transcript into a single string prompt."""
        if not messages:
            return ""
        parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def _record_prompt_signature(self, prompt: str) -> tuple[str, int]:
        """Record prompt fingerprint to estimate cache hit potential."""
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        count = self._prompt_fingerprints.get(digest, 0) + 1
        self._prompt_fingerprints[digest] = count
        return digest, count

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        qa_pair: QAPair,
        metadata: NoteMetadata,
        manifest: Manifest,
        lang: str,
    ) -> str:
        """Build user prompt for LLM."""
        # Build reference link
        ref_link = f"[[{manifest.source_path}#{manifest.source_anchor}]]"

        prompt = f"""Generate an APF card in HTML format.

Metadata:
- Topic: {metadata.topic}
- Subtopics: {", ".join(metadata.subtopics) if metadata.subtopics else "None"}
- Difficulty: {metadata.difficulty or "Not specified"}
- Language: {lang}
- Slug: {manifest.slug}

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

        # Add requirements
        prompt += f"""
Requirements:
- CardType: Simple (or Missing if {{{{c}}}} detected in answer, or Draw if diagram marker present)
- Tags: EXACTLY 3-6 snake_case tags with underscores (e.g., by_keyword NOT by-keyword), include primary language/tech
- Primary language tag: {lang}
- Topic-based tags: {metadata.topic.lower().replace(" ", "_")}
- Use EXACT slug "{manifest.slug}" in both card header AND manifest (do not create your own slug)
- Add "Ref: {ref_link}" in Other notes section
- Follow APF v2.1 format strictly
- Output ONLY the card HTML, no explanations
- ALL cards MUST be wrapped with `<!-- BEGIN_CARDS -->` and `<!-- END_CARDS -->` markers.
- End with END_OF_CARDS on its own line

SPACED REPETITION PRINCIPLES (apply strictly):
- ONE CARD = ONE ATOMIC FACT. If testing multiple things, you're doing it wrong.
- ACTIVE RECALL: Force retrieval, not recognition. Avoid yes/no questions.
  BAD: "Is Flow cold?" GOOD: "What is Flow's default emission behavior?"
- NO SPOILERS: Title must NOT reveal the answer.
  BAD: "Use Kotlin's by keyword" GOOD: "Implement Class Delegation for Repository"
- MEANINGFUL CLOZE: Hide the CONCEPT, not trivial syntax.
  BAD: {{{{c1::fun}}}} getData() GOOD: fun getData(): {{{{c1::Flow}}}}<Data>
- CONTEXT INDEPENDENCE: Card must be understandable standalone, even after 6 months.
- CONCRETE EXAMPLES: Show real-world usage, not toy examples.
- WHY IT MATTERS: Include "You need this when..." motivation.

CARD QUALITY REQUIREMENTS:
- Code: Keep focused on the concept being tested, remove irrelevant implementation details
- Key point notes: Include 5-7 detailed bullets covering:
  * WHY it works (underlying mechanism, how compiler/runtime handles it)
  * WHEN to use (practical use cases, "You need this when...")
  * CONSTRAINTS and limitations
  * COMMON MISTAKES developers make
  * COMPARISON with alternatives ("Unlike X, this does Y because...")
  * PERFORMANCE implications (if relevant)

CRITICAL FORMATTING (violations will cause rejection):
- Output is HTML, NOT Markdown. NEVER use **bold** or *italic* markdown - use <strong> and <em> HTML tags
- ALL code MUST use <pre><code class="language-{self._code_language_hint(metadata)}">...</code></pre> blocks
- NEVER use markdown backtick fences (```)
- NEVER use inline <code> tags outside <pre> - use <strong> for inline tokens instead
- Code blocks are for ACTUAL CODE ONLY - NEVER use // comments to write explanatory text
- Explanations go in Key point notes as <ul><li> bullets, NOT as code comments
- MUST include exact field headers: <!-- Title -->, <!-- Key point (code block / image) -->, <!-- Key point notes -->
- Field headers must appear EXACTLY as specified in the template
"""

        return prompt

    def _build_fix_prompt(self, errors: list[str]) -> str:
        """Build a prompt asking the LLM to fix validation errors."""
        error_list = "\n".join(f"- {e}" for e in errors)
        return f"""Your card has validation errors that must be fixed:

{error_list}

Please regenerate the COMPLETE card with these errors fixed. Output the full corrected card HTML, not just the fixes.

FORMATTING RULES:
- Use HTML, NOT Markdown (no **bold**, use <strong>)
- Code blocks use <pre><code class="language-X">, NOT backticks
- Code blocks contain ONLY actual code, NOT explanatory comments
- Explanations go in Key point notes as <ul><li> bullets
- Include all required field headers (<!-- Title -->, <!-- Key point -->, etc.)
- End with END_OF_CARDS on its own line

QUALITY RULES:
- Title should NOT reveal the answer (avoid "spoiler effect")
- Key point notes should have 5-7 detailed bullets covering WHY, WHEN, CONSTRAINTS, COMMON MISTAKES
- Code should be focused on the concept, not cluttered with unrelated implementation"""

    def _determine_note_type(self, metadata: NoteMetadata, apf_html: str) -> str:
        """Determine note type from metadata and content."""
        # Check frontmatter first
        if metadata.anki_note_type:
            return metadata.anki_note_type

        # Check for cloze markers in HTML
        if "{{c" in apf_html:
            return "APF::Missing (Cloze)"

        # Check for draw marker
        if "<!-- DRAW_CARD -->" in apf_html or "CardType: Draw" in apf_html:
            return "APF::Draw"

        # Default
        return "APF::Simple"

    def _ensure_manifest(
        self,
        apf_html: str,
        manifest: Manifest,
        tags: list[str],
        note_type: str,
    ) -> str:
        """Ensure manifest comment exists and contains required fields."""
        pattern = re.compile(r"<!--\s*manifest:\s*({.*?})\s*-->")
        match = pattern.search(apf_html)
        if not match:
            msg = "APF output missing manifest comment"
            raise ValueError(msg)

        try:
            manifest_data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            msg = f"Invalid manifest JSON: {exc}"
            raise ValueError(msg) from exc

        manifest_data.update(
            {
                "slug": manifest.slug,
                "slug_base": manifest.slug_base,
                "lang": manifest.lang,
                "guid": manifest.guid,
                "type": note_type,
                "tags": tags,
            }
        )

        new_comment = f"<!-- manifest: {json.dumps(manifest_data, ensure_ascii=False, separators=(',', ':'))} -->"
        start, end = match.span()
        return apf_html[:start] + new_comment + apf_html[end:]

    def _extract_tags(self, metadata: NoteMetadata, lang: str) -> list[str]:
        """Extract tags from metadata."""
        tags: set[str] = set()

        # Add language
        lang_tag = self._sanitize_tag(lang, lowercase=True)
        if lang_tag:
            tags.add(lang_tag)

        # Add topic
        topic_tag = self._sanitize_tag(metadata.topic, lowercase=True)
        if topic_tag:
            tags.add(topic_tag)

        # Add subtopics
        for subtopic in metadata.subtopics:
            subtopic_tag = self._sanitize_tag(subtopic, lowercase=True)
            if subtopic_tag:
                tags.add(subtopic_tag)

        # Add metadata tags
        for tag in metadata.tags:
            meta_tag = self._sanitize_tag(tag)
            if meta_tag:
                tags.add(meta_tag)

        return sorted(tags)

    def _code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks."""
        return detect_code_language_from_metadata(metadata)

    def _normalize_code_blocks(self, apf_html: str, default_lang: str) -> str:
        """Convert Markdown code fences to <pre><code> blocks if present.

        Includes iteration limit to prevent infinite loops on malformed input.
        """
        if "```" not in apf_html:
            return apf_html

        normalized_parts: list[str] = []
        cursor = 0
        default_class = self._format_code_language(default_lang)

        # Prevent infinite loops - max iterations based on content length
        max_iterations = len(apf_html) // 3 + 10  # At least 3 chars per fence
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            start = apf_html.find("```", cursor)
            if start == -1:
                normalized_parts.append(apf_html[cursor:])
                break

            end = apf_html.find("```", start + 3)
            if end == -1:
                # No closing fence; leave the rest untouched
                normalized_parts.append(apf_html[cursor:])
                break

            fence_content = apf_html[start + 3 : end]
            if "\n" in fence_content:
                lang_spec, code_body = fence_content.split("\n", 1)
            else:
                lang_spec, code_body = fence_content, ""

            lang = lang_spec.strip()
            lang_class = self._format_code_language(lang or default_class)

            # Preserve text before the fence
            normalized_parts.append(apf_html[cursor:start])

            code_text = code_body.replace("\r\n", "\n")
            code_text = code_text.rstrip("\n")
            escaped_code = html.escape(code_text, quote=False)

            normalized_parts.append(
                f'<pre><code class="language-{lang_class}">{escaped_code}'
                "\n</code></pre>"
            )

            new_cursor = end + 3
            if new_cursor <= cursor:
                # Cursor didn't advance - break to prevent infinite loop
                logger.warning(
                    "code_block_normalization_stuck",
                    cursor=cursor,
                    new_cursor=new_cursor,
                )
                normalized_parts.append(apf_html[cursor:])
                break
            cursor = new_cursor
        else:
            # Max iterations reached
            logger.warning(
                "code_block_normalization_max_iterations",
                iterations=iteration,
                content_length=len(apf_html),
            )
            # Append remaining content
            normalized_parts.append(apf_html[cursor:])

        return "".join(normalized_parts)

    def _format_code_language(self, lang: str) -> str:
        """Normalize language identifiers for code block classes."""
        if not lang:
            return "plaintext"
        normalized = lang.strip().lower().replace(" ", "-").replace("/", "-")
        normalized = re.sub(r"[^a-z0-9_\-+.]", "", normalized)
        normalized = normalized.removeprefix("language-")
        return normalized or "plaintext"

    def _sanitize_tag(self, tag: str | None, lowercase: bool = False) -> str:
        """Normalize tag strings for Anki compatibility."""
        if not tag:
            return ""

        text = str(tag).strip()
        if not text:
            return ""

        if lowercase:
            text = text.lower()

        text = text.replace(" ", "_").replace("/", "_")
        text = re.sub(r"[^A-Za-z0-9:_-]", "_", text)

        # Collapse consecutive underscores
        text = re.sub(r"_+", "_", text).strip("_")
        return text
