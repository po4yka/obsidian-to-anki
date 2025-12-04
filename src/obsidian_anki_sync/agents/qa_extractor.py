"""Q&A extraction agent using LLM for flexible note parsing.

This agent intelligently extracts question-answer pairs from notes
regardless of their markdown format, using LLM to understand structure.
"""

import re
import time
from pathlib import Path

from obsidian_anki_sync.exceptions import TruncationError
from obsidian_anki_sync.models import NoteMetadata, QAPair
from obsidian_anki_sync.providers.base import BaseLLMProvider
from obsidian_anki_sync.utils.llm_logging import log_slow_llm_request
from obsidian_anki_sync.utils.logging import get_logger

from .json_schemas import get_qa_extraction_schema
from .llm_errors import categorize_llm_error, format_llm_error_for_user, log_llm_error

logger = get_logger(__name__)


class QAExtractorAgent:
    """Agent for flexible Q&A extraction using LLM.

    Uses LLM to intelligently extract Q&A pairs from notes,
    supporting various markdown formats and structures.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "qwen3:8b",
        temperature: float = 0.0,
        reasoning_enabled: bool = False,
        enable_content_generation: bool = True,
        repair_missing_sections: bool = True,
        slow_request_threshold: float = 60.0,
    ):
        """Initialize Q&A extractor agent.

        Args:
            llm_provider: LLM provider instance
            model: Model to use for extraction
            temperature: Sampling temperature (0.0 for deterministic)
            reasoning_enabled: Enable reasoning mode for models that support it
            enable_content_generation: Allow LLM to generate missing content
            repair_missing_sections: Generate missing language sections
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        self.reasoning_enabled = reasoning_enabled
        self.enable_content_generation = enable_content_generation
        self.repair_missing_sections = repair_missing_sections
        self.slow_request_threshold = slow_request_threshold
        logger.info(
            "qa_extractor_agent_initialized",
            model=model,
            enable_content_generation=enable_content_generation,
            repair_missing_sections=repair_missing_sections,
        )

    def _build_extraction_prompt(
        self, note_content: str, metadata: NoteMetadata
    ) -> str:
        """Build extraction prompt for the LLM.

        Args:
            note_content: Full note content including frontmatter
            metadata: Parsed metadata

        Returns:
            Formatted prompt string
        """
        languages = metadata.language_tags
        language_list = ", ".join(languages)

        return f"""<task>
Extract all question-answer pairs from the provided educational note. Identify questions and their corresponding answers regardless of markdown format.

CRITICAL: Your response MUST be COMPLETE. Include ALL required fields for each Q&A pair with FULL answers. Do not truncate answers mid-sentence or mid-word. Ensure the JSON structure is valid and complete.
</task>

<input>
<metadata>
Title: {metadata.title}
Topic: {metadata.topic}
Languages: {language_list}
Expected Languages per Q&A: {language_list}
</metadata>

<note_content>
{note_content}
</note_content>
</input>

<rules>
1. ONLY extract Q&A pairs where BOTH the question AND answer are explicitly present in the note
2. SKIP sections labeled "Follow-up Questions", "Additional Questions", or any questions without answers
3. Do NOT create separate Q&A pairs for unanswered questions
4. Questions may appear in various formats:
   - Explicit headers: "# Question (EN)", "## What is...", "# Вопрос (RU)"
   - Numbered or bulleted lists
   - Q&A formatted blocks
   - Implicit questions within text
   - **IMPORTANT**: Questions and answers may be in SEPARATE sections:
     * Questions first: "# Question (EN)" and "# Вопрос (RU)" at the top
     * Answers later: "## Answer (EN)" and "## Ответ (RU)" below
     * In this case, pair the corresponding language versions together
5. Extract answers that directly correspond to each question
6. For bilingual notes (en, ru), extract BOTH language versions of questions and answers:
   - If questions appear together (e.g., "# Question (EN)" then "# Вопрос (RU)"), they belong to the SAME Q&A pair
   - If answers appear together (e.g., "## Answer (EN)" then "## Ответ (RU)"), they belong to the SAME Q&A pair
   - Pair question_en with answer_en and question_ru with answer_ru
   {"- If a language section is missing but the note specifies both languages in language_tags, GENERATE the missing section by translating from the existing language" if self.enable_content_generation and self.repair_missing_sections else ""}
7. Preserve the semantic relationship between questions and answers
8. Number all Q&A pairs sequentially starting from 1
9. Maintain the order of Q&A pairs as they appear in the note
10. Preserve markdown formatting in questions and answers
11. **CRITICAL**: When questions and answers are in separate sections, match them by semantic content and language, not just by position

<field_extraction>
For each Q&A pair, extract these fields:
- card_index: Sequential number (1, 2, 3, ...)
- question_en: Question in English (if 'en' in language_tags, otherwise empty string)
- question_ru: Question in Russian (if 'ru' in language_tags, otherwise empty string)
- answer_en: Answer in English (if 'en' in language_tags, otherwise empty string)
- answer_ru: Answer in Russian (if 'ru' in language_tags, otherwise empty string)
- context: Any contextual information before the Q&A (optional)
- followups: Follow-up questions mentioned (for reference only, NOT as separate cards)
- references: References or citations (optional)
- related: Related topics or questions (optional)
</field_extraction>

<constraints>
- NEVER extract Q&A pairs with missing questions or answers
- NEVER create cards from "Follow-up Questions" or "Additional Questions" sections
- If a language is not in language_tags, leave that field as empty string
- If no complete Q&A pairs exist, return empty list with explanation in extraction_notes
- Follow-up questions go in the 'followups' field, NOT as separate Q&A pairs
{"- When language_tags specifies multiple languages but content is missing for one language: GENERATE the missing content by translating from the existing language, preserving technical terms and formatting" if self.enable_content_generation and self.repair_missing_sections else ""}
</constraints>
</rules>

<output_format>
Respond with valid JSON matching this exact structure. Ensure ALL fields are populated:

{{
  "qa_pairs": [
    {{
      "card_index": 1,
      "question_en": "Question text in English",
      "question_ru": "Текст вопроса на русском",
      "answer_en": "Answer text in English - MUST be complete, not truncated",
      "answer_ru": "Текст ответа на русском - MUST be complete, not truncated",
      "context": "Optional context before Q&A",
      "followups": "Optional follow-up questions",
      "references": "Optional references",
      "related": "Optional related topics"
    }}
  ],
  "extraction_notes": "Notes about the extraction process or why certain pairs were skipped",
  "total_pairs": 1
}}

CRITICAL REQUIREMENTS:
- ALL answer fields (answer_en, answer_ru) MUST be complete and not truncated
- If a language is in language_tags, the corresponding answer field MUST contain the full answer
- Do not cut off answers mid-sentence or mid-word
- Ensure the JSON structure is complete and valid
- **For notes with separated Q&A sections**: When you see "# Question (EN)" and "# Вопрос (RU)" followed later by "## Answer (EN)" and "## Ответ (RU)", these form ONE Q&A pair with bilingual content
- **Pairing logic**: Match questions and answers by language (EN with EN, RU with RU) and semantic content
</output_format>

<examples>
<example_1>
Input note with explicit headers:
```
# Question (EN)
What is polymorphism?

## Answer (EN)
Polymorphism is the ability of objects to take on multiple forms.
```

Expected extraction:
{{
  "qa_pairs": [{{
    "card_index": 1,
    "question_en": "What is polymorphism?",
    "question_ru": "",
    "answer_en": "Polymorphism is the ability of objects to take on multiple forms.",
    "answer_ru": ""
  }}],
  "total_pairs": 1
}}
</example_1>

<example_2>
Input note with SEPARATED Q&A sections (questions first, answers later):
```
# Вопрос (RU)
Как работают BFS и DFS? Когда следует использовать каждый?

# Question (EN)
How do BFS and DFS work? When should you use each?

---

## Ответ (RU)
Графы — структуры данных...

## Answer (EN)
Graphs are data structures...
```

Expected extraction (ONE Q&A pair with bilingual content):
{{
  "qa_pairs": [{{
    "card_index": 1,
    "question_en": "How do BFS and DFS work? When should you use each?",
    "question_ru": "Как работают BFS и DFS? Когда следует использовать каждый?",
    "answer_en": "Graphs are data structures...",
    "answer_ru": "Графы — структуры данных..."
  }}],
  "total_pairs": 1
}}
</example_2>

<example_3>
Input note with unanswered follow-up section:
```
# Question (EN)
What is inheritance?

## Answer (EN)
Inheritance allows classes to derive properties from other classes.

## Follow-up Questions
- What about multiple inheritance?
```

Expected extraction:
{{
  "qa_pairs": [{{
    "card_index": 1,
    "question_en": "What is inheritance?",
    "answer_en": "Inheritance allows classes to derive properties from other classes.",
    "followups": "What about multiple inheritance?"
  }}],
  "extraction_notes": "Skipped 1 unanswered follow-up question",
  "total_pairs": 1
}}
</example_2>
</examples>
"""

    def _check_note_processable(
        self, note_content: str, file_path: Path | None = None
    ) -> tuple[bool, str, int, int, int]:
        """Check if note fits in model's context window.

        Only returns False (needs chunking) if the note + prompt + expected output
        exceeds the model's context window. Output token limits are handled by
        increasing max_tokens up to the model's maximum.

        Args:
            note_content: Full note content
            file_path: Optional file path for logging

        Returns:
            Tuple of (processable, reason, content_tokens, total_required, context_window)
        """
        from obsidian_anki_sync.providers.openrouter import (
            DEFAULT_CONTEXT_WINDOW,
            DEFAULT_MAX_OUTPUT_TOKENS,
            MODEL_CONTEXT_WINDOWS,
            MODEL_MAX_OUTPUT_TOKENS,
        )

        # Get model limits
        context_window = MODEL_CONTEXT_WINDOWS.get(self.model, DEFAULT_CONTEXT_WINDOW)
        model_max_output = MODEL_MAX_OUTPUT_TOKENS.get(
            self.model, DEFAULT_MAX_OUTPUT_TOKENS
        )

        # Rough estimate: 1 token ~= 4 characters
        content_tokens = len(note_content) // 4

        # Estimate prompt overhead (system prompt + extraction prompt template)
        # System prompt is ~500 tokens, template adds ~200 tokens
        prompt_overhead = 700

        # For bilingual QA extraction, output is typically 2-3x input content
        # (bilingual Q&A pairs + JSON structure)
        output_multiplier = 3.0
        estimated_output = int(content_tokens * output_multiplier)

        # Cap estimated output at model's max output limit
        estimated_output = min(estimated_output, model_max_output)

        # Total tokens needed: input content + prompt overhead + output
        total_input = content_tokens + prompt_overhead
        total_required = total_input + estimated_output

        # Use 95% of context window as safe threshold
        safe_context = int(context_window * 0.95)

        if total_required > safe_context:
            reason = (
                f"Note exceeds context window: ~{content_tokens} content tokens + "
                f"{prompt_overhead} prompt + ~{estimated_output} output = "
                f"~{total_required} total, but context window is {context_window}"
            )
            logger.info(
                "note_exceeds_context_window",
                file=str(file_path) if file_path else "unknown",
                content_tokens=content_tokens,
                prompt_overhead=prompt_overhead,
                estimated_output=estimated_output,
                total_required=total_required,
                context_window=context_window,
                model=self.model,
            )
            return False, reason, content_tokens, total_required, context_window

        return True, "", content_tokens, total_required, context_window

    def _chunk_note_content(
        self, content: str, max_chunk_tokens: int = 3000
    ) -> list[str]:
        """Split note into processable chunks by markdown sections.

        Splits on markdown headers (## or ###) while respecting max chunk size.
        Preserves YAML frontmatter in the first chunk.

        Args:
            content: Full note content
            max_chunk_tokens: Maximum tokens per chunk (default 3000)

        Returns:
            List of content chunks
        """
        # Convert tokens to chars (rough: 4 chars per token)
        max_chunk_chars = max_chunk_tokens * 4

        # Preserve frontmatter - extract it separately
        frontmatter = ""
        body = content
        frontmatter_match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(0)
            body = content[len(frontmatter) :]

        # Split by ## or ### headers (lookahead to keep header with section)
        sections = re.split(r"(?=^##+ )", body, flags=re.MULTILINE)
        sections = [s for s in sections if s.strip()]  # Remove empty sections

        if not sections:
            # No headers found, treat entire content as one chunk
            return [content]

        chunks: list[str] = []
        current_chunk = frontmatter  # Start first chunk with frontmatter

        for section in sections:
            section_chars = len(section)
            current_chars = len(current_chunk)

            # If adding this section exceeds limit and we have content, start new chunk
            if (
                current_chars + section_chars > max_chunk_chars
                and current_chunk.strip()
            ):
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += section

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If we only got one chunk that's still too big, return original
        # (will fail with clear error rather than infinite recursion)
        if len(chunks) == 1:
            return [content]

        logger.info(
            "note_chunked",
            original_chars=len(content),
            chunks=len(chunks),
            chunk_sizes=[len(c) for c in chunks],
        )

        return chunks

    def _extract_from_chunks(
        self,
        chunks: list[str],
        metadata: NoteMetadata,
        file_path: Path | None = None,
    ) -> list[QAPair]:
        """Extract QA pairs from multiple chunks and merge results.

        Args:
            chunks: List of content chunks to process
            metadata: Note metadata
            file_path: Optional file path for logging

        Returns:
            Merged list of QA pairs with corrected indices
        """
        all_qa_pairs: list[QAPair] = []

        for i, chunk in enumerate(chunks):
            logger.info(
                "processing_chunk",
                chunk_number=i + 1,
                total_chunks=len(chunks),
                chunk_chars=len(chunk),
                file=str(file_path) if file_path else "unknown",
            )

            # Extract from this chunk using the single-chunk method
            chunk_pairs = self._extract_single_chunk(chunk, metadata, file_path)

            # Reindex QA pairs to maintain sequential numbering
            for qa in chunk_pairs:
                qa.card_index = len(all_qa_pairs) + 1
                all_qa_pairs.append(qa)

            logger.debug(
                "chunk_extraction_complete",
                chunk_number=i + 1,
                pairs_extracted=len(chunk_pairs),
                total_pairs=len(all_qa_pairs),
            )

        logger.info(
            "chunked_extraction_complete",
            total_chunks=len(chunks),
            total_pairs=len(all_qa_pairs),
            file=str(file_path) if file_path else "unknown",
        )

        return all_qa_pairs

    def _extract_single_chunk(
        self,
        note_content: str,
        metadata: NoteMetadata,
        file_path: Path | None = None,
    ) -> list[QAPair]:
        """Extract QA pairs from a single chunk of content.

        This is the core extraction logic, separated to support chunking.

        Args:
            note_content: Note content (may be a chunk)
            metadata: Note metadata
            file_path: Optional file path for logging

        Returns:
            List of extracted QA pairs
        """
        # This method contains the actual LLM call logic
        # It's called by both extract_qa_pairs (for normal notes)
        # and _extract_from_chunks (for large notes)
        return self._do_extraction(note_content, metadata, file_path)

    def _do_extraction(
        self,
        note_content: str,
        metadata: NoteMetadata,
        file_path: Path | None = None,
    ) -> list[QAPair]:
        """Perform the actual LLM extraction.

        Args:
            note_content: Content to extract from
            metadata: Note metadata
            file_path: Optional file path for logging

        Returns:
            List of extracted QA pairs
        """
        prompt = self._build_extraction_prompt(note_content, metadata)

        system_prompt = """<role>
You are an expert Q&A extraction system specializing in educational note analysis. Your purpose is to identify and extract complete question-answer pairs from structured and unstructured markdown notes.
</role>

<capabilities>
- Pattern recognition across diverse markdown formatting styles
- Semantic understanding of question-answer relationships
- Multilingual content extraction (English and Russian)
- Distinction between answered and unanswered questions
</capabilities>

<critical_rules>
1. ONLY extract Q&A pairs where BOTH question AND answer are explicitly present
2. SKIP any sections labeled "Follow-up Questions", "Additional Questions", or similar
3. NEVER create Q&A pairs for unanswered questions
4. Unanswered questions may be noted in the 'followups' field for context only
5. Always respond in valid JSON format matching the provided schema
6. Be flexible in recognizing Q&A patterns regardless of markdown formatting
</critical_rules>

<output_requirements>
- Valid JSON structure only
- Complete extraction of all answerable Q&A pairs
- Accurate language field population based on language_tags
- Sequential card_index numbering starting from 1
- Keep extraction_notes concise (max 500 characters)
- Preserve original content verbatim - do not summarize or paraphrase answers
</output_requirements>

<efficiency_guidelines>
- Extract only what is explicitly present in the note
- Do not add explanations or commentary beyond what's needed
- Keep field values concise but complete
- Focus on accuracy over verbosity
</efficiency_guidelines>"""

        llm_start_time = time.time()

        logger.info(
            "qa_extraction_llm_start",
            model=self.model,
            note_id=metadata.id,
            title=metadata.title,
            file=str(file_path) if file_path else "unknown",
            prompt_length=len(prompt),
        )

        # Get JSON schema for structured output
        json_schema = get_qa_extraction_schema()

        # Calculate reasonable max_tokens based on input size
        prompt_tokens_estimate = len(prompt) // 4
        system_tokens_estimate = len(system_prompt) // 4
        total_input_tokens = prompt_tokens_estimate + system_tokens_estimate

        multiplier = 4.5 if total_input_tokens > 3000 else 4.0
        estimated_max_tokens = int(total_input_tokens * multiplier)

        from obsidian_anki_sync.providers.openrouter import (
            DEFAULT_MAX_OUTPUT_TOKENS,
            MODEL_MAX_OUTPUT_TOKENS,
        )

        model_max_output = MODEL_MAX_OUTPUT_TOKENS.get(
            self.model, DEFAULT_MAX_OUTPUT_TOKENS
        )
        # Use full model output capacity - chunking already handles context limits
        min_tokens_for_extraction = min(4096, model_max_output)
        estimated_max_tokens = max(estimated_max_tokens, min_tokens_for_extraction)
        estimated_max_tokens = min(estimated_max_tokens, model_max_output)

        # Temporarily adjust max_tokens
        original_max_tokens = None
        if hasattr(self.llm_provider, "max_tokens"):
            original_max_tokens = self.llm_provider.max_tokens
            if (
                original_max_tokens is not None
                and estimated_max_tokens > original_max_tokens
            ):
                self.llm_provider.max_tokens = estimated_max_tokens
                logger.info(
                    "increased_max_tokens_for_extraction",
                    original_max_tokens=original_max_tokens,
                    increased_max_tokens=estimated_max_tokens,
                    prompt_length=len(prompt),
                    prompt_tokens_estimate=prompt_tokens_estimate,
                    reason="Ensuring complete JSON responses without truncation",
                )

        # Update progress display if available
        progress_display = getattr(self, "progress_display", None)
        if progress_display:
            note_name = metadata.title[:50] if metadata.title else "Unknown"
            progress_display.update_operation("Extracting Q&A pairs", note_name)

        try:
            # Retry logic for JSON errors
            max_retries = 3
            last_error = None
            result = None

            import json

            for attempt in range(max_retries):
                try:
                    start_time = time.perf_counter()
                    result = self.llm_provider.generate_json(
                        model=self.model,
                        prompt=prompt,
                        system=system_prompt,
                        temperature=self.temperature,
                        json_schema=json_schema,
                        reasoning_enabled=self.reasoning_enabled,
                    )
                    duration = time.perf_counter() - start_time
                    log_slow_llm_request(
                        duration_seconds=duration,
                        threshold_seconds=self.slow_request_threshold,
                        model=self.model,
                        operation="qa_extraction",
                    )
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    logger.warning(
                        "qa_extraction_json_error",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    if attempt == max_retries - 1:
                        raise last_error
                    time.sleep(1)

            if result is None:
                return []

            # Extract and display reflections if available
            if progress_display:
                from obsidian_anki_sync.utils.progress_display import (
                    extract_reasoning_from_response,
                )

                reasoning = extract_reasoning_from_response(result, self.model)
                if reasoning:
                    progress_display.add_reflection(
                        f"Extraction reasoning: {reasoning[:200]}"
                    )

                extraction_notes = result.get("extraction_notes", "")
                if extraction_notes and len(extraction_notes) > 30:
                    progress_display.add_reflection(
                        f"Extraction notes: {extraction_notes[:200]}"
                    )
        finally:
            # Restore original max_tokens
            if (
                hasattr(self.llm_provider, "max_tokens")
                and original_max_tokens is not None
            ):
                self.llm_provider.max_tokens = original_max_tokens

        llm_duration = time.time() - llm_start_time

        logger.info(
            "qa_extraction_llm_complete",
            llm_duration=round(llm_duration, 2),
        )

        # Parse result
        qa_pairs_data = result.get("qa_pairs", [])
        extraction_notes = result.get("extraction_notes", "")

        # Convert to QAPair objects
        qa_pairs: list[QAPair] = []
        for qa_data in qa_pairs_data:
            try:
                qa_pair = QAPair(
                    card_index=qa_data.get("card_index", len(qa_pairs) + 1),
                    question_en=(qa_data.get("question_en") or "").strip(),
                    question_ru=(qa_data.get("question_ru") or "").strip(),
                    answer_en=(qa_data.get("answer_en") or "").strip(),
                    answer_ru=(qa_data.get("answer_ru") or "").strip(),
                    context=(qa_data.get("context") or "").strip(),
                    followups=(qa_data.get("followups") or "").strip(),
                    references=(qa_data.get("references") or "").strip(),
                    related=(qa_data.get("related") or "").strip(),
                )

                # Validate that required language content is present
                has_en = "en" in metadata.language_tags
                has_ru = "ru" in metadata.language_tags

                valid = True
                if has_en and (not qa_pair.question_en or not qa_pair.answer_en):
                    if self.enable_content_generation and self.repair_missing_sections:
                        if has_ru and qa_pair.question_ru and qa_pair.answer_ru:
                            if not qa_pair.question_en:
                                logger.info(
                                    "generating_missing_en_question",
                                    note_id=metadata.id,
                                    card_index=qa_pair.card_index,
                                )
                            if not qa_pair.answer_en:
                                logger.info(
                                    "generating_missing_en_answer",
                                    note_id=metadata.id,
                                    card_index=qa_pair.card_index,
                                )
                        else:
                            logger.warning(
                                "missing_en_content_in_extraction",
                                note_id=metadata.id,
                                card_index=qa_pair.card_index,
                                has_question=bool(qa_pair.question_en),
                                has_answer=bool(qa_pair.answer_en),
                            )
                            valid = False
                    else:
                        logger.warning(
                            "missing_en_content_in_extraction",
                            note_id=metadata.id,
                            card_index=qa_pair.card_index,
                            has_question=bool(qa_pair.question_en),
                            has_answer=bool(qa_pair.answer_en),
                        )
                        valid = False

                if has_ru and (not qa_pair.question_ru or not qa_pair.answer_ru):
                    if self.enable_content_generation and self.repair_missing_sections:
                        if has_en and qa_pair.question_en and qa_pair.answer_en:
                            if not qa_pair.question_ru:
                                logger.info(
                                    "generating_missing_ru_question",
                                    note_id=metadata.id,
                                    card_index=qa_pair.card_index,
                                )
                            if not qa_pair.answer_ru:
                                logger.info(
                                    "generating_missing_ru_answer",
                                    note_id=metadata.id,
                                    card_index=qa_pair.card_index,
                                )
                        else:
                            logger.warning(
                                "missing_ru_content_in_extraction",
                                note_id=metadata.id,
                                card_index=qa_pair.card_index,
                                has_question=bool(qa_pair.question_ru),
                                has_answer=bool(qa_pair.answer_ru),
                            )
                            valid = False
                    else:
                        logger.warning(
                            "missing_ru_content_in_extraction",
                            note_id=metadata.id,
                            card_index=qa_pair.card_index,
                            has_question=bool(qa_pair.question_ru),
                            has_answer=bool(qa_pair.answer_ru),
                        )
                        valid = False

                if valid:
                    qa_pairs.append(qa_pair)
                else:
                    logger.warning(
                        "skipping_invalid_qa_pair",
                        note_id=metadata.id,
                        card_index=qa_pair.card_index,
                    )

            except Exception as e:
                logger.error(
                    "qa_pair_conversion_error",
                    note_id=metadata.id,
                    card_index=qa_data.get("card_index", "unknown"),
                    error=str(e),
                )
                continue

        if not qa_pairs:
            logger.warning(
                "no_valid_qa_pairs_extracted",
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                extraction_notes=extraction_notes,
            )

        return qa_pairs

    def extract_qa_pairs(
        self,
        note_content: str,
        metadata: NoteMetadata,
        file_path: Path | None = None,
    ) -> list[QAPair]:
        """Extract Q&A pairs from note content using LLM.

        Automatically handles large notes by chunking them into smaller
        pieces that fit within model output limits.

        Args:
            note_content: Full note content including frontmatter
            metadata: Parsed metadata
            file_path: Optional file path for logging

        Returns:
            List of extracted Q&A pairs
        """
        start_time = time.time()

        logger.info(
            "qa_extraction_start",
            title=metadata.title,
            note_id=metadata.id,
            file=str(file_path) if file_path else "unknown",
            content_length=len(note_content),
            expected_languages=metadata.language_tags,
        )

        try:
            # Check if note fits in model's context window
            # Only chunk if note exceeds context window, not output limits
            processable, reason, content_tokens, total_required, context_window = (
                self._check_note_processable(note_content, file_path)
            )

            if not processable:
                # Try chunking approach for notes that exceed context window
                chunks = self._chunk_note_content(note_content)

                if len(chunks) > 1:
                    logger.info(
                        "chunking_large_note",
                        chunks=len(chunks),
                        reason=reason,
                        file=str(file_path) if file_path else "unknown",
                        content_tokens=content_tokens,
                        context_window=context_window,
                    )
                    qa_pairs = self._extract_from_chunks(chunks, metadata, file_path)
                else:
                    # Single chunk still too large - raise TruncationError
                    logger.warning(
                        "note_too_large_cannot_chunk",
                        file=str(file_path) if file_path else "unknown",
                        content_tokens=content_tokens,
                        total_required=total_required,
                        context_window=context_window,
                    )
                    msg = f"Note exceeds context window: {reason}"
                    raise TruncationError(
                        msg,
                        content_tokens=content_tokens,
                        required_output_tokens=total_required,
                        model_limit=context_window,
                        note_path=str(file_path) if file_path else None,
                        suggestion=(
                            "Consider splitting this note into smaller sections "
                            "with ## headers, or use a model with larger context window."
                        ),
                    )
            else:
                # Normal extraction - note fits in context window
                qa_pairs = self._do_extraction(note_content, metadata, file_path)

            total_duration = time.time() - start_time

            if qa_pairs:
                logger.info(
                    "qa_extraction_summary",
                    count=len(qa_pairs),
                    time=round(total_duration, 2),
                    pairs=[
                        {
                            "q_en": q.question_en[:50],
                            "a_en": q.answer_en[:50],
                            "q_ru": q.question_ru[:50],
                        }
                        for q in qa_pairs
                    ],
                )

            return qa_pairs

        except TruncationError:
            # Re-raise TruncationError for caller to handle
            raise

        except Exception as e:
            total_duration = time.time() - start_time

            # Categorize and log the error
            llm_error = categorize_llm_error(
                error=e,
                model=self.model,
                operation="qa-extraction",
                duration=total_duration,
            )

            log_llm_error(
                llm_error,
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                content_length=len(note_content),
                prompt_length=0,
            )

            logger.error(
                "qa_extraction_llm_error",
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                model=self.model,
                error_type=llm_error.error_type.value,
                error=str(llm_error),
                user_message=format_llm_error_for_user(llm_error),
                time=round(total_duration, 2),
            )

            # Return empty list - caller can fall back to rigid parser
            return []
