"""Q&A extraction agent using LLM for flexible note parsing.

This agent intelligently extracts question-answer pairs from notes
regardless of their markdown format, using LLM to understand structure.
"""

import time
from pathlib import Path

from ..models import NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
from .json_schemas import get_qa_extraction_schema
from .llm_errors import (
    categorize_llm_error,
    format_llm_error_for_user,
    log_llm_error,
)

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

    def extract_qa_pairs(
        self,
        note_content: str,
        metadata: NoteMetadata,
        file_path: Path | None = None,
    ) -> list[QAPair]:
        """Extract Q&A pairs from note content using LLM.

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
            # For extraction, we typically need 4-4.5x the input size for bilingual content
            # Using character count as rough estimate (1 token ≈ 4 characters)
            prompt_tokens_estimate = len(prompt) // 4
            system_tokens_estimate = len(system_prompt) // 4
            total_input_tokens = prompt_tokens_estimate + system_tokens_estimate

            # For bilingual Q&A extraction, use more conservative multiplier
            # Large prompts (>3000 tokens) need 4.5x, smaller ones can use 4.0x
            multiplier = 4.5 if total_input_tokens > 3000 else 4.0
            estimated_max_tokens = int(total_input_tokens * multiplier)

            # Ensure we have reasonable tokens for complex extractions
            # But respect model-specific output limits (not context window limits)
            from ..providers.openrouter import (
                DEFAULT_MAX_OUTPUT_TOKENS,
                MODEL_MAX_OUTPUT_TOKENS,
            )

            model_max_output = MODEL_MAX_OUTPUT_TOKENS.get(
                self.model, DEFAULT_MAX_OUTPUT_TOKENS
            )

            # Add safety margin: use 90% of model's max output to prevent hitting hard limits
            safe_max_output = int(model_max_output * 0.9)

            # Set minimum to 4096 for QA extraction (reasonable for most models)
            # But don't exceed model's safe output limit
            min_tokens_for_extraction = min(4096, safe_max_output)
            estimated_max_tokens = max(estimated_max_tokens, min_tokens_for_extraction)
            estimated_max_tokens = min(estimated_max_tokens, safe_max_output)

            # Temporarily adjust max_tokens for this extraction to ensure complete responses
            # Store original max_tokens to restore later
            original_max_tokens = None
            if hasattr(self.llm_provider, "max_tokens"):
                original_max_tokens = self.llm_provider.max_tokens
                # Always use the larger of: configured max or estimated needed
                # This ensures we don't truncate responses
                # Handle case where original_max_tokens might be None
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
                result = self.llm_provider.generate_json(
                    model=self.model,
                    prompt=prompt,
                    system=system_prompt,
                    temperature=self.temperature,
                    json_schema=json_schema,
                    reasoning_enabled=self.reasoning_enabled,
                )

                # Extract and display reflections if available
                if progress_display:
                    from ..utils.progress_display import extract_reasoning_from_response

                    reasoning = extract_reasoning_from_response(result, self.model)
                    if reasoning:
                        progress_display.add_reflection(
                            f"Extraction reasoning: {reasoning[:200]}"
                        )

                    # Also check extraction_notes
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
            total_duration = time.time() - start_time

            logger.info(
                "qa_extraction_llm_complete",
                llm_duration=round(llm_duration, 2),
                total_duration=round(total_duration, 2),
            )

            # Parse result
            qa_pairs_data = result.get("qa_pairs", [])
            extraction_notes = result.get("extraction_notes", "")
            total_pairs = result.get("total_pairs", len(qa_pairs_data))

            logger.info(
                "qa_extraction_complete",
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                pairs_extracted=len(qa_pairs_data),
                expected_total=total_pairs,
                extraction_notes=extraction_notes,
                time=round(total_duration, 2),
            )

            # Convert to QAPair objects
            qa_pairs = []
            for qa_data in qa_pairs_data:
                try:
                    qa_pair = QAPair(
                        card_index=qa_data.get("card_index", len(qa_pairs) + 1),
                        question_en=qa_data.get("question_en", "").strip(),
                        question_ru=qa_data.get("question_ru", "").strip(),
                        answer_en=qa_data.get("answer_en", "").strip(),
                        answer_ru=qa_data.get("answer_ru", "").strip(),
                        context=qa_data.get("context", "").strip(),
                        followups=qa_data.get("followups", "").strip(),
                        references=qa_data.get("references", "").strip(),
                        related=qa_data.get("related", "").strip(),
                    )

                    # Validate that required language content is present
                    has_en = "en" in metadata.language_tags
                    has_ru = "ru" in metadata.language_tags

                    valid = True
                    if has_en and (not qa_pair.question_en or not qa_pair.answer_en):
                        if (
                            self.enable_content_generation
                            and self.repair_missing_sections
                        ):
                            # Generate missing EN content from RU if available
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
                        if (
                            self.enable_content_generation
                            and self.repair_missing_sections
                        ):
                            # Generate missing RU content from EN if available
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

        except Exception as e:
            llm_duration = (
                time.time() - llm_start_time if "llm_start_time" in locals() else 0
            )
            total_duration = time.time() - start_time

            # Categorize and log the error
            llm_error = categorize_llm_error(
                error=e,
                model=self.model,
                operation="qa-extraction",
                duration=llm_duration,
            )

            log_llm_error(
                llm_error,
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                content_length=len(note_content),
                prompt_length=len(prompt) if "prompt" in locals() else 0,
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
