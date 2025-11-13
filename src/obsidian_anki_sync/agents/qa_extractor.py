"""Q&A extraction agent using LLM for flexible note parsing.

This agent intelligently extracts question-answer pairs from notes
regardless of their markdown format, using LLM to understand structure.
"""

import json
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
    ):
        """Initialize Q&A extractor agent.

        Args:
            llm_provider: LLM provider instance
            model: Model to use for extraction
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.llm_provider = llm_provider
        self.model = model
        self.temperature = temperature
        logger.info("qa_extractor_agent_initialized", model=model)

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
5. Extract answers that directly correspond to each question
6. For bilingual notes (en, ru), extract BOTH language versions of questions and answers
7. Preserve the semantic relationship between questions and answers
8. Number all Q&A pairs sequentially starting from 1
9. Maintain the order of Q&A pairs as they appear in the note
10. Preserve markdown formatting in questions and answers

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
</constraints>
</rules>

<output_format>
Respond with valid JSON matching this exact structure:

{{
  "qa_pairs": [
    {{
      "card_index": 1,
      "question_en": "Question text in English",
      "question_ru": "Текст вопроса на русском",
      "answer_en": "Answer text in English",
      "answer_ru": "Текст ответа на русском",
      "context": "Optional context before Q&A",
      "followups": "Optional follow-up questions",
      "references": "Optional references",
      "related": "Optional related topics"
    }}
  ],
  "extraction_notes": "Notes about the extraction process or why certain pairs were skipped",
  "total_pairs": 1
}}
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
</output_requirements>"""

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

            result = self.llm_provider.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
                json_schema=json_schema,
            )

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
                        logger.warning(
                            "missing_en_content_in_extraction",
                            note_id=metadata.id,
                            card_index=qa_pair.card_index,
                            has_question=bool(qa_pair.question_en),
                            has_answer=bool(qa_pair.answer_en),
                        )
                        valid = False

                    if has_ru and (not qa_pair.question_ru or not qa_pair.answer_ru):
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
            llm_duration = time.time() - llm_start_time if "llm_start_time" in locals() else 0
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
