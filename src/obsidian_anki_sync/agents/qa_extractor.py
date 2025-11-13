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

        return f"""You are an expert at extracting question-answer pairs from educational notes.

TASK: Extract all question-answer pairs from this note, regardless of their format.

NOTE METADATA:
- Title: {metadata.title}
- Topic: {metadata.topic}
- Languages: {language_list}
- Expected Languages: Each Q&A should have content in: {language_list}

FULL NOTE CONTENT:
{note_content}

EXTRACTION RULES:
1. Look for questions and their corresponding answers throughout the note
2. Questions might be in various formats:
   - Explicit headers like "# Question (EN)" or "## What is..."
   - Numbered or bulleted lists
   - Implicit questions within text
   - Q&A formatted blocks
3. Extract answers that correspond to each question
4. For bilingual notes (en, ru), extract both language versions of questions/answers
5. Preserve the semantic relationship between questions and answers
6. Include all Q&A pairs found, number them sequentially starting from 1
7. For each Q&A pair, extract:
   - question_en: Question in English (if language includes 'en')
   - question_ru: Question in Russian (if language includes 'ru')
   - answer_en: Answer in English (if language includes 'en')
   - answer_ru: Answer in Russian (if language includes 'ru')
   - context: Any contextual information before the Q&A
   - followups: Any follow-up questions mentioned
   - references: Any references or citations
   - related: Related topics or questions

OUTPUT FORMAT:
Respond with a JSON object containing a list of extracted Q&A pairs:

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
  "extraction_notes": "Any notes about the extraction process",
  "total_pairs": 1
}}

IMPORTANT:
- If a language is not in the language_tags, leave that field empty
- Preserve markdown formatting in questions and answers
- Be thorough - extract ALL Q&A pairs you can identify
- If no clear Q&A pairs exist, return an empty list with explanation in extraction_notes
- Maintain the order of Q&A pairs as they appear in the note
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

            system_prompt = """You are an expert Q&A extraction system.
Your job is to identify and extract question-answer pairs from educational notes.
Be flexible and intelligent - recognize Q&A patterns regardless of formatting.
Always respond in valid JSON format.
Be thorough and extract all Q&A pairs you can find."""

            llm_start_time = time.time()

            logger.info(
                "qa_extraction_llm_start",
                model=self.model,
                note_id=metadata.id,
                title=metadata.title,
                file=str(file_path) if file_path else "unknown",
                prompt_length=len(prompt),
            )

            result = self.llm_provider.generate_json(
                model=self.model,
                prompt=prompt,
                system=system_prompt,
                temperature=self.temperature,
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
