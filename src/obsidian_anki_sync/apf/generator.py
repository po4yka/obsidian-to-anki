"""APF card generation via OpenRouter LLM."""

import hashlib
import json
import re
from pathlib import Path

import httpx  # type: ignore
from openai import OpenAI  # type: ignore

from ..config import Config
from ..models import Card, Manifest, NoteMetadata, QAPair
from ..utils.logging import get_logger
from ..utils.retry import retry

logger = get_logger(__name__)


def compute_content_hash(qa_pair: QAPair, metadata: NoteMetadata, lang: str) -> str:
    """
    Compute a content hash that captures all card-relevant sections.

    Args:
        qa_pair: Parsed Q/A pair
        metadata: Note metadata (for tags/contextual fields)
        lang: Language code ('en' or 'ru')

    Returns:
        SHA256 hash string
    """
    question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
    answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

    components = [
        f"lang:{lang}",
        f"question:{question.strip()}",
        f"answer:{answer.strip()}",
        f"followups:{qa_pair.followups.strip()}",
        f"references:{qa_pair.references.strip()}",
        f"related:{qa_pair.related.strip()}",
        f"context:{qa_pair.context.strip()}",
        f"title:{metadata.title}",
        f"topic:{metadata.topic}",
        f"subtopics:{','.join(sorted(metadata.subtopics))}",
        f"tags:{','.join(sorted(metadata.tags))}",
    ]

    payload = "\n".join(components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class APFGenerator:
    """Generate APF cards using OpenRouter LLM."""

    def __init__(self, config: Config):
        """Initialize the generator."""
        self.config = config
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.openrouter_api_key,
        )

        # Load CARDS_PROMPT template
        prompt_path = Path(__file__).parents[3] / ".docs" / "CARDS_PROMPT.md"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            logger.warning("cards_prompt_not_found", path=str(prompt_path))
            self.system_prompt = "Generate APF cards following strict APF format."

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
        # Select language-specific content
        question = qa_pair.question_en if lang == "en" else qa_pair.question_ru
        answer = qa_pair.answer_en if lang == "en" else qa_pair.answer_ru

        # Build user prompt
        user_prompt = self._build_user_prompt(
            question, answer, qa_pair, metadata, manifest, lang
        )

        # Call LLM
        logger.debug(
            "calling_llm",
            model=self.config.openrouter_model,
            temp=self.config.llm_temperature,
            slug=manifest.slug,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.openrouter_model,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            apf_html = response.choices[0].message.content

            if not apf_html:
                raise ValueError("LLM returned empty response")

            logger.debug(
                "llm_response_received", slug=manifest.slug, length=len(apf_html)
            )

        except Exception as e:
            logger.error("llm_call_failed", slug=manifest.slug, error=str(e))
            raise

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
- Subtopics: {', '.join(metadata.subtopics) if metadata.subtopics else 'None'}
- Difficulty: {metadata.difficulty or 'Not specified'}
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
- Tags: Derive from topic/subtopics, 3-6 snake_case tags, include primary language/tech
- Primary language tag: {lang}
- Topic-based tags: {metadata.topic.lower().replace(' ', '_')}
- Preserve every code block using <pre><code class="language-{self._code_language_hint(metadata)}"> ... </code></pre> with original indentation; do NOT fall back to Markdown fences.
- Include manifest at end with slug "{manifest.slug}"
- Add "Ref: {ref_link}" in Other notes section
- Follow APF v2.1 format strictly
- Output ONLY the card HTML, no explanations
"""

        return prompt

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
            raise ValueError("APF output missing manifest comment")

        try:
            manifest_data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {exc}") from exc

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
        tags = set()

        # Add language
        tags.add(lang)

        # Add topic
        if metadata.topic:
            tags.add(metadata.topic.lower().replace(" ", "_"))

        # Add subtopics
        for subtopic in metadata.subtopics:
            tags.add(subtopic.lower().replace(" ", "_"))

        # Add metadata tags
        tags.update(metadata.tags)

        return sorted(list(tags))

    def _code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks."""
        candidates = list(metadata.tags) + metadata.subtopics + [metadata.topic]
        known_languages = {
            "kotlin",
            "java",
            "python",
            "swift",
            "cpp",
            "c",
            "csharp",
            "go",
            "rust",
            "javascript",
            "typescript",
            "sql",
            "bash",
            "shell",
            "yaml",
            "json",
            "html",
            "css",
            "gradle",
            "groovy",
        }
        for raw in candidates:
            if not raw:
                continue
            normalized = (
                raw.lower()
                .replace("language/", "")
                .replace("lang/", "")
                .replace("topic/", "")
            )
            normalized = normalized.replace("/", "_")
            if normalized in known_languages:
                return normalized
        return "plaintext"
