"""APF card generation via OpenRouter LLM."""

import html
import json
import re
from pathlib import Path

import httpx  # type: ignore
from openai import OpenAI  # type: ignore

from ..config import Config
from ..models import Card, Manifest, NoteMetadata, QAPair
from ..utils.code_detection import detect_code_language_from_metadata
from ..utils.content_hash import compute_content_hash
from ..utils.logging import get_logger
from ..utils.retry import retry

logger = get_logger(__name__)


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

        # Normalize Markdown code fences (if any) into HTML blocks
        default_lang = self._code_language_hint(metadata)
        apf_html = self._normalize_code_blocks(apf_html, default_lang)

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
- Tags: Derive from topic/subtopics, 3-6 snake_case tags, include primary language/tech
- Primary language tag: {lang}
- Topic-based tags: {metadata.topic.lower().replace(" ", "_")}
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
        """Convert Markdown code fences to <pre><code> blocks if present."""
        if "```" not in apf_html:
            return apf_html

        normalized_parts: list[str] = []
        cursor = 0
        default_class = self._format_code_language(default_lang)

        while True:
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

            cursor = end + 3

        return "".join(normalized_parts)

    def _format_code_language(self, lang: str) -> str:
        """Normalize language identifiers for code block classes."""
        if not lang:
            return "plaintext"
        normalized = lang.strip().lower().replace(" ", "-").replace("/", "-")
        normalized = re.sub(r"[^a-z0-9_\-+.]", "", normalized)
        if normalized.startswith("language-"):
            normalized = normalized[len("language-") :]
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
