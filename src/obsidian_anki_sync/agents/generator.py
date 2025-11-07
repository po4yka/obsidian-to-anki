"""Generator agent for APF card generation using local LLM.

This agent generates APF cards from Q/A pairs using Ollama with powerful models
like qwen3:32b for high-quality card generation.
"""

import json
import re
import time
from pathlib import Path

from ..models import Manifest, NoteMetadata, QAPair
from ..providers.base import BaseLLMProvider
from ..utils.logging import get_logger
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

        logger.info(
            "card_generation_complete",
            cards_generated=len(generated_cards),
            time=generation_time,
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

        # Generate GUID (deterministic based on slug)
        guid = str(hash(slug))

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

        # Build user prompt (reuse logic from APFGenerator)
        user_prompt = self._build_user_prompt(
            question=question,
            answer=answer,
            qa_pair=qa_pair,
            metadata=metadata,
            manifest=manifest,
            lang=lang,
        )

        logger.info(
            "generating_single_card",
            model=self.model,
            slug=manifest.slug,
            card_index=qa_pair.card_index,
            lang=lang,
            prompt_length=len(user_prompt),
            system_length=len(self.system_prompt),
        )

        try:
            # Call Ollama LLM
            llm_start_time = time.time()
            result = self.ollama_client.generate(
                model=self.model,
                prompt=user_prompt,
                system=self.system_prompt,
                temperature=self.temperature,
            )
            llm_duration = time.time() - llm_start_time

            apf_html = result.get("response", "")

            if not apf_html:
                raise ValueError("LLM returned empty response")

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
            )

            return GeneratedCard(
                card_index=qa_pair.card_index,
                slug=manifest.slug,
                lang=lang,
                apf_html=apf_html,
                confidence=confidence,
            )

        except Exception as e:
            card_duration = time.time() - card_start_time
            logger.error(
                "card_generation_failed",
                slug=manifest.slug,
                error=str(e),
                duration=round(card_duration, 2),
            )
            raise

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
            detected_lang if detected_lang != "text" else self._code_language_hint(metadata)
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
- Preserve code blocks with <pre><code class="language-{code_lang}">...</code></pre>
- Add "Ref: {ref_link}" in Other notes section
- Follow exact structure below

CRITICAL: Output ONLY the card HTML. NO explanations before or after.

Example Structure (follow exactly):

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

Now generate the card following this structure:
"""

        return prompt

    def _code_language_hint(self, metadata: NoteMetadata) -> str:
        """Derive a language hint for code blocks.

        Reuses logic from APFGenerator._code_language_hint.
        """
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
                .replace("/", "_")
            )
            if normalized in known_languages:
                return normalized

        return "plaintext"

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

        code_lower = code.lower().strip()

        # Kotlin patterns
        if any(
            k in code_lower
            for k in [
                "suspend fun",
                "data class",
                "sealed class",
                "sealed interface",
                "inline class",
                "value class",
            ]
        ):
            return "kotlin"
        if "fun " in code_lower and ("val " in code_lower or "var " in code_lower):
            return "kotlin"

        # Java patterns
        if any(
            k in code_lower
            for k in ["public class", "private class", "protected class"]
        ):
            return "java"
        if "import java." in code_lower or "package " in code_lower:
            return "java"

        # Python patterns (check before JavaScript due to 'async' keyword overlap)
        if any(k in code_lower for k in ["def ", "async def"]):
            # Strong Python indicators
            if any(k in code_lower for k in ["__init__", "self.", "import ", "from "]):
                return "python"
            # async def is Python-specific
            if "async def" in code_lower:
                return "python"

        # JavaScript/TypeScript
        if any(k in code_lower for k in ["function ", "const ", "let ", "=>", "async "]):
            # TypeScript-specific
            if any(
                k in code
                for k in ["interface ", "type ", ": string", ": number", "<T>"]
            ):
                return "typescript"
            return "javascript"

        # Shell/Bash
        if code.startswith("#!") or any(
            k in code_lower for k in ["#!/bin/", "export ", "echo ", "sudo "]
        ):
            return "bash"

        # YAML
        if re.match(r"^[a-z_]+:\s", code_lower, re.MULTILINE) and "-" in code:
            return "yaml"

        # JSON
        if code.strip().startswith("{") and '"' in code and ":" in code:
            return "json"

        # SQL
        if any(k in code_lower for k in ["select ", "insert ", "update ", "delete "]):
            return "sql"

        # Swift
        if any(k in code_lower for k in ["func ", "var ", "let ", "import swift"]):
            return "swift"

        # Go
        if "package main" in code_lower or "func main()" in code_lower:
            return "go"

        # Rust
        if any(k in code_lower for k in ["fn ", "let mut", "impl ", "pub fn"]):
            return "rust"

        # Fallback to metadata hint
        return "text"

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
        return f'<!-- manifest: {json.dumps(manifest_dict, ensure_ascii=False)} -->'

    def _post_process_apf(
        self, apf_html: str, metadata: NoteMetadata, manifest: Manifest
    ) -> str:
        """Post-process APF HTML to ensure correctness.

        This method:
        1. Strips markdown code fences
        2. Removes explanatory text before/after card
        3. Detects card type
        4. Generates and injects correct manifest
        5. Ensures proper formatting

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

        # 2. Strip any text before first card comment
        card_start = apf_html.find("<!-- Card")
        if card_start > 0:
            logger.debug(
                "stripped_text_before_card",
                slug=manifest.slug,
                chars_removed=card_start,
            )
            apf_html = apf_html[card_start:]

        # 3. Strip any text after manifest comment (if present)
        manifest_match = re.search(r"(<!-- manifest:.*?-->)", apf_html, re.DOTALL)
        if manifest_match:
            end_pos = manifest_match.end()
            # Keep only until end of manifest
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

        # 8. Ensure proper formatting
        apf_html = apf_html.strip()

        return apf_html
