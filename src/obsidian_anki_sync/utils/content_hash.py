"""Utilities for computing stable content hashes for cards."""

from __future__ import annotations

import hashlib

from ..models import NoteMetadata, QAPair


def compute_content_hash(qa_pair: QAPair, metadata: NoteMetadata, lang: str) -> str:
    """Compute a content hash that captures all card-relevant sections.

    Args:
        qa_pair: Parsed Q/A pair.
        metadata: Note metadata (for tags/contextual fields).
        lang: Language code ("en" or "ru").

    Returns:
        SHA256 hash string that changes whenever any card component changes.
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
