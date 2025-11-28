"""Document chunker for RAG system.

Parses Obsidian markdown files into structured chunks suitable for
vector embedding and retrieval. Preserves metadata and relationships.
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import frontmatter

from obsidian_anki_sync.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkType(str, Enum):
    """Type of document chunk."""

    SUMMARY_EN = "summary_en"
    SUMMARY_RU = "summary_ru"
    KEY_POINTS_EN = "key_points_en"
    KEY_POINTS_RU = "key_points_ru"
    CODE_EXAMPLE = "code_example"
    QUESTION_EN = "question_en"
    QUESTION_RU = "question_ru"
    ANSWER_EN = "answer_en"
    ANSWER_RU = "answer_ru"
    FULL_CONTENT = "full_content"
    SECTION = "section"


@dataclass
class DocumentChunk:
    """A chunk of document content with metadata."""

    chunk_id: str
    content: str
    chunk_type: ChunkType
    source_file: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Content hash for change detection
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Compute content hash if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "source_file": self.source_file,
            "content_hash": self.content_hash,
            **self.metadata,
        }


class DocumentChunker:
    """Chunker for Obsidian markdown documents.

    Extracts structured chunks from markdown files:
    - YAML frontmatter as metadata
    - Summary sections (EN/RU)
    - Key points
    - Code examples
    - Q&A pairs

    Preserves relationships via metadata for RAG retrieval.
    """

    # Section header patterns
    SECTION_PATTERNS = {
        "summary_en": re.compile(
            r"^##\s+(?:Summary\s*\(EN\)|Summary \(EN\))\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "summary_ru": re.compile(
            r"^##\s+(?:Краткое Описание\s*\(RU\)|Краткое Описание \(RU\))\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "key_points_en": re.compile(
            r"^##\s+(?:Key Points\s*\(EN\)|Key Points \(EN\))\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "key_points_ru": re.compile(
            r"^##\s+(?:Ключевые Моменты\s*\(RU\)|Ключевые Моменты \(RU\))\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "question_en": re.compile(
            r"^#\s+(?:Question\s*\(EN\)|Question \(EN\))\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        "question_ru": re.compile(
            r"^#\s+(?:Вопрос\s*\(RU\)|Вопрос \(RU\))\s*$", re.IGNORECASE | re.MULTILINE
        ),
        "answer_en": re.compile(
            r"^##\s+(?:Answer\s*\(EN\)|Answer \(EN\))\s*$", re.IGNORECASE | re.MULTILINE
        ),
        "answer_ru": re.compile(
            r"^##\s+(?:Ответ\s*\(RU\)|Ответ \(RU\))\s*$", re.IGNORECASE | re.MULTILINE
        ),
    }

    # Code block pattern
    CODE_BLOCK_PATTERN = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        include_code_blocks: bool = True,
        min_chunk_size: int = 50,
    ):
        """Initialize document chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            include_code_blocks: Whether to extract code blocks as separate chunks
            min_chunk_size: Minimum chunk size (skip smaller content)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_code_blocks = include_code_blocks
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_path: Path) -> list[DocumentChunk]:
        """Parse and chunk a single markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            List of document chunks
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self.chunk_content(content, str(file_path))

        except Exception as e:
            logger.error(
                "file_chunking_failed",
                file=str(file_path),
                error=str(e),
            )
            return []

    def chunk_content(
        self,
        content: str,
        source_file: str,
    ) -> list[DocumentChunk]:
        """Parse and chunk markdown content.

        Args:
            content: Markdown content
            source_file: Source file path for metadata

        Returns:
            List of document chunks
        """
        chunks: list[DocumentChunk] = []

        # Parse frontmatter
        try:
            parsed = frontmatter.loads(content)
            metadata = dict(parsed.metadata)
            body = parsed.content
        except Exception:
            metadata = {}
            body = content

        # Extract base metadata
        base_metadata = self._extract_base_metadata(metadata, source_file)

        # Extract sections
        sections = self._extract_sections(body)

        # Create chunks for each section
        for section_type, section_content in sections.items():
            if (
                not section_content
                or len(section_content.strip()) < self.min_chunk_size
            ):
                continue

            chunk_type = self._get_chunk_type(section_type)
            chunk_id = self._generate_chunk_id(source_file, section_type)

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=section_content.strip(),
                    chunk_type=chunk_type,
                    source_file=source_file,
                    metadata={
                        **base_metadata,
                        "section": section_type,
                    },
                )
            )

        # Extract code blocks if enabled
        if self.include_code_blocks:
            code_chunks = self._extract_code_blocks(body, source_file, base_metadata)
            chunks.extend(code_chunks)

        # If no structured chunks found, create full content chunk
        if not chunks and body.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=self._generate_chunk_id(source_file, "full"),
                    content=self._truncate_content(body.strip()),
                    chunk_type=ChunkType.FULL_CONTENT,
                    source_file=source_file,
                    metadata=base_metadata,
                )
            )

        logger.debug(
            "content_chunked",
            source=source_file,
            chunks=len(chunks),
            metadata_keys=list(base_metadata.keys()),
        )

        return chunks

    def _extract_base_metadata(
        self,
        frontmatter_metadata: dict[str, Any],
        source_file: str,
    ) -> dict[str, Any]:
        """Extract base metadata from frontmatter.

        Args:
            frontmatter_metadata: Parsed frontmatter
            source_file: Source file path

        Returns:
            Base metadata dictionary
        """
        metadata: dict[str, Any] = {
            "source_file": source_file,
        }

        # Extract key metadata fields
        key_fields = [
            "id",
            "title",
            "topic",
            "subtopics",
            "difficulty",
            "language_tags",
            "moc",
            "related",
            "question_kind",
            "original_language",
            "tags",
            "status",
        ]

        for field_name in key_fields:
            if field_name in frontmatter_metadata:
                value = frontmatter_metadata[field_name]
                # Convert lists to strings for filtering
                if isinstance(value, list):
                    metadata[field_name] = ",".join(str(v) for v in value)
                else:
                    metadata[field_name] = str(value)

        return metadata

    def _extract_sections(self, body: str) -> dict[str, str]:
        """Extract named sections from markdown body.

        Args:
            body: Markdown content without frontmatter

        Returns:
            Dictionary mapping section names to content
        """
        sections: dict[str, str] = {}

        # Find all section headers and their positions
        header_positions: list[tuple[str, int, int]] = []

        for section_name, pattern in self.SECTION_PATTERNS.items():
            for match in pattern.finditer(body):
                header_positions.append(
                    (
                        section_name,
                        match.start(),
                        match.end(),
                    )
                )

        # Sort by position
        header_positions.sort(key=lambda x: x[1])

        # Extract content between headers
        for i, (section_name, start, end) in enumerate(header_positions):
            # Find next header or end of document
            if i + 1 < len(header_positions):
                next_start = header_positions[i + 1][1]
            else:
                next_start = len(body)

            # Extract section content (skip the header itself)
            section_content = body[end:next_start].strip()

            # Remove markdown blockquotes if present (common in Q&A format)
            section_content = self._clean_section_content(section_content)

            if section_content:
                sections[section_name] = section_content

        return sections

    def _clean_section_content(self, content: str) -> str:
        """Clean section content.

        Args:
            content: Raw section content

        Returns:
            Cleaned content
        """
        # Remove leading blockquote markers
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove leading > from blockquotes
            if line.startswith(">"):
                line = line[1:].strip()
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _extract_code_blocks(
        self,
        body: str,
        source_file: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Extract code blocks as separate chunks.

        Args:
            body: Markdown content
            source_file: Source file path
            base_metadata: Base metadata for chunks

        Returns:
            List of code block chunks
        """
        chunks: list[DocumentChunk] = []

        for i, match in enumerate(self.CODE_BLOCK_PATTERN.finditer(body)):
            language = match.group(1) or "unknown"
            code = match.group(2).strip()

            if len(code) < self.min_chunk_size:
                continue

            chunk_id = self._generate_chunk_id(source_file, f"code_{i}")

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=code,
                    chunk_type=ChunkType.CODE_EXAMPLE,
                    source_file=source_file,
                    metadata={
                        **base_metadata,
                        "code_language": language,
                        "code_index": i,
                    },
                )
            )

        return chunks

    def _get_chunk_type(self, section_type: str) -> ChunkType:
        """Map section type to chunk type.

        Args:
            section_type: Section type string

        Returns:
            ChunkType enum value
        """
        mapping = {
            "summary_en": ChunkType.SUMMARY_EN,
            "summary_ru": ChunkType.SUMMARY_RU,
            "key_points_en": ChunkType.KEY_POINTS_EN,
            "key_points_ru": ChunkType.KEY_POINTS_RU,
            "question_en": ChunkType.QUESTION_EN,
            "question_ru": ChunkType.QUESTION_RU,
            "answer_en": ChunkType.ANSWER_EN,
            "answer_ru": ChunkType.ANSWER_RU,
        }
        return mapping.get(section_type, ChunkType.SECTION)

    def _generate_chunk_id(self, source_file: str, section: str) -> str:
        """Generate unique chunk ID.

        Args:
            source_file: Source file path
            section: Section identifier

        Returns:
            Unique chunk ID
        """
        # Use file name + section for readable ID
        file_name = Path(source_file).stem
        return f"{file_name}_{section}"

    def _truncate_content(self, content: str) -> str:
        """Truncate content to chunk size if needed.

        Args:
            content: Content to truncate

        Returns:
            Truncated content
        """
        if len(content) <= self.chunk_size:
            return content

        # Truncate at word boundary
        truncated = content[: self.chunk_size]
        last_space = truncated.rfind(" ")

        if last_space > self.chunk_size // 2:
            truncated = truncated[:last_space]

        return truncated + "..."

    def chunk_vault(
        self,
        vault_path: Path,
        source_dirs: list[Path] | None = None,
        file_pattern: str = "*.md",
    ) -> list[DocumentChunk]:
        """Chunk all markdown files in a vault.

        Args:
            vault_path: Path to Obsidian vault
            source_dirs: Optional list of subdirectories to process
            file_pattern: Glob pattern for files

        Returns:
            List of all document chunks
        """
        all_chunks: list[DocumentChunk] = []

        # Determine directories to search
        if source_dirs:
            search_paths = [vault_path / d for d in source_dirs]
        else:
            search_paths = [vault_path]

        # Process each directory
        for search_path in search_paths:
            if not search_path.exists():
                logger.warning(
                    "search_path_not_found",
                    path=str(search_path),
                )
                continue

            # Find all markdown files
            md_files = list(search_path.rglob(file_pattern))

            logger.info(
                "chunking_directory",
                path=str(search_path),
                files=len(md_files),
            )

            for md_file in md_files:
                # Skip hidden files and directories
                if any(part.startswith(".") for part in md_file.parts):
                    continue
                # Skip template files
                if "_templates" in str(md_file):
                    continue

                chunks = self.chunk_file(md_file)
                all_chunks.extend(chunks)

        logger.info(
            "vault_chunking_complete",
            vault=str(vault_path),
            total_chunks=len(all_chunks),
        )

        return all_chunks
