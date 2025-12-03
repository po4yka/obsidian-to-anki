import importlib.util
import re
import sys
from pathlib import Path

# Import the module directly to avoid pulling in heavyweight optional dependencies
sys.path.insert(0, str(Path("src").resolve()))

module_path = Path("src/obsidian_anki_sync/rag/document_chunker.py")
spec = importlib.util.spec_from_file_location("document_chunker", module_path)
document_chunker = importlib.util.module_from_spec(spec) if spec else None

if spec and spec.loader and document_chunker:
    sys.modules[spec.name] = document_chunker
    spec.loader.exec_module(document_chunker)
else:
    message = "Failed to load document_chunker module for testing"
    raise ImportError(message)

DocumentChunker = document_chunker.DocumentChunker


def test_chunk_id_includes_path_hash() -> None:
    chunker = DocumentChunker()

    chunk_id = chunker._generate_chunk_id("topic1/note.md", "summary")

    assert re.match(r"^note_summary_[0-9a-f]{8}$", chunk_id)


def test_chunk_id_differs_for_same_filename_in_different_dirs() -> None:
    chunker = DocumentChunker()

    first = chunker._generate_chunk_id("topic1/note.md", "summary")
    second = chunker._generate_chunk_id("topic2/note.md", "summary")

    assert first != second
    assert first.rsplit("_", 1)[0] == second.rsplit("_", 1)[0]
