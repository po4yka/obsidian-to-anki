"""RAG (Retrieval-Augmented Generation) module for obsidian-anki-sync.

This module provides RAG capabilities for:
- Context enrichment during flashcard generation
- Duplicate detection via semantic similarity
- Few-shot example retrieval for improved generation quality

Components:
- EmbeddingProvider: Generates embeddings via OpenRouter/OpenAI
- DocumentChunker: Parses Obsidian markdown into searchable chunks
- VaultVectorStore: ChromaDB wrapper for vector storage
- RAGService: High-level API for RAG operations
- RAGIntegration: Helper for integrating RAG into agent pipeline
"""

from .document_chunker import DocumentChunk, DocumentChunker
from .embedding_provider import EmbeddingProvider
from .integration import RAGIntegration, get_rag_integration
from .rag_service import RAGService, get_rag_service
from .vector_store import VaultVectorStore

__all__ = [
    "DocumentChunk",
    "DocumentChunker",
    "EmbeddingProvider",
    "RAGIntegration",
    "RAGService",
    "VaultVectorStore",
    "get_rag_integration",
    "get_rag_service",
]
