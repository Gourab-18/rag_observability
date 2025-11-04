"""Retrieval module."""
from .retriever import DenseRetriever, RetrievedChunk
from .compressor import ChunkCompressor
from .hybrid_search import HybridRetriever

# Reranker is NOT exported by default to avoid loading local models
# Import directly when needed: from src.retrieval.reranker import Reranker
__all__ = ["DenseRetriever", "RetrievedChunk", "ChunkCompressor", "HybridRetriever"]

