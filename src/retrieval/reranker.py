# reranking using cross-encoder models (optional, requires local models)

import os
# Set environment variables BEFORE any imports that might load TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from typing import List
import logging
import sys
from contextlib import contextmanager

from src.retrieval.retriever import RetrievedChunk
from src.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


class Reranker:
    # rerank retrieved chunks using cross-encoder (optional, disabled by default)
    
    def __init__(self):
        # initialize reranker model (only if enabled in settings)
        self.model = None
        if settings.use_reranking:
            self._initialize_model()
        else:
            logger.info("Reranking disabled in settings")
    
    def _initialize_model(self):
        # initialize cross-encoder model for reranking (requires sentence-transformers)
        # only loads if use_reranking=True in settings
        # lazy import to avoid loading local models unless explicitly needed
        try:
            import sentence_transformers  # check if available
            from sentence_transformers import CrossEncoder
            # use a small, fast cross-encoder model
            # Suppress stderr during model loading to hide mutex warnings
            with suppress_stderr():
                self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder reranker model loaded")
        except ImportError:
            logger.warning("sentence-transformers not available, reranking disabled. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load reranker model: {e}. Reranking disabled.")
            self.model = None
    
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        # rerank chunks by relevance to query
        # returns original chunks if reranking not available or disabled
        if not settings.use_reranking:
            logger.debug("Reranking disabled, returning original chunks")
            return chunks
            
        if not self.model or not chunks:
            logger.debug("Reranker model not available, returning original chunks")
            return chunks
        
        try:
            # prepare pairs for cross-encoder
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # get reranking scores
            scores = self.model.predict(pairs)
            
            # combine scores with original chunks
            reranked_chunks = list(zip(chunks, scores))
            
            # sort by reranking score (descending)
            reranked_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # update chunks with new scores and return
            reranked = []
            for chunk, new_score in reranked_chunks:
                # create new chunk with updated score
                reranked.append(RetrievedChunk(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(new_score),
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.chunk_id
                ))
            
            logger.info(f"Reranked {len(reranked)} chunks")
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking: {e}")
            # return original chunks if reranking fails
            return chunks