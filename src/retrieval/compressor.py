# compression strategies: score filtering, BM25, LLM-based compression

from typing import List, Literal
import logging

from src.retrieval.retriever import RetrievedChunk
from src.config import settings

logger = logging.getLogger(__name__)


class ChunkCompressor:
    # compress retrieved chunks to reduce token count
    
    def compress_chunks(
        self,
        chunks: List[RetrievedChunk],
        strategy: Literal['score_filter', 'bm25', 'llm'] = 'score_filter',
        target_reduction: float = 0.3
    ) -> List[RetrievedChunk]:
        # compress chunks using specified strategy
        if strategy == 'score_filter':
            return self._score_filter(chunks)
        elif strategy == 'bm25':
            return self._bm25_filter(chunks)
        elif strategy == 'llm':
            return self._llm_compress(chunks)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _score_filter(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        # filter chunks by similarity score threshold
        threshold = settings.similarity_threshold
        filtered = [chunk for chunk in chunks if chunk.score >= threshold]
        
        logger.info(f"Score filtering: {len(chunks)} -> {len(filtered)} chunks (threshold: {threshold})")
        return filtered
    
    def _bm25_filter(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        # BM25 filtering for hybrid retrieval (sparse + dense)
        # this would require implementing BM25 or using a library
        # for now, return chunks as-is (placeholder)
        logger.warning("BM25 filtering not yet implemented, returning original chunks")
        return chunks
    
    def _llm_compress(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        # LLM-based compression using LangChain compressor
        # this would use LangChain's ContextualCompressionRetriever
        # for now, return chunks as-is (placeholder)
        logger.warning("LLM compression not yet implemented, returning original chunks")
        return chunks
    
    def calculate_token_reduction(
        self,
        original_chunks: List[RetrievedChunk],
        compressed_chunks: List[RetrievedChunk]
    ) -> float:
        # calculate token reduction percentage
        if not original_chunks:
            return 0.0
        
        # approximate token count (rough estimate: 4 chars = 1 token)
        original_tokens = sum(len(chunk.text) // 4 for chunk in original_chunks)
        compressed_tokens = sum(len(chunk.text) // 4 for chunk in compressed_chunks)
        
        if original_tokens == 0:
            return 0.0
        
        reduction = (original_tokens - compressed_tokens) / original_tokens
        return reduction

