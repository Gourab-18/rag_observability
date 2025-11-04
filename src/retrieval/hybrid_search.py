# hybrid search combining dense and sparse retrieval

from typing import List
import logging

from src.retrieval.retriever import DenseRetriever, RetrievedChunk
from src.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    # hybrid retrieval combining dense (vector) and sparse (BM25) search
    
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.bm25_index = None  # will be initialized if BM25 is used
    
    def hybrid_search(
        self,
        query: str,
        dense_k: int = 10,
        sparse_k: int = 5
    ) -> List[RetrievedChunk]:
        # hybrid search: combine dense vector search with sparse BM25 search
        # for now, return dense search results (BM25 implementation pending)
        
        logger.info(f"Hybrid search: dense_k={dense_k}, sparse_k={sparse_k}")
        
        # dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)
        
        # sparse retrieval (BM25) - placeholder for now
        # sparse_results = self._bm25_search(query, sparse_k)
        
        # combine and deduplicate results
        # for now, just return dense results
        combined_results = dense_results
        
        logger.info(f"Hybrid search returned {len(combined_results)} chunks")
        return combined_results
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        # BM25 sparse retrieval (placeholder)
        # would require building BM25 index from chunks
        logger.warning("BM25 search not yet implemented")
        return []

