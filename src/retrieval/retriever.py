# dense vector retrieval from Qdrant

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from qdrant_client import QdrantClient

from src.config import settings
from src.utils.embeddings import EmbeddingGenerator
from src.cache import cache_manager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    # represents a retrieved chunk with similarity score
    text: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    doc_id: str


class DenseRetriever:
    # dense vector retrieval using Qdrant
    
    def __init__(self):
        # initialize retriever with Qdrant client and embedding generator
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedding_generator = EmbeddingGenerator()
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        collection_name: str = None,
        score_threshold: float = None,
        use_cache: bool = True
    ) -> List[RetrievedChunk]:
        # retrieve top-k relevant chunks for a query
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        collection_name = collection_name or settings.qdrant_collection_name
        # use default threshold only if not explicitly provided (None)
        if score_threshold is None:
            score_threshold = settings.similarity_threshold
        
        # Check cache first
        if use_cache:
            cached_results = cache_manager.get_retrieval_results(query)
            if cached_results:
                # Convert cached dicts back to RetrievedChunk objects
                return [
                    RetrievedChunk(
                        text=item['text'],
                        metadata=item['metadata'],
                        score=item['score'],
                        chunk_id=item['chunk_id'],
                        doc_id=item['doc_id']
                    )
                    for item in cached_results
                ]
        
        # Check cache for embedding
        query_embedding = None
        if use_cache:
            query_embedding = cache_manager.get_query_embedding(query)
        
        # Generate embedding if not cached
        if query_embedding is None:
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            # Cache embedding
            if use_cache:
                cache_manager.cache_query_embedding(query, query_embedding)
        
        try:
            # search in Qdrant
            # don't pass score_threshold if it's 0.0 (show all results)
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_embedding.tolist(),
                "limit": top_k
            }
            # only add threshold if it's > 0 (filters results)
            if score_threshold > 0:
                search_params["score_threshold"] = score_threshold
            
            results = self.client.search(**search_params)
            
            # convert to RetrievedChunk objects
            retrieved_chunks = []
            for result in results:
                payload = result.payload
                # get text from payload (stored during indexing)
                chunk_text = payload.get('text', '')
                
                retrieved_chunks.append(RetrievedChunk(
                    text=chunk_text,
                    metadata=payload,
                    score=result.score,
                    chunk_id=payload.get('chunk_id', ''),
                    doc_id=payload.get('doc_id', '')
                ))
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            
            # Cache results
            if use_cache:
                cache_manager.cache_retrieval_results(query, retrieved_chunks)
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def get_chunk_text(self, chunk_id: str, collection_name: str = None) -> Optional[str]:
        # get full chunk text by chunk_id (for when we need to fetch from storage)
        # this would require storing chunk text in Qdrant payload or separate storage
        # for now, we'll return None and use metadata
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # search for the specific chunk by chunk_id in payload
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [{
                        "key": "chunk_id",
                        "match": {"value": chunk_id}
                    }]
                },
                limit=1
            )
            
            if results[0]:
                return results[0][0].payload.get('text', '')
            return None
            
        except Exception as e:
            logger.warning(f"Could not retrieve chunk text for {chunk_id}: {e}")
            return None

