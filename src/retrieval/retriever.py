# dense vector retrieval from Qdrant

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from qdrant_client import QdrantClient

from src.config import settings
from src.utils.embeddings import EmbeddingGenerator
# Note: Chunk is imported but not used in this file - can be removed if needed

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
        score_threshold: float = None
    ) -> List[RetrievedChunk]:
        # retrieve top-k relevant chunks for a query
        top_k = top_k if top_k is not None else settings.retrieval_top_k
        collection_name = collection_name or settings.qdrant_collection_name
        # use default threshold only if not explicitly provided (None)
        if score_threshold is None:
            score_threshold = settings.similarity_threshold
        
        # generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
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

