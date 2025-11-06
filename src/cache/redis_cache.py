"""
Redis caching layer for embeddings, retrieval results, and LLM responses.
Implements caching with TTLs to improve performance and reduce API costs.
"""

import json
import hashlib
import logging
from typing import Optional, List, Any, Dict
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache manager for RAG pipeline components."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client with connection pooling."""
        try:
            import redis
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache connected: {self.redis_url}")
        except ImportError:
            logger.warning("redis package not installed. Caching disabled.")
            self.client = None
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
            self.client = None
    
    def _generate_key(self, prefix: str, query: str) -> str:
        """Generate cache key from query string."""
        # Normalize query (lowercase, strip whitespace)
        normalized = query.lower().strip()
        # Create hash for long queries
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"rag:{prefix}:{query_hash}"
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to JSON bytes."""
        return json.dumps(embedding.tolist()).encode('utf-8')
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return np.array(json.loads(data.decode('utf-8')))
    
    def _serialize_retrieval(self, chunks: List[Any]) -> bytes:
        """Serialize retrieval results to JSON bytes."""
        # Convert RetrievedChunk objects to dicts
        serializable = []
        for chunk in chunks:
            if hasattr(chunk, '__dict__'):
                serializable.append({
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'score': chunk.score,
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id
                })
            else:
                serializable.append(chunk)
        return json.dumps(serializable).encode('utf-8')
    
    def _deserialize_retrieval(self, data: bytes) -> List[Dict]:
        """Deserialize bytes to retrieval results."""
        return json.loads(data.decode('utf-8'))
    
    def cache_query_embedding(self, query: str, embedding: np.ndarray) -> None:
        """
        Cache query embedding with TTL.
        
        Args:
            query: Query string
            embedding: Embedding vector
        """
        if not self.client:
            return
        
        try:
            key = self._generate_key("embedding", query)
            value = self._serialize_embedding(embedding)
            self.client.setex(key, settings.redis_ttl_embeddings, value)
            logger.debug(f"Cached embedding for query: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Get cached query embedding if available.
        
        Args:
            query: Query string
            
        Returns:
            Cached embedding or None
        """
        if not self.client:
            return None
        
        try:
            key = self._generate_key("embedding", query)
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache hit for embedding: {query[:50]}...")
                return self._deserialize_embedding(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
            return None
    
    def cache_retrieval_results(self, query: str, chunks: List[Any]) -> None:
        """
        Cache retrieval results with TTL.
        
        Args:
            query: Query string
            chunks: Retrieved chunks
        """
        if not self.client:
            return
        
        try:
            key = self._generate_key("retrieval", query)
            value = self._serialize_retrieval(chunks)
            self.client.setex(key, settings.redis_ttl_retrieval, value)
            logger.debug(f"Cached retrieval results for query: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache retrieval results: {e}")
    
    def get_retrieval_results(self, query: str) -> Optional[List[Dict]]:
        """
        Get cached retrieval results if available.
        
        Args:
            query: Query string
            
        Returns:
            Cached chunks or None
        """
        if not self.client:
            return None
        
        try:
            key = self._generate_key("retrieval", query)
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache hit for retrieval: {query[:50]}...")
                return self._deserialize_retrieval(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached retrieval results: {e}")
            return None
    
    def cache_llm_response(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache LLM response with TTL.
        
        Args:
            query: Query string
            response: LLM response dict (answer, sources, etc.)
        """
        if not self.client:
            return
        
        try:
            key = self._generate_key("llm", query)
            value = json.dumps(response).encode('utf-8')
            self.client.setex(key, settings.redis_ttl_llm, value)
            logger.debug(f"Cached LLM response for query: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache LLM response: {e}")
    
    def get_llm_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM response if available.
        
        Args:
            query: Query string
            
        Returns:
            Cached response dict or None
        """
        if not self.client:
            return None
        
        try:
            key = self._generate_key("llm", query)
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache hit for LLM response: {query[:50]}...")
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached LLM response: {e}")
            return None
    
    def clear_cache(self, prefix: str = None) -> None:
        """
        Clear cache entries.
        
        Args:
            prefix: Optional prefix to clear (e.g., "embedding", "retrieval", "llm")
                    If None, clears all RAG cache entries
        """
        if not self.client:
            return
        
        try:
            if prefix:
                pattern = f"rag:{prefix}:*"
            else:
                pattern = "rag:*"
            
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries for pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with counts of cached items by type
        """
        if not self.client:
            return {}
        
        try:
            stats = {}
            for prefix in ["embedding", "retrieval", "llm"]:
                pattern = f"rag:{prefix}:*"
                count = len(list(self.client.scan_iter(match=pattern)))
                stats[prefix] = count
            return stats
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}


# Global cache instance
cache_manager = RedisCache()

