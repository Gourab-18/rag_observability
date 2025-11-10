# embedding generation and Qdrant upsert operations

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from src.config import settings
from src.utils.embeddings import EmbeddingGenerator
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorIndexManager:
    # manages vector index operations with Qdrant
    
    def __init__(self):
        # initialize Qdrant client with retry logic
        self.client = self._create_client_with_retry()
        self.embedding_generator = EmbeddingGenerator()
    
    def _create_client_with_retry(self, max_retries: int = 3, delay: int = 2) -> QdrantClient:
        # creates Qdrant client with retry logic
        for attempt in range(max_retries):
            try:
                client = QdrantClient(url=settings.qdrant_url)
                # test connection
                collections = client.get_collections()
                logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
                return client
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                    raise
    
    def create_collection(self, collection_name: str, vector_size: int):
        # creates Qdrant collection with proper schema
        try:
            # check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return
            
            # create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def upsert_to_qdrant(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        collection_name: str
    ):
        # upserts vectors with metadata to Qdrant
        if len(vectors) != len(metadata):
            raise ValueError(f"Vectors ({len(vectors)}) and metadata ({len(metadata)}) must have same length")
        
        try:
            # prepare points for upsert
            points = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                point = PointStruct(
                    id=i, 
                    vector=vector,
                    payload=meta
                )
                points.append(point)
            
            # batch upsert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} vectors to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Error upserting to Qdrant: {e}")
            raise
    
    def embed_and_index_chunks(
        self,
        chunks: List[Chunk],
        collection_name: str = None
    ):
        # generates embeddings and indexes chunks in Qdrant
        collection_name = collection_name or settings.qdrant_collection_name
        
        # create collection if it doesn't exist
        self.create_collection(collection_name, settings.qdrant_vector_size)
        
        # extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks using OpenAI...")
        
        # generate embeddings using OpenAI
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # prepare metadata (include chunk text for retrieval)
        metadata_list = []
        for chunk in chunks:
            metadata_list.append({
                'doc_id': chunk.doc_id,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,  # store text in payload for retrieval
                'source': chunk.metadata.get('file_path', ''),
                'chunk_pos': chunk.overlap_info.get('chunk_position', 0),
                'created_at': datetime.now().isoformat()
            })
        
        # upsert to Qdrant
        self.upsert_to_qdrant(
            embeddings.tolist(),
            metadata_list,
            collection_name
        )
    
    def query_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        # queries Qdrant to validate vector index returns relevant chunks
        collection_name = collection_name or settings.qdrant_collection_name
        
        # generate query embedding using OpenAI
        query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]
        
        try:
            # search in Qdrant
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            # format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'score': result.score,
                    'payload': result.payload,
                    'id': result.id
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            raise

