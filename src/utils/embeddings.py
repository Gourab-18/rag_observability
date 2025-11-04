# embedding generation using OpenAI

from typing import List
import numpy as np
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    # generates embeddings using OpenAI API
    
    def __init__(self):
        pass
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # generates embeddings for a list of texts using OpenAI
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=settings.openai_api_key)
            
            if not client.api_key:
                raise ValueError("OpenAI API key not found in settings")
            
            # batch process texts
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts
            )
            
            # extract embeddings
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> np.ndarray:
        # generates embeddings in batches for large datasets
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.vstack(all_embeddings)

