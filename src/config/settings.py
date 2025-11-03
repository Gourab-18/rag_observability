"""
Configuration management with environment variables.
Centralized settings for the RAG pipeline.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Vector DB Configuration
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field(default="rag_collection", env="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=1536, env="QDRANT_VECTOR_SIZE")  # text-embedding-3-small
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Local LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_ttl_embeddings: int = Field(default=86400, env="REDIS_TTL_EMBEDDINGS")  # 24h in seconds
    redis_ttl_retrieval: int = Field(default=3600, env="REDIS_TTL_RETRIEVAL")  # 1h in seconds
    redis_ttl_llm: int = Field(default=21600, env="REDIS_TTL_LLM")  # 6h in seconds
    
    # Chunking Configuration
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")  # tokens
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")  # tokens
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=10, env="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    use_reranking: bool = Field(default=True, env="USE_RERANKING")
    
    # Observability Configuration
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    otlp_endpoint: Optional[str] = Field(default=None, env="OTLP_ENDPOINT")
    
    # API Configuration
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    rate_limit_per_hour: int = Field(default=100, env="RATE_LIMIT_PER_HOUR")
    
    # Evaluation Configuration
    precision_at_k: List[int] = Field(default=[1, 3, 5], env="PRECISION_AT_K")
    hallucination_threshold: float = Field(default=0.05, env="HALLUCINATION_THRESHOLD")  # 5%
    target_latency_p95: float = Field(default=2.0, env="TARGET_LATENCY_P95")  # seconds
    target_latency_p99: float = Field(default=3.0, env="TARGET_LATENCY_P99")  # seconds
    target_cache_hit_rate: float = Field(default=0.7, env="TARGET_CACHE_HIT_RATE")  # 70%
    target_token_reduction: float = Field(default=0.3, env="TARGET_TOKEN_REDUCTION")  # 30%
    
    # Data Paths
    data_raw_path: str = Field(default="data/raw", env="DATA_RAW_PATH")
    data_processed_path: str = Field(default="data/processed", env="DATA_PROCESSED_PATH")
    data_manifest_path: str = Field(default="data/manifest.json", env="DATA_MANIFEST_PATH")
    prompts_path: str = Field(default="prompts", env="PROMPTS_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

