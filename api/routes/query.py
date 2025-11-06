"""
Query endpoint for RAG pipeline.
Handles user queries and returns answers with sources.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time

from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
from src.observability import create_trace, get_current_trace_id
from src.config import settings
from api.middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(default=5, description="Number of chunks to retrieve")
    prompt_version: Optional[str] = Field(default="v1", description="Prompt version to use")
    use_cache: Optional[bool] = Field(default=True, description="Whether to use cache")


class Source(BaseModel):
    """Source citation model."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[Source]
    trace_id: str
    latency: float
    token_usage: Dict[str, int]
    cost_estimate: float
    model_used: str
    prompt_version: str


@router.post("/query")
async def query_rag(
    request: QueryRequest,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Main query endpoint for RAG pipeline.
    
    Processes user query, retrieves relevant chunks, and generates answer.
    Returns answer with source citations and metadata.
    """
    # Verify API key if configured (skip if not set or is placeholder)
    if settings.api_key and settings.api_key != "your_api_key_here":
        try:
            verify_api_key(api_key)
        except HTTPException:
            raise
    
    # Create trace
    trace_id = create_trace(f"query_{int(time.time())}")
    
    try:
        start_time = time.time()
        
        # Initialize components
        retriever = DenseRetriever()
        generator = LLMGenerator()
        
        # Retrieve chunks (use score_threshold=0.0 to get all results, then filter if needed)
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_cache=request.use_cache,
            score_threshold=0.0  # Get all results, don't filter by threshold
        )
        
        if not chunks:
            # Return empty response instead of error for better UX
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents for your query. Please try rephrasing your question or ensure documents are indexed.",
                sources=[],
                trace_id=trace_id,
                latency=time.time() - start_time,
                token_usage={},
                cost_estimate=0.0,
                model_used="none",
                prompt_version=request.prompt_version
            )
        
        # Generate answer
        response = generator.generate(
            query=request.query,
            context_chunks=chunks,
            prompt_version=request.prompt_version,
            model_type='openai',
            fallback=True,
            use_cache=request.use_cache
        )
        
        latency = time.time() - start_time
        
        # Format sources
        sources = [
            Source(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=chunk.score,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            trace_id=trace_id,
            latency=latency,
            token_usage=response.token_usage,
            cost_estimate=response.cost_estimate,
            model_used=response.model_used,
            prompt_version=response.prompt_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

