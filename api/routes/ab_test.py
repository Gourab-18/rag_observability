"""
A/B testing endpoint for comparing prompt versions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.ab_testing import ABTester
from src.retrieval import DenseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ABTestRequest(BaseModel):
    """Request model for A/B test endpoint."""
    query: str = Field(..., description="User query to test")
    variants: Optional[List[str]] = Field(default=["v1", "v2"], description="Prompt versions to compare")
    top_k: Optional[int] = Field(default=5, description="Number of chunks to retrieve")
    model_type: Optional[str] = Field(default="openai", description="Model type to use")


class ABTestResponse(BaseModel):
    """Response model for A/B test endpoint."""
    query_id: str
    query: str
    prompt_version_a: str
    prompt_version_b: str
    winner: Optional[str]
    metrics_a: Dict[str, Any]
    metrics_b: Dict[str, Any]
    response_a: Dict[str, Any]
    response_b: Dict[str, Any]
    message: str


@router.post("/ab-test", response_model=ABTestResponse)
async def run_ab_test(request: ABTestRequest):
    """
    Run A/B test comparing different prompt versions.
    
    Compares prompt versions side-by-side and returns:
    - Response quality metrics
    - Cost comparison
    - Latency comparison
    - Winner determination
    """
    try:
        # Initialize components
        ab_tester = ABTester()
        retriever = DenseRetriever()
        
        # Retrieve chunks (use score_threshold=0.0 to get all results)
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_cache=True,
            score_threshold=0.0  # Get all results, don't filter by threshold
        )
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No relevant chunks found for the query: '{request.query}'. Please ensure documents are indexed in Qdrant."
            )
        
        # Run A/B test
        result = ab_tester.ab_test_prompt(
            query=request.query,
            context_chunks=chunks,
            variants=request.variants,
            model_type=request.model_type
        )
        
        # Format response
        return ABTestResponse(
            query_id=result.query_id,
            query=result.query,
            prompt_version_a=result.prompt_version_a,
            prompt_version_b=result.prompt_version_b,
            winner=result.winner,
            metrics_a=result.metrics_a,
            metrics_b=result.metrics_b,
            response_a={
                "answer": result.response_a.answer,
                "sources": len(result.response_a.sources),
                "trace_id": result.response_a.trace_id
            },
            response_b={
                "answer": result.response_b.answer,
                "sources": len(result.response_b.sources),
                "trace_id": result.response_b.trace_id
            },
            message=f"A/B test completed. Winner: {result.winner or 'tie'}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error running A/B test: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error running A/B test: {str(e)}"
        )


@router.get("/ab-test/results")
async def get_ab_test_results(limit: int = 10):
    """
    Get recent A/B test results.
    """
    try:
        ab_tester = ABTester()
        analysis = ab_tester.analyze_results(limit=limit)
        
        return {
            "summary": analysis,
            "message": "A/B test results retrieved successfully"
        }
    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting results: {str(e)}"
        )

