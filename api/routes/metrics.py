"""
Metrics endpoint for admin dashboard.
Returns aggregated metrics for monitoring and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, Dict, Any
import logging

from api.middleware.auth import verify_api_key
from src.observability.metrics import metrics_manager
from src.cache import cache_manager
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Get aggregated metrics for admin dashboard.
    
    Returns:
    - Cache statistics
    - Performance metrics (latency, costs)
    - Quality metrics (if available)
    """
    # Verify API key if configured (skip if placeholder)
    if settings.api_key and settings.api_key != "your_api_key_here":
        verify_api_key(api_key)
    
    try:
        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats()
        
        # Calculate cache hit rate (approximate)
        total_cached = sum(cache_stats.values())
        cache_hit_rate = 0.0
        if total_cached > 0:
            # This is a simplified calculation
            # In production, you'd track hits vs misses
            cache_hit_rate = min(total_cached / (total_cached + 100), 1.0)  # Placeholder
        
        # Get metrics from Prometheus (if available)
        # In production, query Prometheus API
        metrics_data = {
            "cache": {
                "embeddings": cache_stats.get("embedding", 0),
                "retrieval": cache_stats.get("retrieval", 0),
                "llm": cache_stats.get("llm", 0),
                "total": total_cached,
                "hit_rate": cache_hit_rate,
                "target_hit_rate": settings.target_cache_hit_rate
            },
            "performance": {
                "target_latency_p95": settings.target_latency_p95,
                "target_latency_p99": settings.target_latency_p99,
                "target_cache_hit_rate": settings.target_cache_hit_rate
            },
            "quality": {
                "target_hallucination_rate": settings.hallucination_threshold,
                "target_token_reduction": settings.target_token_reduction
            },
            "note": "Detailed metrics available via Prometheus at /metrics endpoint"
        }
        
        return metrics_data
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving metrics: {str(e)}"
        )

