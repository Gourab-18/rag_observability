"""
Trace inspection endpoint.
Allows viewing trace details for debugging and analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, Dict, Any
import logging
import json
from pathlib import Path

from api.middleware.auth import verify_api_key
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trace/{trace_id}")
async def get_trace(
    trace_id: str,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Get trace information for a specific trace_id.
    
    Returns trace details including:
    - Trace ID
    - Timestamp
    - Query (if available)
    - Response metadata
    - Performance metrics
    """
    # Verify API key if configured (skip if placeholder)
    if settings.api_key and settings.api_key != "your_api_key_here":
        verify_api_key(api_key)
    
    try:
        # In a production system, you'd query OpenTelemetry backend
        # For now, return basic trace info
        # You could also store trace data in a database or file
        
        trace_data = {
            "trace_id": trace_id,
            "status": "available",
            "note": "Full trace data available via OpenTelemetry collector",
            "endpoints": {
                "prometheus": "http://localhost:9090",
                "grafana": "http://localhost:3000"
            }
        }
        
        # Try to find feedback for this trace
        feedback_file = Path("data/feedback/feedback.jsonl")
        if feedback_file.exists():
            feedback_entries = []
            with open(feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("trace_id") == trace_id:
                            feedback_entries.append(entry)
            
            if feedback_entries:
                trace_data["feedback"] = feedback_entries
        
        return trace_data
        
    except Exception as e:
        logger.error(f"Error retrieving trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving trace: {str(e)}"
        )

