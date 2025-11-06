"""
Feedback endpoint for collecting user feedback.
Stores feedback with trace_id for correlation with queries.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import json
from pathlib import Path
from datetime import datetime

from api.middleware.auth import verify_api_key
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Feedback storage directory
FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    trace_id: str = Field(..., description="Trace ID from query response")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 (optional)")
    thumbs_up: Optional[bool] = Field(None, description="Thumbs up (optional)")
    thumbs_down: Optional[bool] = Field(None, description="Thumbs down (optional)")
    text_feedback: Optional[str] = Field(None, description="Text feedback (optional)")
    query: Optional[str] = Field(None, description="Original query (optional)")
    answer: Optional[str] = Field(None, description="Answer received (optional)")


class FeedbackResponse(BaseModel):
    """Response model for feedback endpoint."""
    success: bool
    feedback_id: str
    message: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Submit user feedback for a query.
    
    Stores feedback with trace_id for correlation.
    Can include rating, thumbs up/down, and text feedback.
    """
    # Verify API key if configured
    if settings.api_key and settings.api_key != "your_api_key_here":
        verify_api_key(api_key)
    
    try:
        # Validate feedback
        if not any([request.rating, request.thumbs_up, request.thumbs_down, request.text_feedback]):
            raise HTTPException(
                status_code=400,
                detail="At least one feedback field must be provided"
            )
        
        # Create feedback entry
        feedback_entry = {
            "feedback_id": f"fb_{int(datetime.now().timestamp() * 1000)}",
            "trace_id": request.trace_id,
            "timestamp": datetime.now().isoformat(),
            "rating": request.rating,
            "thumbs_up": request.thumbs_up,
            "thumbs_down": request.thumbs_down,
            "text_feedback": request.text_feedback,
            "query": request.query,
            "answer": request.answer
        }
        
        # Save to JSONL file (one entry per line)
        feedback_file = FEEDBACK_DIR / "feedback.jsonl"
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        logger.info(f"Feedback submitted: {feedback_entry['feedback_id']} for trace {request.trace_id}")
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_entry['feedback_id'],
            message="Feedback submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.get("/feedback/{trace_id}")
async def get_feedback_by_trace(
    trace_id: str,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Get feedback for a specific trace_id.
    
    Returns all feedback entries associated with the trace.
    """
    # Verify API key if configured
    if settings.api_key and settings.api_key != "your_api_key_here":
        verify_api_key(api_key)
    
    try:
        feedback_file = FEEDBACK_DIR / "feedback.jsonl"
        
        if not feedback_file.exists():
            return {"trace_id": trace_id, "feedback": []}
        
        # Read all feedback entries
        feedback_entries = []
        with open(feedback_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("trace_id") == trace_id:
                        feedback_entries.append(entry)
        
        return {
            "trace_id": trace_id,
            "feedback": feedback_entries
        }
        
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving feedback: {str(e)}"
        )

