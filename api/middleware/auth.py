"""
Authentication middleware for API endpoints.
Supports API key authentication.
"""

from fastapi import HTTPException, Header
from typing import Optional

from src.config import settings


def verify_api_key(api_key: Optional[str] = None):
    """
    Verify API key from header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not settings.api_key:
        # API key not configured, allow access
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

