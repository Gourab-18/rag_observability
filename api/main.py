"""
FastAPI application for RAG pipeline.
Provides REST API endpoints for querying, feedback, metrics, and trace inspection.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routes import query, feedback, metrics, trace, upload, admin, ab_test
from api.middleware.auth import verify_api_key
from src.config import settings

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="REST API for RAG (Retrieval Augmented Generation) pipeline",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
app.include_router(trace.router, prefix="/api", tags=["trace"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(ab_test.router, prefix="/api", tags=["ab-test"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

