"""
File upload endpoint for document ingestion.
Allows users to upload PDFs and text files for indexing.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.middleware.auth import verify_api_key
from src.config import settings
from src.ingest import DocumentLoader, DocumentChunker
from src.ingest.embed_and_upsert import VectorIndexManager
from src.config.manifest_manager import ManifestManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed file types
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md'}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Upload and index a document file.
    
    Accepts PDF, TXT, or MD files.
    Processes the file, chunks it, generates embeddings, and indexes in Qdrant.
    """
    # Verify API key if configured
    if settings.api_key and settings.api_key != "your_api_key_here":
        try:
            verify_api_key(api_key)
        except HTTPException:
            raise
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    file_path = None
    try:
        # Save uploaded file
        upload_dir = Path(settings.data_raw_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        logger.info(f"Starting file upload: {file.filename} ({file.size} bytes)")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        
        # Process the file
        logger.info(f"Loading document: {file_path}")
        loader = DocumentLoader()
        doc = loader.load_document(file_path)
        logger.info(f"Document loaded: {len(doc['text'])} characters")
        
        # Chunk the document
        logger.info("Chunking document...")
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        
        doc_id = file_path.stem
        chunks = chunker.chunk_document(
            doc['text'],
            doc_id,
            doc['metadata']
        )
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks
        output_dir = Path(settings.data_processed_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc_id}_chunks.jsonl"
        DocumentChunker.save_chunks(chunks, output_path)
        logger.info(f"Chunks saved to: {output_path}")
        
        # Index in Qdrant (this is the slow part - generating embeddings)
        # Estimate: ~0.5s per chunk for OpenAI embeddings
        estimated_time = len(chunks) * 0.5
        logger.info(f"Generating embeddings for {len(chunks)} chunks (estimated: {estimated_time:.1f}s)...")
        
        index_manager = VectorIndexManager()
        index_manager.embed_and_index_chunks(chunks)
        logger.info(f"Indexing complete for {len(chunks)} chunks")
        
        # Update manifest
        manifest_manager = ManifestManager()
        manifest_manager.add_document(
            doc_id=doc_id,
            file_name=file.filename,
            file_path=str(file_path.absolute()),
            file_type=file_ext,
            file_size=file_path.stat().st_size
        )
        manifest_manager.update_document_status(doc_id, "processed", len(chunks))
        
        return {
            "success": True,
            "message": f"File uploaded and indexed successfully",
            "doc_id": doc_id,
            "chunks": len(chunks),
            "file_path": str(file_path),
            "chunks_file": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        # Clean up uploaded file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

