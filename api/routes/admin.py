"""
Admin endpoints for managing the RAG pipeline.
Includes operations like clearing all context, resetting the database, etc.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from src.config import settings
from src.config.manifest_manager import ManifestManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.delete("/admin/clear-all")
async def clear_all_context() -> Dict[str, Any]:
    """
    Clear all context from the database.
    
    This will:
    - Delete all points from Qdrant collection
    - Clear the manifest
    - Optionally delete processed chunks (keeps raw files)
    
    WARNING: This is irreversible!
    """
    try:
        logger.warning("Clearing all context from database...")
        
        # Connect to Qdrant
        client = QdrantClient(url=settings.qdrant_url)
        
        # Get collection info before deletion
        try:
            collection_info = client.get_collection(settings.qdrant_collection_name)
            points_before = collection_info.points_count
        except Exception:
            points_before = 0
        
        # Delete all points from Qdrant
        if points_before > 0:
            # Get all point IDs and delete them
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=10000,  # Large limit to get all points
                with_payload=False,
                with_vectors=False
            )
            point_ids = [point.id for point in scroll_result[0]]
            if point_ids:
                client.delete(
                    collection_name=settings.qdrant_collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted {len(point_ids)} points from Qdrant")
        
        # Clear manifest
        manifest_manager = ManifestManager()
        manifest_file = Path("data/manifest.json")
        if manifest_file.exists():
            manifest_file.write_text("[]")
            logger.info("Cleared manifest.json")
        
        # Optionally clear processed chunks (keep raw files)
        processed_dir = Path(settings.data_processed_path)
        if processed_dir.exists():
            chunk_files = list(processed_dir.glob("*.jsonl"))
            for chunk_file in chunk_files:
                chunk_file.unlink()
            logger.info(f"Deleted {len(chunk_files)} processed chunk files")
        
        return {
            "success": True,
            "message": "All context cleared successfully",
            "deleted": {
                "qdrant_points": points_before,
                "processed_files": len(chunk_files) if processed_dir.exists() else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error clearing context: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing context: {str(e)}"
        )


@router.get("/admin/stats")
async def get_admin_stats() -> Dict[str, Any]:
    """
    Get statistics about the current context.
    """
    try:
        client = QdrantClient(url=settings.qdrant_url)
        
        # Get collection info
        try:
            collection_info = client.get_collection(settings.qdrant_collection_name)
            points_count = collection_info.points_count
        except Exception:
            points_count = 0
        
        # Get document count from manifest
        manifest_file = Path("data/manifest.json")
        processed_docs = 0
        if manifest_file.exists():
            import json
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                processed_docs = len([d for d in manifest if d.get('status') == 'processed'])
        
        # Count processed files
        processed_dir = Path(settings.data_processed_path)
        processed_files = len(list(processed_dir.glob("*.jsonl"))) if processed_dir.exists() else 0
        
        return {
            "qdrant_points": points_count,
            "processed_documents": processed_docs,
            "processed_files": processed_files,
            "collection_name": settings.qdrant_collection_name
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

