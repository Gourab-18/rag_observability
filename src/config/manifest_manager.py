# extracts information from manifest.json
# controls everything related to the manifest.json file

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.config import settings


class ManifestManager:
    
    def __init__(self, manifest_path: str = None):

        self.manifest_path = Path(manifest_path or settings.data_manifest_path)
        self._ensure_manifest_exists()

        # creates manifest file if it doesn't exist
    
    def _ensure_manifest_exists(self):
        if not self.manifest_path.exists():
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, 'w') as f:
                json.dump([], f)
    

# loads manifest from file

    def load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    # saves manifest to file
    def save_manifest(self, manifest: List[Dict[str, Any]]):
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

  # adds a new document to the manifest ( returns document metadata )
    def add_document(
        self,
        doc_id: str,
        file_name: str,
        file_path: str,
        file_type: str,
        file_size: int,
        title: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        manifest = self.load_manifest()
        # schema for document
        doc_metadata = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_type,
            "file_size": file_size,
            "upload_date": datetime.now().isoformat() + "Z",
            "title": title or file_name,
            "author": author or "Unknown",
            "tags": tags or [],
            "status": "pending",
            "chunk_count": 0,
            "processed_date": None
        }
        
        manifest.append(doc_metadata)
        self.save_manifest(manifest)
        
        return doc_metadata
    
    # updates document wrt id
    def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: Optional[int] = None
    ):

        manifest = self.load_manifest()
        
        for doc in manifest:
            if doc["doc_id"] == doc_id:
                doc["status"] = status
                if chunk_count is not None:
                    doc["chunk_count"] = chunk_count
                if status == "processed":
                    doc["processed_date"] = datetime.now().isoformat() + "Z"
                break
        
        self.save_manifest(manifest)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:

        manifest = self.load_manifest()
        for doc in manifest:
            if doc["doc_id"] == doc_id:
                return doc
        return None
    
    def list_documents(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        manifest = self.load_manifest()
        if status:
            return [doc for doc in manifest if doc["status"] == status]
        return manifest

