#!/usr/bin/env python3


# script for installing 

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest import DocumentLoader, DocumentChunker
from src.config import settings, ManifestManager


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_document.py <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Loading document: {file_path}")
    
    # Load document
    loader = DocumentLoader()
    try:
        doc = loader.load_document(file_path)
        print(f"✓ Document loaded: {len(doc['text'])} characters")
    except Exception as e:
        print(f"✗ Error loading document: {e}")
        sys.exit(1)
    
    # Chunk document
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )
    
    doc_id = file_path.stem
    print(f"\nChunking document with chunk_size={chunker.chunk_size}, overlap={chunker.overlap}")
    
    try:
        chunks = chunker.chunk_document(
            doc['text'],
            doc_id,
            doc['metadata']
        )
        print(f"✓ Created {len(chunks)} chunks")
        
        # Print chunk statistics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  Chunk size range: {min(c.token_count for c in chunks)} - {max(c.token_count for c in chunks)} tokens")
        
    except Exception as e:
        print(f"✗ Error chunking document: {e}")
        sys.exit(1)
    
    # Save chunks to JSONL
    output_dir = Path(settings.data_processed_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_id}_chunks.jsonl"
    
    try:
        DocumentChunker.save_chunks(chunks, output_path)
        print(f"\n✓ Saved chunks to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving chunks: {e}")
        sys.exit(1)
    
    # Update manifest
    try:
        manifest_manager = ManifestManager()
        manifest_manager.add_document(
            doc_id=doc_id,
            file_name=file_path.name,
            file_path=str(file_path.absolute()),
            file_type=file_path.suffix,
            file_size=file_path.stat().st_size
        )
        manifest_manager.update_document_status(doc_id, "processed", len(chunks))
        print(f"✓ Updated manifest with document: {doc_id}")
    except Exception as e:
        print(f"⚠ Warning: Could not update manifest: {e}")
    
    print("\n✅ Document ingestion complete!")


if __name__ == "__main__":
    main()

