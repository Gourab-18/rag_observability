#!/usr/bin/env python3
# script to generate embeddings and index chunks in Qdrant

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingest import Chunk, VectorIndexManager
from src.config import settings


def load_chunks_from_jsonl(jsonl_path: Path) -> list:
    # loads chunks from JSONL file
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk_data = json.loads(line)
            chunk = Chunk(
                doc_id=chunk_data['doc_id'],
                chunk_id=chunk_data['chunk_id'],
                text=chunk_data['text'],
                token_count=chunk_data['token_count'],
                metadata=chunk_data['metadata'],
                overlap_info=chunk_data['overlap_info']
            )
            chunks.append(chunk)
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/embed_and_index.py <chunks_file.jsonl>")
        print("Example: python scripts/embed_and_index.py data/processed/Vision-Life_chunks.jsonl")
        sys.exit(1)
    
    chunks_file = Path(sys.argv[1])
    
    if not chunks_file.exists():
        print(f"Error: File not found: {chunks_file}")
        sys.exit(1)
    
    print(f"Loading chunks from: {chunks_file}")
    
    try:
        chunks = load_chunks_from_jsonl(chunks_file)
        print(f"✓ Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Error loading chunks: {e}")
        sys.exit(1)
    
    print(f"\nConnecting to Qdrant at {settings.qdrant_url}")
    
    try:
        index_manager = VectorIndexManager()
    except Exception as e:
        print(f"✗ Error connecting to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose up -d")
        sys.exit(1)
    
    print(f"\nIndexing chunks in collection '{settings.qdrant_collection_name}'...")
    print("Using OpenAI embeddings...")
    
    try:
        index_manager.embed_and_index_chunks(chunks)
        print(f"\n✓ Successfully indexed {len(chunks)} chunks in Qdrant")
        
        # test query to validate
        print("\nTesting vector index with sample query...")
        test_query = chunks[0].text[:100] if chunks else "test"
        print(test_query)
        results = index_manager.query_chunks(test_query, top_k=3)
        print(f"✓ Query returned {len(results)} results")
        if results:
            print(f"  Top result score: {results[0]['score']:.4f}")
        
    except Exception as e:
        print(f"✗ Error indexing chunks: {e}")
        sys.exit(1)
    
    print("\n✅ Embedding and indexing complete!")


if __name__ == "__main__":
    main()

