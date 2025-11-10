#!/usr/bin/env python3
# Simplified retrieval test - OpenAI only, no local models

import os
import sys
from pathlib import Path

# Suppress TensorFlow/JAX warnings BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Minimal imports - only what we need
from src.retrieval.retriever import DenseRetriever, RetrievedChunk
from src.retrieval.compressor import ChunkCompressor
from src.config import settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_retrieval_simple.py <query>")
        print("Example: python scripts/test_retrieval_simple.py 'What is your vision?'")
        sys.exit(1)
    
    query = sys.argv[1]
    
    print(f"Query: {query}\n")
    
    # retrieve chunks
    print("Step 1: Dense retrieval from Qdrant...")
    try:
        retriever = DenseRetriever()
        # use 0.0 threshold to show all results
        chunks = retriever.retrieve(query, top_k=10, score_threshold=0.0)
        
        print(f"✓ Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  {i}. Score: {chunk.score:.4f}, Chunk: {chunk.chunk_id}")
            print(f"     Text preview: {chunk.text[:100]}...\n")
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Skip reranking - it's disabled anyway
    print("Step 2: Reranking skipped (disabled in settings)\n")
    
    # compression
    print("Step 3: Compressing chunks (score filtering)...")
    try:
        compressor = ChunkCompressor()
        print("chunks")
        print(chunks)
        compressed_chunks = compressor.compress_chunks(chunks, strategy='bm25')
        print("compressed_chunks")
        print(compressed_chunks)
        
        print(f"✓ Compressed: {len(chunks)} -> {len(compressed_chunks)} chunks")
        
        if len(chunks) > 0:
            reduction = compressor.calculate_token_reduction(chunks, compressed_chunks)
            print(f"  Token reduction: {reduction*100:.1f}%")
    except Exception as e:
        print(f"⚠ Compression error: {e}")
    
    print("\n✅ Retrieval test complete!")


if __name__ == "__main__":
    main()

