#!/usr/bin/env python3
# test script for LLM generation

import os
import sys
from pathlib import Path

# Suppress TensorFlow/JAX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever
from src.generator import LLMGenerator
from src.config import settings


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_generation.py <query> [prompt_version]")
        print("Example: python scripts/test_generation.py 'What is your vision?' v1")
        sys.exit(1)
    
    query = sys.argv[1]
    prompt_version = sys.argv[2] if len(sys.argv) > 2 else 'v1'
    
    print(f"Query: {query}\n")
    print(f"Prompt Version: {prompt_version}\n")
    
    # Step 1: Retrieve chunks
    print("Step 1: Retrieving relevant chunks...")
    retriever = DenseRetriever()
    chunks = retriever.retrieve(query, top_k=5, score_threshold=0.0)
    
    print(f"✓ Retrieved {len(chunks)} chunks")
    if chunks:
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  {i}. Score: {chunk.score:.4f}, Chunk: {chunk.chunk_id}")
            print(f"     Text: {chunk.text[:100]}...\n")
    else:
        print("⚠ No chunks retrieved. Using empty context.\n")
    
    # Step 2: Generate response
    print("Step 2: Generating LLM response...")
    try:
        generator = LLMGenerator()
        response = generator.generate(
            query=query,
            context_chunks=chunks,
            prompt_version=prompt_version,
            model_type='openai',
            fallback=False
        )
        
        print(f"\n{'='*60}")
        print("Response:")
        print(f"{'='*60}")
        print(response.answer)
        print(f"\n{'='*60}")
        print("Metadata:")
        print(f"{'='*60}")
        print(f"  Model: {response.model_used}")
        print(f"  Prompt Version: {response.prompt_version}")
        print(f"  Trace ID: {response.trace_id}")
        print(f"  Sources: {len(response.sources)} chunks")
        print(f"  Token Usage:")
        print(f"    Prompt: {response.token_usage.get('prompt_tokens', 0)}")
        print(f"    Completion: {response.token_usage.get('completion_tokens', 0)}")
        print(f"    Total: {response.token_usage.get('total_tokens', 0)}")
        print(f"  Cost Estimate: ${response.cost_estimate:.4f}")
        print(f"\n✅ Generation complete!")
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

