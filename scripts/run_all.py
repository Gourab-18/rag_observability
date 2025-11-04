#!/usr/bin/env python3
"""
Master script to run the entire RAG pipeline end-to-end.
Usage: python scripts/run_all.py [--skip-docker] [--skip-ingest] [--skip-index] [--skip-retrieval]
"""

import sys
import subprocess
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent


def check_docker():
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def check_qdrant():
    """Check if Qdrant is accessible."""
    try:
        import requests
        response = requests.get("http://localhost:6333/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def run_command(cmd, description, check=True):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=project_root,
        capture_output=False
    )
    
    if result.returncode != 0 and check:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Success: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run entire RAG pipeline")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker/Qdrant setup")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip document ingestion")
    parser.add_argument("--skip-index", action="store_true", help="Skip embedding and indexing")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval test")
    parser.add_argument("--query", type=str, default="What is your vision?", help="Query for retrieval test")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 RAG Pipeline - End-to-End Execution")
    print("="*60)
    
    # Step 1: Docker/Qdrant Setup
    if not args.skip_docker:
        if not check_docker():
            print("\n❌ Docker is not running. Please start Docker Desktop.")
            print("   Then run: docker-compose up -d")
            sys.exit(1)
        
        if not check_qdrant():
            print("\n📦 Starting Qdrant...")
            if not run_command("docker-compose up -d", "Start Qdrant container", check=False):
                print("⚠️  Qdrant may already be running or failed to start")
        else:
            print("\n✅ Qdrant is already running")
    else:
        print("\n⏭️  Skipping Docker setup")
    
    # Step 2: Document Ingestion
    if not args.skip_ingest:
        pdf_files = list((project_root / "data/raw").glob("*.pdf"))
        if not pdf_files:
            print("\n⚠️  No PDF files found in data/raw/")
        else:
            for pdf_file in pdf_files:
                run_command(
                    f"python3 scripts/ingest_document.py {pdf_file}",
                    f"Ingest document: {pdf_file.name}"
                )
    else:
        print("\n⏭️  Skipping document ingestion")
    
    # Step 3: Embedding and Indexing
    if not args.skip_index:
        chunk_files = list((project_root / "data/processed").glob("*_chunks.jsonl"))
        if not chunk_files:
            print("\n⚠️  No chunk files found in data/processed/")
        else:
            for chunk_file in chunk_files:
                run_command(
                    f"python3 scripts/embed_and_index.py {chunk_file}",
                    f"Index chunks: {chunk_file.name}"
                )
    else:
        print("\n⏭️  Skipping embedding and indexing")
    
    # Step 4: Retrieval Test
    if not args.skip_retrieval:
        run_command(
            f"python3 scripts/test_retrieval_simple.py '{args.query}'",
            f"Test retrieval with query: '{args.query}'"
        )
    else:
        print("\n⏭️  Skipping retrieval test")
    
    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Test with your own query: python3 scripts/test_retrieval.py 'your query'")
    print("  - Check Qdrant dashboard: http://localhost:6333/dashboard")
    print("  - View manifest: cat data/manifest.json")


if __name__ == "__main__":
    main()

