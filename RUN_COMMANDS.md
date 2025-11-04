# RAG Observability - Complete Run Commands

This document contains all commands to run the entire RAG pipeline.

## 📋 Quick Start (One Command)

```bash
# Run everything automatically
python3 scripts/run_all.py
```

## 🔧 Step-by-Step Commands

### 1. Initial Setup (One-time)

```bash
# Navigate to project
cd /Users/gourabnanda/Desktop/ML/rag_observability

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant (Vector Database)

```bash
# Start Qdrant container
docker-compose up -d

# Verify it's running
docker ps | grep qdrant

# Check health
curl http://localhost:6333/health

# View logs (if needed)
docker-compose logs qdrant
```

### 3. Document Ingestion (Task 2)

```bash
# Process a single document
python3 scripts/ingest_document.py data/raw/Vision-Life.pdf

# Process all PDFs in data/raw/
for pdf in data/raw/*.pdf; do
    echo "Processing: $pdf"
    python3 scripts/ingest_document.py "$pdf"
done

# Process a text file
python3 scripts/ingest_document.py data/raw/example.txt
```

**What it does:**
- Loads document (PDF or text)
- Chunks it into smaller pieces
- Saves chunks to `data/processed/{doc_id}_chunks.jsonl`
- Updates `data/manifest.json`

### 4. Embedding and Indexing (Task 3)

```bash
# Index chunks from a document
python3 scripts/embed_and_index.py data/processed/Vision-Life_chunks.jsonl

# Index all chunk files
for chunks_file in data/processed/*_chunks.jsonl; do
    echo "Indexing: $chunks_file"
    python3 scripts/embed_and_index.py "$chunks_file"
done
```

**What it does:**
- Generates OpenAI embeddings for each chunk
- Stores embeddings in Qdrant vector database
- Makes chunks searchable

### 5. Retrieval Test (Task 4)

```bash
# Test retrieval with a query
python3 scripts/test_retrieval.py "What is your vision?"

# Test with different queries
python3 scripts/test_retrieval.py "What are the main points?"
python3 scripts/test_retrieval.py "Tell me about the future"
```

**What it does:**
- Retrieves relevant chunks from Qdrant
- Optionally reranks results (if enabled)
- Compresses/filters chunks
- Shows top results with scores

## 🚀 Complete End-to-End Workflow

```bash
# 1. Setup (one-time)
cd /Users/gourabnanda/Desktop/ML/rag_observability
pip install -r requirements.txt

# 2. Start Qdrant
docker-compose up -d

# 3. Ingest document
python3 scripts/ingest_document.py data/raw/Vision-Life.pdf

# 4. Index chunks
python3 scripts/embed_and_index.py data/processed/Vision-Life_chunks.jsonl

# 5. Test retrieval
python3 scripts/test_retrieval.py "What is your vision?"
```

## 📝 Individual Script Commands

### Document Ingestion
```bash
# Basic usage
python3 scripts/ingest_document.py <file_path>

# Examples
python3 scripts/ingest_document.py data/raw/Vision-Life.pdf
python3 scripts/ingest_document.py data/raw/document.txt
```

### Embedding and Indexing
```bash
# Basic usage
python3 scripts/embed_and_index.py <chunks_file>

# Examples
python3 scripts/embed_and_index.py data/processed/Vision-Life_chunks.jsonl
```

### Retrieval Testing
```bash
# Basic usage
python3 scripts/test_retrieval.py <query>

# Examples
python3 scripts/test_retrieval.py "What is your vision?"
python3 scripts/test_retrieval.py "Explain the main concepts"
```

## 🔍 Verification Commands

```bash
# Check chunks were created
ls -lh data/processed/

# View manifest
cat data/manifest.json | python3 -m json.tool

# Check Qdrant collections
curl http://localhost:6333/collections

# Test Qdrant query (Python)
python3 -c "
from src.ingest import VectorIndexManager
manager = VectorIndexManager()
results = manager.query_chunks('What is your vision?', top_k=3)
print(f'Found {len(results)} results')
for r in results:
    print(f\"Score: {r['score']:.4f}, Chunk: {r['payload']['chunk_id']}\")
"

# Test retrieval (Python)
python3 -c "
from src.retrieval import DenseRetriever
retriever = DenseRetriever()
chunks = retriever.retrieve('What is your vision?', top_k=3)
print(f'Retrieved {len(chunks)} chunks')
for chunk in chunks:
    print(f\"Score: {chunk.score:.4f}, Text: {chunk.text[:100]}...\")
"
```

## 🛠️ Maintenance Commands

```bash
# Stop Qdrant
docker-compose down

# Restart Qdrant
docker-compose restart

# View Qdrant logs
docker-compose logs qdrant
docker-compose logs -f qdrant  # Follow logs

# Check Docker status
docker-compose ps

# Remove Qdrant container and data (careful!)
docker-compose down -v
```

## 🔧 Configuration

Edit `.env` file to change settings:
```bash
# View current settings
cat .env

# Key settings:
# - OPENAI_API_KEY: Your OpenAI API key (required)
# - QDRANT_URL: Qdrant URL (default: http://localhost:6333)
# - RETRIEVAL_TOP_K: Number of chunks to retrieve (default: 10)
# - SIMILARITY_THRESHOLD: Minimum similarity score (default: 0.7)
# - USE_RERANKING: Enable reranking (default: false)
```

## 🐛 Troubleshooting

```bash
# If Qdrant not responding
docker-compose restart

# If import errors
pip install --upgrade -r requirements.txt

# If API key not found
python3 -c "from src.config import settings; print('Key loaded:', bool(settings.openai_api_key))"

# Check Python environment
which python3
python3 --version

# Check if Docker is running
docker ps

# Check Qdrant health
curl http://localhost:6333/health
```

## 📊 Useful One-Liners

```bash
# Complete workflow in one command
cd /Users/gourabnanda/Desktop/ML/rag_observability && \
pip install -q -r requirements.txt && \
docker-compose up -d && \
sleep 5 && \
python3 scripts/ingest_document.py data/raw/Vision-Life.pdf && \
python3 scripts/embed_and_index.py data/processed/Vision-Life_chunks.jsonl && \
python3 scripts/test_retrieval.py "What is your vision?"

# Process all PDFs and index them
for pdf in data/raw/*.pdf; do
    python3 scripts/ingest_document.py "$pdf" && \
    python3 scripts/embed_and_index.py "data/processed/$(basename $pdf .pdf)_chunks.jsonl"
done

# Test multiple queries
for query in "What is vision?" "What are the goals?" "Tell me about the future"; do
    echo "Query: $query"
    python3 scripts/test_retrieval.py "$query"
    echo ""
done
```

## 🎯 Next Steps

After running the pipeline:
1. **Task 5**: Generator Module (LLM response generation)
2. **Task 6**: Observability (OpenTelemetry, Prometheus, Grafana)
3. **Task 7**: FastAPI Backend
4. **Task 8**: React Frontend
5. **Task 9**: Caching with Redis
6. **Task 10**: Evaluation Framework

---

**Last Updated**: Task 4 (Retriever and Compressor Module) completed

