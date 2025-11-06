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

## 📊 Observability and Metrics Testing (Task 6)

### Start Observability Stack

```bash
# Start all Docker services (Qdrant, Prometheus, Grafana)
docker-compose up -d

# Check all containers are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|rag_)"

# Verify services
curl http://localhost:6333/health  # Qdrant
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health  # Grafana
```

### Start Metrics Server

```bash
# Option 1: Start metrics server (keeps running)
python3 scripts/test_observability.py

# Option 2: Generate test metrics and start server
python3 scripts/generate_test_metrics.py

# Option 3: Start server in background
python3 -c "
from src.observability.metrics import metrics_manager
import time
metrics_manager.start_metrics_server()
print('Metrics server running on port 8000')
print('Press Ctrl+C to stop')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Stopping metrics server...')
" &
```

### Check Metrics Server Status

```bash
# Check if metrics endpoint is accessible
curl -s http://localhost:8000/metrics | head -30

# Check for RAG-specific metrics
curl -s http://localhost:8000/metrics | grep -E "^rag_" | head -20

# View all RAG metrics
curl -s http://localhost:8000/metrics | grep -E "^rag_"
```

### Check Prometheus Targets

```bash
# Check all Prometheus targets status (formatted)
curl -s 'http://localhost:9090/api/v1/targets' | python3 -c "import sys, json; data = json.load(sys.stdin); targets = data['data']['activeTargets']; print('Prometheus Targets Status:'); [print(f\"  {t['labels']['job']}: {t['health']} - {t.get('lastError', 'OK')}\") for t in targets]"

# Check targets (raw JSON)
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | head -40
```

### Query Prometheus for RAG Metrics

```bash
# List all available RAG metrics
curl -s 'http://localhost:9090/api/v1/label/__name__/values' | python3 -c "import sys, json; data = json.load(sys.stdin); rag_metrics = [m for m in data.get('data', []) if m.startswith('rag_')]; print('Available RAG Metrics:'); [print(f\"  - {m}\") for m in sorted(rag_metrics)]"

# Query specific metric (documents total)
curl -s 'http://localhost:9090/api/v1/query?query=rag_documents_total' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Documents Total:'); [print(f\"  Value: {r['value'][1]}\") for r in data.get('data', {}).get('result', [])]"

# Query token usage
curl -s 'http://localhost:9090/api/v1/query?query=rag_token_usage_total' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Token Usage Metrics:'); [print(f\"  {r['metric']}: {r['value'][1]}\") for r in data.get('data', {}).get('result', [])]"

# Query latency count
curl -s 'http://localhost:9090/api/v1/query?query=rag_request_latency_seconds_count' | python3 -c "import sys, json; data = json.load(sys.stdin); print('Latency Metrics Count:'); [print(f\"  {r['metric']}: {r['value'][1]}\") for r in data.get('data', {}).get('result', [])]"

# Sum of token usage
curl -s 'http://localhost:9090/api/v1/query?query=sum(rag_token_usage_total)' | python3 -c "import sys, json; data = json.load(sys.stdin); result = data.get('data', {}).get('result', []); print('Total Token Usage:', result[0]['value'][1] if result else 'No data yet')"
```

### Test Grafana Connection

```bash
# Test if Grafana can reach Prometheus from inside Docker
docker exec rag_grafana wget -qO- http://prometheus:9090/api/v1/status/config | head -5

# Check Prometheus config from Grafana container
docker exec rag_grafana wget -qO- http://prometheus:9090/api/v1/status/config 2>&1 | head -1
```

### Generate Test Metrics

```bash
# Generate test metrics using the script
python3 scripts/generate_test_metrics.py

# Generate metrics programmatically
python3 -c "
from src.observability.metrics import metrics_manager
from src.observability.instrumentation import record_token_usage, record_cost
import time

# Generate more metrics
for i in range(5):
    metrics_manager.record_latency('generation', 1.5 + i * 0.1)
    record_token_usage('openai', 'prompt', 2000 + i * 100)
    record_token_usage('openai', 'completion', 100 + i * 10)
    record_cost('openai', 'gpt-3.5-turbo', 0.004 + i * 0.0001)
    time.sleep(0.5)

print('✅ Generated 5 more metric batches')
"

# Generate real RAG pipeline metrics by running queries
python3 scripts/test_generation.py "What is bad science?" v1
python3 scripts/test_generation.py "What are the main problems with scientific research?" v1
```

### Comprehensive Status Check

```bash
# Full system status check
echo "=== Quick Status Check ===" && \
echo "" && \
echo "1. Metrics Server:" && \
curl -s http://localhost:8000/metrics > /dev/null && echo "   ✅ Running" || echo "   ❌ Not running" && \
echo "" && \
echo "2. Prometheus Targets:" && \
curl -s 'http://localhost:9090/api/v1/targets' | python3 -c "import sys, json; data = json.load(sys.stdin); targets = data['data']['activeTargets']; [print(f'   ✅ {t[\"labels\"][\"job\"]}: {t[\"health\"]}') for t in targets]" && \
echo "" && \
echo "3. Available RAG Metrics:" && \
curl -s 'http://localhost:9090/api/v1/label/__name__/values' | python3 -c "import sys, json; data = json.load(sys.stdin); rag = [m for m in data.get('data', []) if m.startswith('rag_')]; print(f'   Found {len(rag)} metrics')" && \
echo "" && \
echo "Access URLs:" && \
echo "  - Grafana: http://localhost:3000 (admin/admin)" && \
echo "  - Prometheus: http://localhost:9090" && \
echo "  - Metrics: http://localhost:8000/metrics"
```

### Verify Metrics from Prometheus Container

```bash
# Test if Prometheus container can reach metrics server
docker exec rag_prometheus wget -qO- http://host.docker.internal:8000/metrics | head -20
```

### Docker Container Management

```bash
# Check Docker containers status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|rag_)"

# Restart Grafana (after adding datasource config)
docker-compose restart grafana

# View logs
docker-compose logs prometheus
docker-compose logs grafana
docker-compose logs -f prometheus  # Follow logs
```

### Access Observability Dashboards

```bash
# Open in browser (or use these URLs):
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - Metrics endpoint: http://localhost:8000/metrics
# - Qdrant dashboard: http://localhost:6333/dashboard
```

### Quick Reference: Essential Observability Commands

```bash
# Start metrics server
python3 scripts/test_observability.py

# Check metrics server
curl http://localhost:8000/metrics | grep rag_

# Check Prometheus targets
curl 'http://localhost:9090/api/v1/targets' | python3 -m json.tool

# Test Grafana connection
docker exec rag_grafana wget -qO- http://prometheus:9090/api/v1/status/config

# Generate test data
python3 scripts/generate_test_metrics.py
python3 scripts/test_generation.py "your query" v1
```

## 🎯 Next Steps

After running the pipeline:
1. **Task 5**: Generator Module (LLM response generation) ✅
2. **Task 6**: Observability (OpenTelemetry, Prometheus, Grafana) ✅
3. **Task 7**: FastAPI Backend
4. **Task 8**: React Frontend
5. **Task 9**: Caching with Redis
6. **Task 10**: Evaluation Framework

---

**Last Updated**: Task 6 (Observability and Tracing) completed

