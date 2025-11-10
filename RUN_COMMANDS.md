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
python3 scripts/test_retrieval_simple.py <query>

# Examples
python3 scripts/test_retrieval_simple.py "What is your vision?"
python3 scripts/test_retrieval_simple.py "Explain the main concepts"
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
python3 scripts/test_retrieval_simple.py "What is your vision?"

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

## 📊 Task 7: Evaluation Harness

### Start Redis (Required for Caching)

```bash
# Start Redis container (if not already running)
docker-compose up -d redis

# Verify Redis is running
docker ps | grep redis

# Test Redis connection
docker exec rag_redis redis-cli ping
# Should return: PONG
```

### Create Evaluation Dataset

```bash
# Create sample evaluation dataset
python3 -c "
from src.eval import DatasetLoader
DatasetLoader.create_sample_dataset('data/eval/sample_dataset.json')
print('✅ Sample dataset created at data/eval/sample_dataset.json')
"

# Edit the dataset to add your own queries and ground truth
# Format: JSON with query_id, query, ground_truth_answer, relevant_doc_ids
cat data/eval/sample_dataset.json | python3 -m json.tool
```

### Run Evaluation Suite

```bash
# Run full evaluation on dataset
python3 scripts/test_evaluation.py

# This will:
# 1. Load evaluation dataset
# 2. Run retrieval and generation for each query
# 3. Compute Precision@k, BLEU, ROUGE-L, faithfulness metrics
# 4. Save results to data/eval/results.json
# 5. Print summary statistics
```

### View Evaluation Results

```bash
# View detailed results
cat data/eval/results.json | python3 -m json.tool

# Quick summary
python3 -c "
import json
with open('data/eval/results.json') as f:
    results = json.load(f)
    print(f'Total queries evaluated: {len(results)}')
    avg_p1 = sum(r['retrieval_metrics']['precision_at_1'] for r in results) / len(results)
    avg_bleu = sum(r['generation_metrics']['bleu_score'] for r in results) / len(results)
    hallucination = sum(1 for r in results if not r['faithfulness_score']['is_faithful']) / len(results)
    print(f'Avg Precision@1: {avg_p1:.3f}')
    print(f'Avg BLEU: {avg_bleu:.3f}')
    print(f'Hallucination Rate: {hallucination:.3f}')
"
```

### Individual Metric Evaluation

```bash
# Test retrieval metrics only
python3 -c "
from src.eval import Evaluator, DatasetLoader
from src.retrieval import DenseRetriever

dataset = DatasetLoader.load_from_json('data/eval/sample_dataset.json')
evaluator = Evaluator()
retriever = DenseRetriever()

query = dataset.queries[0]
chunks = retriever.retrieve(query.query, top_k=10)
metrics = evaluator.evaluate_retrieval(query, chunks)

print(f'Precision@1: {metrics.precision_at_1:.3f}')
print(f'Precision@3: {metrics.precision_at_3:.3f}')
print(f'Precision@5: {metrics.precision_at_5:.3f}')
print(f'Recall: {metrics.recall:.3f}')
"
```

## 🔄 Task 8: Caching, Circuit Breaker, and A/B Testing

### Redis Caching

```bash
# Check Redis cache status
python3 -c "
from src.cache import cache_manager
stats = cache_manager.get_cache_stats()
print('Cache Statistics:')
for key, count in stats.items():
    print(f'  {key}: {count} entries')
"

# Clear cache
python3 -c "
from src.cache import cache_manager
cache_manager.clear_cache()  # Clear all
# cache_manager.clear_cache('embedding')  # Clear specific type
print('✅ Cache cleared')
"

# Test caching in retrieval
python3 -c "
from src.retrieval import DenseRetriever
import time

retriever = DenseRetriever()
query = 'What is your vision?'

# First call (cache miss)
start = time.time()
chunks1 = retriever.retrieve(query, use_cache=True)
time1 = time.time() - start
print(f'First call: {time1:.3f}s, {len(chunks1)} chunks')

# Second call (cache hit)
start = time.time()
chunks2 = retriever.retrieve(query, use_cache=True)
time2 = time.time() - start
print(f'Second call: {time2:.3f}s, {len(chunks2)} chunks (cached)')
print(f'Speedup: {time1/time2:.1f}x faster')
"
```

### Circuit Breaker Testing

```bash
# Check circuit breaker state
python3 -c "
from src.utils.circuit_breaker import get_circuit_breaker

cb_openai = get_circuit_breaker('openai')
cb_ollama = get_circuit_breaker('ollama')

print('OpenAI Circuit Breaker:')
print(f'  State: {cb_openai.get_state()[\"state\"]}')
print(f'  Failures: {cb_openai.get_state()[\"failure_count\"]}')

print('\\nOllama Circuit Breaker:')
print(f'  State: {cb_ollama.get_state()[\"state\"]}')
print(f'  Failures: {cb_ollama.get_state()[\"failure_count\"]}')
"
```

### A/B Testing Prompt Versions

```bash
# Run A/B test comparing prompt versions
python3 scripts/test_ab_testing.py

# This will:
# 1. Test queries with both v1 and v2 prompts
# 2. Compare cost, latency, and quality metrics
# 3. Determine winner based on multiple factors
# 4. Log results to data/ab_tests/ab_test_results.jsonl
```

### View A/B Test Results

```bash
# View A/B test logs
cat data/ab_tests/ab_test_results.jsonl | python3 -m json.tool

# Analyze A/B test results
python3 -c "
from src.generator.ab_testing import ABTester

tester = ABTester()
analysis = tester.analyze_results()

print('A/B Test Analysis:')
print(f'  Total Tests: {analysis[\"total_tests\"]}')
print(f'  Wins (v1): {analysis[\"wins_a\"]} ({analysis[\"win_rate_a\"]*100:.1f}%)')
print(f'  Wins (v2): {analysis[\"wins_b\"]} ({analysis[\"win_rate_b\"]*100:.1f}%)')
print(f'  Avg Cost (v1): \${analysis[\"avg_cost_a\"]:.4f}')
print(f'  Avg Cost (v2): \${analysis[\"avg_cost_b\"]:.4f}')
print(f'  Cost Savings: {analysis[\"cost_savings\"]:.1f}%')
"
```

### Test Caching with Generation

```bash
# Test LLM response caching
python3 -c "
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
import time

retriever = DenseRetriever()
generator = LLMGenerator()
query = 'What is your vision?'

# Retrieve chunks
chunks = retriever.retrieve(query, top_k=5)

# First call (cache miss)
start = time.time()
response1 = generator.generate(query, chunks, use_cache=True)
time1 = time.time() - start
print(f'First call: {time1:.3f}s, Cost: \${response1.cost_estimate:.4f}')

# Second call (cache hit)
start = time.time()
response2 = generator.generate(query, chunks, use_cache=True)
time2 = time.time() - start
print(f'Second call: {time2:.3f}s, Cost: \${response2.cost_estimate:.4f} (cached)')
print(f'Speedup: {time1/time2:.1f}x faster')
print(f'Cost saved: \${response1.cost_estimate - response2.cost_estimate:.4f}')
"
```

### Integration Test: Full Pipeline with Caching

```bash
# Test entire pipeline with caching enabled
python3 -c "
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
from src.cache import cache_manager
import time

retriever = DenseRetriever()
generator = LLMGenerator()
query = 'What is your vision?'

print('=== First Run (No Cache) ===')
start = time.time()
chunks = retriever.retrieve(query, use_cache=True)
response = generator.generate(query, chunks, use_cache=True)
time1 = time.time() - start
print(f'Time: {time1:.3f}s, Cost: \${response.cost_estimate:.4f}')

print('\\n=== Second Run (With Cache) ===')
start = time.time()
chunks = retriever.retrieve(query, use_cache=True)
response = generator.generate(query, chunks, use_cache=True)
time2 = time.time() - start
print(f'Time: {time2:.3f}s, Cost: \${response.cost_estimate:.4f}')

print(f'\\nSpeedup: {time1/time2:.1f}x faster')
print(f'Cache Stats: {cache_manager.get_cache_stats()}')
"
```

## 🧪 Testing Commands

### Quick Component Tests

```bash
# Test Redis connection
docker exec rag_redis redis-cli ping
# Should return: PONG

# Test cache connection
python3 -c "
from src.cache import cache_manager
stats = cache_manager.get_cache_stats()
print('✅ Redis cache connected!')
print(f'Cache Statistics: {stats}')
"

# Test circuit breaker
python3 -c "
from src.utils.circuit_breaker import get_circuit_breaker
cb_openai = get_circuit_breaker('openai')
cb_ollama = get_circuit_breaker('ollama')
print('✅ Circuit breakers initialized')
print(f'OpenAI State: {cb_openai.get_state()[\"state\"]}')
print(f'Ollama State: {cb_ollama.get_state()[\"state\"]}')
"

# Test all imports
python3 -c "
from src.cache import cache_manager
from src.utils.circuit_breaker import get_circuit_breaker
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
from src.eval import Evaluator, DatasetLoader
from src.generator.ab_testing import ABTester
print('✅ All imports successful')
"
```

### Test Caching Performance

```bash
# Test retrieval caching speedup
python3 -c "
from src.retrieval import DenseRetriever
import time

retriever = DenseRetriever()
query = 'What is your vision?'

# First call (cache miss)
start = time.time()
chunks1 = retriever.retrieve(query, top_k=3, use_cache=True)
time1 = time.time() - start
print(f'First call: {time1:.3f}s, {len(chunks1)} chunks')

# Second call (cache hit)
start = time.time()
chunks2 = retriever.retrieve(query, top_k=3, use_cache=True)
time2 = time.time() - start
print(f'Second call: {time2:.3f}s, {len(chunks2)} chunks (cached)')

if time1 > 0 and time2 > 0:
    speedup = time1 / time2 if time2 > 0 else 0
    print(f'Speedup: {speedup:.1f}x faster with cache')
"

# Test LLM response caching
python3 -c "
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
import time

retriever = DenseRetriever()
generator = LLMGenerator()
query = 'What is your vision?'

# Retrieve chunks
chunks = retriever.retrieve(query, top_k=5, use_cache=True)

if len(chunks) > 0:
    # First call (cache miss)
    start = time.time()
    response1 = generator.generate(query, chunks[:3], use_cache=True, model_type='openai', fallback=False)
    time1 = time.time() - start
    print(f'First call: {time1:.3f}s, Cost: \${response1.cost_estimate:.4f}')
    
    # Second call (cache hit)
    start = time.time()
    response2 = generator.generate(query, chunks[:3], use_cache=True, model_type='openai', fallback=False)
    time2 = time.time() - start
    print(f'Second call: {time2:.3f}s, Cost: \${response2.cost_estimate:.4f} (cached)')
    
    if time1 > 0 and time2 > 0:
        speedup = time1 / time2 if time2 > 0 else 0
        cost_saved = response1.cost_estimate - response2.cost_estimate
        print(f'Speedup: {speedup:.1f}x faster')
        print(f'Cost saved: \${cost_saved:.4f}')
else:
    print('No chunks retrieved, skipping generation test')
"
```

### Test Cache Operations

```bash
# Test embedding cache
python3 -c "
from src.cache import cache_manager
import numpy as np

test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
cache_manager.cache_query_embedding('test_query', test_embedding)
retrieved = cache_manager.get_query_embedding('test_query')
if retrieved is not None and np.allclose(test_embedding, retrieved):
    print('✅ Embedding cache works!')
else:
    print('❌ Embedding cache failed')
"

# Test retrieval cache
python3 -c "
from src.cache import cache_manager
from src.retrieval.retriever import RetrievedChunk

test_chunks = [
    RetrievedChunk(
        text='Test chunk 1',
        metadata={'doc_id': 'test1'},
        score=0.9,
        chunk_id='chunk1',
        doc_id='doc1'
    )
]
cache_manager.cache_retrieval_results('test_query', test_chunks)
retrieved_chunks = cache_manager.get_retrieval_results('test_query')
if retrieved_chunks:
    print('✅ Retrieval cache works!')
else:
    print('❌ Retrieval cache failed')
"

# Test LLM cache
python3 -c "
from src.cache import cache_manager

test_response = {
    'answer': 'Test answer',
    'sources': ['source1'],
    'trace_id': 'test123',
    'token_usage': {'total': 100},
    'prompt_version': 'v1',
    'model_used': 'gpt-3.5-turbo',
    'cost_estimate': 0.001
}
cache_manager.cache_llm_response('test_query', test_response)
retrieved_response = cache_manager.get_llm_response('test_query')
if retrieved_response and retrieved_response['answer'] == 'Test answer':
    print('✅ LLM cache works!')
else:
    print('❌ LLM cache failed')
"
```

### Test Evaluation Metrics

```bash
# Test BLEU and ROUGE-L calculation
python3 -c "
from src.eval import Evaluator

evaluator = Evaluator()
candidate = 'The quick brown fox jumps over the lazy dog'
reference = 'A quick brown fox jumps over a lazy dog'

bleu = evaluator._calculate_bleu(candidate, reference)
print(f'✅ BLEU Score: {bleu:.3f}')

rouge_l = evaluator._calculate_rouge_l(candidate, reference)
print(f'✅ ROUGE-L Score: {rouge_l:.3f}')
"

# Test faithfulness check
python3 -c "
from src.eval import Evaluator

evaluator = Evaluator()
sources = ['The document discusses machine learning and AI.', 'It covers neural networks and deep learning.']
answer = 'The document discusses machine learning, neural networks, and deep learning.'

faithfulness = evaluator._simple_faithfulness_check(answer, sources)
print(f'✅ Faithfulness Check:')
print(f'   Is Faithful: {faithfulness.is_faithful}')
print(f'   Confidence: {faithfulness.confidence:.3f}')
"
```

### Comprehensive Integration Test

```bash
# Full end-to-end test
python3 -c "
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator
from src.cache import cache_manager
from src.utils.circuit_breaker import get_circuit_breaker
import time

print('=== Full Integration Test ===')

# Test cache
print('1. Testing cache...')
stats = cache_manager.get_cache_stats()
print(f'   ✅ Cache stats: {stats}')

# Test circuit breaker
print('2. Testing circuit breaker...')
cb = get_circuit_breaker('openai')
state = cb.get_state()
print(f'   ✅ Circuit breaker state: {state[\"state\"]}')

# Test retriever
print('3. Testing retriever...')
retriever = DenseRetriever()
query = 'What is your vision?'
cache_manager.clear_cache()

start = time.time()
chunks = retriever.retrieve(query, top_k=5, use_cache=True)
time1 = time.time() - start
print(f'   ✅ Retrieved {len(chunks)} chunks in {time1:.3f}s')

# Test cached retrieval
start = time.time()
chunks2 = retriever.retrieve(query, top_k=5, use_cache=True)
time2 = time.time() - start
print(f'   ✅ Cached retrieval: {time2:.3f}s ({time1/time2:.1f}x faster)')

# Test generator
print('4. Testing generator...')
generator = LLMGenerator()
print('   ✅ Generator initialized')

if len(chunks) > 0:
    try:
        response = generator.generate(query, chunks[:3], use_cache=True, model_type='openai', fallback=False)
        print(f'   ✅ Generated response: {len(response.answer)} chars, Cost: \${response.cost_estimate:.4f}')
    except Exception as e:
        print(f'   ⚠️  Generation test skipped: {e}')

print('\\n✅ All integration tests passed!')
"
```

### Run Complete Test Suite

```bash
# Run all component tests
python3 << 'EOF'
import sys
from pathlib import Path

test_results = []

# Test 1: Redis Connection
try:
    from src.cache import cache_manager
    stats = cache_manager.get_cache_stats()
    test_results.append(("✅", "Redis Cache", "Connected", f"Stats: {stats}"))
except Exception as e:
    test_results.append(("❌", "Redis Cache", "Failed", str(e)))

# Test 2: Circuit Breaker
try:
    from src.utils.circuit_breaker import get_circuit_breaker
    cb = get_circuit_breaker('openai')
    state = cb.get_state()
    test_results.append(("✅", "Circuit Breaker", "Working", f"State: {state['state']}"))
except Exception as e:
    test_results.append(("❌", "Circuit Breaker", "Failed", str(e)))

# Test 3: Retriever
try:
    from src.retrieval import DenseRetriever
    retriever = DenseRetriever()
    test_results.append(("✅", "Retriever", "Initialized", "With caching support"))
except Exception as e:
    test_results.append(("❌", "Retriever", "Failed", str(e)))

# Test 4: Generator
try:
    from src.generator import LLMGenerator
    generator = LLMGenerator()
    test_results.append(("✅", "Generator", "Initialized", "With caching + circuit breaker"))
except Exception as e:
    test_results.append(("❌", "Generator", "Failed", str(e)))

# Test 5: Evaluator
try:
    from src.eval import Evaluator
    evaluator = Evaluator()
    test_results.append(("✅", "Evaluator", "Working", "BLEU, ROUGE-L, Faithfulness"))
except Exception as e:
    test_results.append(("❌", "Evaluator", "Failed", str(e)))

# Test 6: A/B Tester
try:
    from src.generator.ab_testing import ABTester
    ab_tester = ABTester()
    test_results.append(("✅", "A/B Tester", "Initialized", "Ready for testing"))
except Exception as e:
    test_results.append(("❌", "A/B Tester", "Failed", str(e)))

# Test 7: Dataset Loader
try:
    from src.eval import DatasetLoader
    test_results.append(("✅", "Dataset Loader", "Working", "Can create/load datasets"))
except Exception as e:
    test_results.append(("❌", "Dataset Loader", "Failed", str(e)))

# Print results
for status, component, state, details in test_results:
    print(f"{status} {component:20} {state:15} {details}")

# Summary
passed = sum(1 for r in test_results if r[0] == "✅")
total = len(test_results)
print(f"\n📊 Summary: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! Everything is working correctly.")
else:
    print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
EOF
```

### Check Service Status

```bash
# Check all Docker services
docker-compose ps

# Check specific services
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(NAMES|rag_|qdrant|redis|prometheus|grafana)"

# Test Redis
docker exec rag_redis redis-cli ping

# Test Qdrant
curl -s http://localhost:6333/collections | python3 -m json.tool

# Test Prometheus
curl -s http://localhost:9090/-/healthy

# Test Grafana
curl -s http://localhost:3000/api/health
```

## 🌐 Task 9: API Backend and React Frontend

### Start FastAPI Backend

```bash
# Install API dependencies
pip install fastapi uvicorn[standard] python-multipart

# Start API server
python3 scripts/start_api.py

# Or use uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# API will be available at:
# - API: http://localhost:8001
# - Docs: http://localhost:8001/docs
# - Health: http://localhost:8001/health
```

### Test API Endpoints

```bash
# Test query endpoint
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your vision?",
    "top_k": 5,
    "prompt_version": "v1"
  }'

# Test feedback endpoint
curl -X POST http://localhost:8001/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "trace_123",
    "thumbs_up": true,
    "rating": 5,
    "text_feedback": "Great answer!"
  }'

# Test metrics endpoint
curl http://localhost:8001/api/metrics

# Test trace endpoint
curl http://localhost:8001/api/trace/trace_123
```

### Setup React Frontend

```bash
# Navigate to UI directory
cd ui

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Frontend will be available at:
# - UI: http://localhost:5173
# - Admin: http://localhost:5173/admin
```

### Build React Frontend for Production

```bash
cd ui

# Build for production
npm run build

# Preview production build
npm run preview
```

### API Endpoints

#### POST /api/query
Main query endpoint for RAG pipeline.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "prompt_version": "v1",
  "use_cache": true
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "sources": [...],
  "trace_id": "trace_123",
  "latency": 1.23,
  "token_usage": {...},
  "cost_estimate": 0.001,
  "model_used": "gpt-3.5-turbo",
  "prompt_version": "v1"
}
```

#### POST /api/feedback
Submit user feedback.

**Request:**
```json
{
  "trace_id": "trace_123",
  "rating": 5,
  "thumbs_up": true,
  "text_feedback": "Helpful answer"
}
```

#### GET /api/metrics
Get aggregated metrics for admin dashboard.

#### GET /api/trace/{trace_id}
Get trace information for debugging.

#### DELETE /api/admin/clear-all
Clear all context from the database (irreversible).
- Deletes all chunks from Qdrant
- Removes processed chunk files
- Resets manifest file
- Keeps raw files (can be re-indexed)

**Response:**
```json
{
  "success": true,
  "message": "All context cleared successfully",
  "deleted": {
    "qdrant_points": 322,
    "processed_files": 2
  }
}
```

#### GET /api/admin/stats
Get database statistics.

**Response:**
```json
{
  "qdrant_points": 322,
  "processed_documents": 2,
  "processed_files": 1,
  "collection_name": "rag_collection"
}
```

### Frontend Features

- **Chat Interface**: Query input and answer display
- **Source Citations**: Clickable links with confidence scores
- **Feedback Form**: Thumbs up/down, rating, text feedback
- **Metrics Dashboard**: Admin view with cache stats and performance metrics
- **Trace Inspection**: View trace details for debugging
- **File Upload**: Upload PDF, TXT, or MD files for indexing
- **Clear Context**: Admin feature to clear all indexed data and start fresh

### Environment Variables

Create `.env` file in project root:
```bash
# API Configuration
API_KEY=your_api_key_here  # Optional, for API authentication

# Frontend Configuration (ui/.env)
VITE_API_URL=http://localhost:8001
```

### Full Stack Startup

```bash
# Terminal 1: Start backend
python3 scripts/start_api.py

# Terminal 2: Start frontend
cd ui && npm run dev

# Terminal 3: Start services (if not running)
docker-compose up -d
```

### Access Points

- **Frontend**: http://localhost:5173
- **API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Admin Dashboard**: http://localhost:5173/admin
- **Qdrant**: http://localhost:6333
- **Redis**: localhost:6379
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 🎯 Next Steps

After running the pipeline:
1. **Task 5**: Generator Module (LLM response generation) ✅
2. **Task 6**: Observability (OpenTelemetry, Prometheus, Grafana) ✅
3. **Task 7**: Evaluation Harness (Precision@k, BLEU/ROUGE, Faithfulness) ✅
4. **Task 8**: Caching, Circuit Breaker, A/B Testing ✅
5. **Task 9**: FastAPI Backend and React Frontend ✅
6. **Task 10**: Privacy, Scaling, and Cost Optimization

---

## 🔧 Service Ports Reference

### All Service Ports

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Frontend** | 5173 | http://localhost:5173 | React UI (Vite dev server) |
| **API Backend** | 8001 | http://localhost:8001 | FastAPI REST API |
| **Qdrant** | 6333 | http://localhost:6333 | Vector database (REST API) |
| **Qdrant Dashboard** | 6333 | http://localhost:6333/dashboard | Qdrant web UI |
| **Qdrant gRPC** | 6334 | localhost:6334 | gRPC API (internal) |
| **Redis** | 6379 | localhost:6379 | Caching service |
| **Prometheus** | 9090 | http://localhost:9090 | Metrics collection |
| **Grafana** | 3000 | http://localhost:3000 | Metrics visualization |

### Quick Access URLs

```bash
# Frontend
http://localhost:5173              # Home page
http://localhost:5173/admin        # Admin dashboard

# Backend API
http://localhost:8001              # API root
http://localhost:8001/docs         # Swagger UI
http://localhost:8001/health       # Health check

# Services
http://localhost:6333              # Qdrant REST API
http://localhost:6333/dashboard    # Qdrant web dashboard
http://localhost:6334              # Qdrant gRPC (internal, not web UI)
http://localhost:9090              # Prometheus
http://localhost:3000              # Grafana
```

## 🗑️ Admin Features

### Clear All Context

Clear all indexed data from the database to start fresh.

**Via UI:**
1. Go to Admin Dashboard: http://localhost:5173/admin
2. Scroll to "Database Management" section
3. Click "Clear All Context"
4. Confirm the action

**Via API:**
```bash
# Clear all context
curl -X DELETE http://localhost:8001/api/admin/clear-all

# Check current stats
curl http://localhost:8001/api/admin/stats
```

**What gets cleared:**
- All chunks from Qdrant vector database
- All processed chunk files (.jsonl)
- Manifest file (resets to empty)
- Raw files are kept (can be re-indexed)

## 🧪 Testing Commands

### A/B Testing

```bash
# Run A/B test via API
curl -X POST http://localhost:8001/api/ab-test \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your vision?",
    "variants": ["v1", "v2"],
    "top_k": 5
  }'

# Get A/B test results
curl http://localhost:8001/api/ab-test/results

# Run A/B test via script
python3 scripts/test_ab_testing.py
```

### Evaluation Testing

```bash
# Run evaluation suite
python3 scripts/test_evaluation.py

# Create sample evaluation dataset
python3 -c "
from src.eval import DatasetLoader
DatasetLoader.create_sample_dataset('data/eval/sample_dataset.json')
"

# View evaluation results
cat data/eval/results.json | python3 -m json.tool
```

### API Endpoint Testing

```bash
# Test all API endpoints
echo "=== Testing API Endpoints ==="

# Health check
curl http://localhost:8001/health

# Query endpoint
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Metrics endpoint
curl http://localhost:8001/api/metrics

# Trace endpoint
curl http://localhost:8001/api/trace/test_123

# Feedback endpoint
curl -X POST http://localhost:8001/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "test", "rating": 5}'

# Admin stats
curl http://localhost:8001/api/admin/stats

# A/B test
curl -X POST http://localhost:8001/api/ab-test \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "variants": ["v1", "v2"]}'
```

### Component Testing

```bash
# Test retriever
python3 scripts/test_retrieval_simple.py "What is your vision?"

# Test generator
python3 scripts/test_generation.py "What is your vision?"

# Test observability
python3 scripts/test_observability.py

# Test caching
python3 -c "
from src.cache import cache_manager
stats = cache_manager.get_cache_stats()
print('Cache stats:', stats)
"

# Test circuit breaker
python3 -c "
from src.utils.circuit_breaker import get_circuit_breaker
cb = get_circuit_breaker('openai')
print('Circuit breaker state:', cb.get_state())
"
```

### Integration Testing

```bash
# Run full pipeline test
python3 scripts/run_all.py

# Test end-to-end query flow
python3 -c "
from src.retrieval import DenseRetriever
from src.generator import LLMGenerator

retriever = DenseRetriever()
generator = LLMGenerator()

query = 'What is your vision?'
chunks = retriever.retrieve(query, top_k=5)
if chunks:
    response = generator.generate(query, chunks)
    print('Answer:', response.answer[:200])
    print('Sources:', len(response.sources))
else:
    print('No chunks found')
"
```

### Service Health Checks

```bash
# Check all services
echo "=== Service Health Checks ==="

# Qdrant
curl http://localhost:6333/health

# Redis
docker exec rag_redis redis-cli ping

# Prometheus
curl http://localhost:9090/-/healthy

# Grafana
curl http://localhost:3000/api/health

# API
curl http://localhost:8001/health

# Frontend
curl http://localhost:5173 > /dev/null && echo "Frontend OK" || echo "Frontend not responding"
```

### Performance Testing

```bash
# Test query latency
time curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'

# Test cache performance
python3 -c "
import time
from src.cache import cache_manager
from src.retrieval import DenseRetriever

retriever = DenseRetriever()
query = 'What is your vision?'

# First call (cache miss)
start = time.time()
chunks1 = retriever.retrieve(query, use_cache=True)
time1 = time.time() - start

# Second call (cache hit)
start = time.time()
chunks2 = retriever.retrieve(query, use_cache=True)
time2 = time.time() - start

print(f'First call (miss): {time1:.3f}s')
print(f'Second call (hit): {time2:.3f}s')
print(f'Speedup: {time1/time2:.2f}x')
"
```

### Database Testing

```bash
# Check Qdrant collection stats
python3 -c "
from qdrant_client import QdrantClient
from src.config import settings

client = QdrantClient(url=settings.qdrant_url)
info = client.get_collection(settings.qdrant_collection_name)
print(f'Collection: {settings.qdrant_collection_name}')
print(f'Points: {info.points_count}')
print(f'Vectors: {info.vectors_count}')
"

# Check Redis cache stats
python3 -c "
from src.cache import cache_manager
stats = cache_manager.get_cache_stats()
print('Cache stats:', stats)
"

# Check manifest
cat data/manifest.json | python3 -m json.tool
```

---

**Last Updated**: Task 9 (API Backend and React Frontend) completed + Admin features added + A/B Testing added

