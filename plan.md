# RAG System with Observability - Implementation Plan

## Project Structure

```
rag_pipeline/
├── src/
│   ├── ingest/              # Document ingestion pipeline
│   ├── retrieval/           # Vector search & compression
│   ├── generator/           # LLM generation & prompt management
│   ├── eval/                # Evaluation framework
│   ├── observability/       # OpenTelemetry, metrics, tracing
│   ├── cache/               # Redis caching layer
│   ├── utils/               # Shared utilities
│   └── config/              # Configuration management
├── api/                     # FastAPI application
├── ui/                      # React + Tailwind frontend
├── tests/                   # Test suite
├── docker/                  # Docker configs (Qdrant, Redis, Grafana)
├── data/
│   ├── raw/                 # Raw documents (PDFs, text files)
│   └── processed/           # Processed chunks
├── prompts/                 # Versioned prompt templates
├── dashboards/              # Grafana dashboard JSON
├── docs/                    # Documentation
└── scripts/                 # Setup & utility scripts
```

## Task Breakdown

### Task 1: Project Spec and Metrics Setup

**Files**: `docs/spec.md`, `data/raw/`, `src/config/settings.py`

- Define problem statement (knowledge base Q&A system)
- User stories: search, ask questions, view sources
- Metrics: precision@k (k=1,3,5), hallucination rate (target <5%), latency p95 <2s, retrieval recall
- Dataset: 50-100 sample documents (mix of PDFs and text files)
- Create data manifest with document metadata
- Configuration system with environment variables

**Key Deliverables**:

- `docs/spec.md` with metrics definitions and acceptance criteria
- Sample documents in `data/raw/`
- `src/config/settings.py` for centralized config management

---

### Task 2: Chunking Pipeline

**Files**: `src/ingest/chunker.py`, `src/ingest/document_loader.py`, `tests/test_chunker.py`

- Document loader supporting PDF (PyPDF2/pdfplumber) and text files
- Token-aware chunking using tiktoken (OpenAI) or transformers tokenizer
- Configurable chunk_size (default 500 tokens) and overlap (default 50 tokens)
- Output JSONL format: `{doc_id, chunk_id, text, token_count, metadata, overlap_info}`
- Chunk validation: no chunk exceeds token limit, overlap continuity verified

**Key Functions**:

- `chunk_document(text, chunk_size, overlap)` → List[Chunk]
- `chunk_file(file_path)` → List[Chunk]
- `save_chunks(chunks, output_path)` → JSONL file

---

### Task 3: Embeddings and Vector Index

**Files**: `src/ingest/embed_and_upsert.py`, `src/utils/embeddings.py`, `docker-compose.yml`

- Embedding generation: OpenAI `text-embedding-3-small` (primary) + local fallback (sentence-transformers)
- Qdrant setup: Docker container, collection creation with proper schema
- Metadata schema: `{doc_id, chunk_id, source, chunk_pos, created_at}`
- Batch embedding and upsert operations
- Vector index validation: query returns relevant chunks

**Key Functions**:

- `generate_embeddings(texts, model_type='openai')` → np.array
- `upsert_to_qdrant(vectors, metadata, collection_name)`
- `create_collection(collection_name, vector_size)`
- Connection management with retry logic

**Docker Setup**:

- `docker-compose.yml` with Qdrant service
- Health check endpoints
- Volume persistence for data

---

### Task 4: Retriever and Compressor Module

**Files**: `src/retrieval/retriever.py`, `src/retrieval/compressor.py`, `src/retrieval/reranker.py`

- Dense vector retrieval: query embedding → Qdrant similarity search (top-k, default k=10)
- Optional reranking with cross-encoder (sentence-transformers)
- Compression strategies:
  - Score filtering (cosine similarity threshold, default 0.7)
  - BM25 filtering (hybrid retrieval)
  - LLM-based compression (optional, LangChain compressor)
- Token reduction target: ≥30% with no precision@1 degradation

**Key Functions**:

- `retrieve(query, top_k=10, use_rerank=True)` → List[RetrievedChunk]
- `compress_chunks(chunks, strategy='score_filter')` → List[Chunk]
- `hybrid_search(query, dense_k=10, sparse_k=5)` → List[Chunk]
- Integration with LangChain's retrieval chain

---

### Task 5: Generator and Prompt Versioning

**Files**: `src/generator/generate.py`, `src/generator/prompt_manager.py`, `prompts/v1/`, `prompts/v2/`

- LLM integration: OpenAI (primary) with automatic fallback to Ollama/local
- Prompt template system: versioned prompts in `prompts/` directory
- Context injection: retrieved chunks + query → formatted prompt
- Response parsing: `{answer, sources, trace_id, token_usage, prompt_version, model_used}`
- Cost tracking: token counts and estimated costs per provider

**Key Functions**:

- `generate(query, context_chunks, prompt_version='v1')` → Response
- `load_prompt_template(version)` → PromptTemplate
- `call_llm(prompt, model_type='openai', fallback=True)` → LLMResponse
- Local LLM setup via Ollama or transformers

**Prompt Structure**:

```
prompts/
├── v1/
│   └── rag_template.txt
├── v2/
│   └── rag_template.txt
└── versions.yaml  # Version metadata
```

---

### Task 6: Observability and Tracing

**Files**: `src/observability/tracing.py`, `src/observability/metrics.py`, `src/observability/instrumentation.py`, `dashboards/grafana/`

- OpenTelemetry instrumentation:
  - Spans for: document ingestion, embedding generation, retrieval, compression, LLM call
  - Custom attributes: query_id, doc_count, token_usage, model_used
- Prometheus metrics:
  - `rag_request_latency_seconds` (histogram)
  - `rag_token_usage_total` (counter, by provider)
  - `rag_cost_per_query` (gauge, USD)
  - `rag_cache_hit_rate` (gauge)
  - `rag_precision_at_k` (histogram, k=1,3,5)
- Grafana dashboard:
  - Latency trends, token usage by provider, cost tracking
  - Retrieval quality metrics, error rates
- Structured logging: JSON logs with trace_id correlation

**Key Components**:

- `instrument_langchain()` - Auto-instrument LangChain calls
- `create_trace(query_id)` - Manual span creation
- `record_metric(name, value, labels)` - Prometheus recording
- `dashboards/grafana/rag_dashboard.json` - Pre-configured dashboard

**Docker Setup**:

- Prometheus + Grafana services in docker-compose
- OTLP exporter configuration

---

### Task 7: Evaluation Harness and CI

**Files**: `src/eval/evaluator.py`, `src/eval/datasets.py`, `.github/workflows/test_eval.yml`

- Metrics computation:
  - Precision@k (k=1,3,5): manual relevance labels
  - BLEU/ROUGE: reference answer comparison
  - LLM-based factuality: use LLM judge to check faithfulness
  - Retrieval recall: relevant docs in retrieved set
- Evaluation dataset: Q&A pairs with ground truth answers and relevant doc IDs
- Automated test suite: run on commit, fail if hallucination rate > threshold
- Results logging: JSON output for tracking over time

**Key Functions**:

- `evaluate_retrieval(queries, ground_truth_docs)` → RetrievalMetrics
- `evaluate_generation(answers, references)` → GenerationMetrics
- `evaluate_faithfulness(answer, sources)` → FaithfulnessScore
- `run_evaluation_suite(dataset_path)` → EvaluationReport

**CI Integration**:

- GitHub Actions workflow runs nightly
- Stores results as artifacts
- Sends alerts on metric degradation

---

### Task 8: Resilience, Caching, and A/B Testing

**Files**: `src/cache/redis_cache.py`, `src/generator/ab_testing.py`, `src/utils/circuit_breaker.py`

- Redis caching:
  - Cache query embeddings (TTL: 24h)
  - Cache retrieval results (TTL: 1h)
  - Cache LLM responses for identical queries (TTL: 6h)
  - Target cache hit rate >70% for repeated queries
- Circuit breaker pattern:
  - Monitor LLM API failures
  - Auto-fallback to local model after threshold failures
  - Exponential backoff for retries
- A/B testing:
  - Prompt version comparison via config flag
  - Side-by-side logging: `{query_id, prompt_version_a, prompt_version_b, metrics}`
  - Analysis script to compare versions

**Key Components**:

- `cache_query_embedding(query)` → Optional[Embedding]
- `cache_llm_response(query, response)` → None
- `CircuitBreaker(provider, failure_threshold=5)` → Context manager
- `ab_test_prompt(query, variants=['v1', 'v2'])` → ComparisonResult

**Docker Setup**:

- Redis service in docker-compose
- Connection pooling and health checks

---

### Task 9: Provenance UI and Feedback Loop

**Files**: `api/routes/feedback.py`, `ui/src/components/`, `ui/src/pages/`

- React frontend:
  - Chat interface with query input
  - Answer display with source citations (clickable links)
  - Confidence scores per source
  - Feedback button (thumbs up/down, text feedback)
  - Metrics dashboard (admin view): latency, costs, quality metrics
- Backend endpoints:
  - `POST /api/query` - Main query endpoint
  - `POST /api/feedback` - User feedback submission
  - `GET /api/metrics` - Admin metrics endpoint
  - `GET /api/trace/{trace_id}` - Trace inspection
- Feedback storage:
  - Store feedback with trace_id for correlation
  - Use feedback to improve retrieval/generation

**UI Structure**:

```
ui/src/
├── components/
│   ├── ChatInterface.tsx
│   ├── SourceList.tsx
│   ├── FeedbackForm.tsx
│   └── MetricsDashboard.tsx
├── pages/
│   ├── Home.tsx
│   └── Admin.tsx
└── hooks/
    └── useRAGQuery.ts
```

**API Structure**:

- FastAPI with CORS for frontend
- WebSocket support for streaming responses (optional enhancement)
- Authentication middleware (basic API key or JWT)

---

### Task 10: Privacy, Scaling, and Cost Optimization

**Files**: `src/utils/pii_redaction.py`, `api/middleware/auth.py`, `api/middleware/rate_limit.py`, `scripts/load_test.py`

- PII redaction:
  - Detect and redact: emails, phone numbers, SSNs, credit cards
  - Apply before chunking and indexing
  - Audit logs: track redaction events
- Authentication & rate limiting:
  - API key authentication or JWT
  - Rate limiting: 100 requests/hour per user (configurable)
  - User quota tracking
- Scaling:
  - Batch embedding processing for bulk ingestion
  - Async LLM calls with concurrent request handling
  - Connection pooling for Qdrant/Redis
- Cost optimization:
  - Cost dashboard: cost per query by provider
  - Automatic routing: expensive queries → local model
  - Token budget alerts
- Load testing:
  - Script to test 100 RPS target
  - Monitor latency degradation
  - Auto-scaling recommendations

**Key Components**:

- `redact_pii(text)` → RedactedText
- `rate_limit_middleware(requests_per_hour=100)` → FastAPI middleware
- `batch_embeddings(texts, batch_size=100)` → List[Embedding]
- `cost_tracker.record_query(cost, provider)` → None

---

## Technical Stack Summary

- **Vector DB**: Qdrant (self-hosted via Docker)
- **LLM**: OpenAI API (primary) + Ollama/local fallback
- **Orchestration**: LangChain
- **Observability**: OpenTelemetry + Prometheus + Grafana
- **Caching**: Redis
- **Frontend**: React + Tailwind CSS
- **Backend**: FastAPI
- **Embeddings**: OpenAI `text-embedding-3-small` + sentence-transformers fallback

## Implementation Order

1. Tasks 1-3: Foundation (spec, chunking, embeddings)
2. Tasks 4-5: Core RAG (retrieval, generation)
3. Task 6: Observability (critical for production)
4. Task 7: Evaluation (quality assurance)
5. Tasks 8-9: Production polish (caching, UI, feedback)
6. Task 10: Hardening (privacy, scaling, costs)

## Success Criteria

- ✅ End-to-end query flow works: query → retrieval → generation → response
- ✅ Observability: all spans visible in Grafana, metrics collected
- ✅ Evaluation suite passes: precision@1 >0.7, hallucination <5%
- ✅ Cache hit rate >70% for repeated queries
- ✅ Load test: 100 RPS with p95 latency <2s
- ✅ Frontend provides source citations and feedback collection
- ✅ Cost tracking and optimization in place