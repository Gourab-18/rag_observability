# RAG System Specification

## Problem Statement

Build a production-ready RAG (Retrieval Augmented Generation) system that enables users to query a knowledge base using natural language questions. The system should retrieve relevant information from a corpus of documents and generate accurate, source-cited answers.

## User Stories

### US-1: Search Knowledge Base
**As a user**, I want to search the knowledge base by asking natural language questions, so that I can quickly find relevant information.

**Acceptance Criteria**:
- User can submit queries via API or UI
- System returns answers within 2 seconds (p95 latency)
- Answers are relevant to the query

### US-2: View Sources
**As a user**, I want to see the sources used to generate answers, so that I can verify the information and access original documents.

**Acceptance Criteria**:
- Each answer includes source citations
- Sources are clickable links to original documents
- Confidence scores are displayed per source

### US-3: Provide Feedback
**As a user**, I want to provide feedback on answer quality, so that the system can improve over time.

**Acceptance Criteria**:
- Feedback mechanism (thumbs up/down) is available
- Optional text feedback can be submitted
- Feedback is correlated with trace_id for analysis

## Metrics Definitions

### Retrieval Metrics

#### Precision@k (k=1, 3, 5)
**Definition**: Fraction of retrieved items in top-k results that are relevant to the query.

**Formula**: `Precision@k = (Number of relevant items in top-k) / k`

**Targets**:
- Precision@1: > 0.7 (70% of top result should be relevant)
- Precision@3: > 0.6 (60% of top 3 results should be relevant)
- Precision@5: > 0.5 (50% of top 5 results should be relevant)

**Evaluation**: Manual relevance labeling of retrieved chunks against query intent.

#### Recall
**Definition**: Fraction of all relevant items that are successfully retrieved.

**Formula**: `Recall = (Number of relevant items retrieved) / (Total number of relevant items)`

**Target**: Recall > 0.8 (80% of relevant items should be in retrieved set)

### Generation Metrics

#### Hallucination Rate
**Definition**: Fraction of generated answers that contain unsupported claims not present in the retrieved sources.

**Target**: < 5% (fewer than 5% of answers should hallucinate)

**Evaluation Method**: LLM-based factuality checking (LLM judge) comparing claims in answer to source chunks.

#### BLEU Score
**Definition**: N-gram precision between generated answer and reference answer.

**Target**: BLEU > 0.5

#### ROUGE-L Score
**Definition**: Longest common subsequence (LCS) based F-measure.

**Target**: ROUGE-L > 0.6

### Performance Metrics

#### Latency (P95)
**Definition**: 95th percentile response time from query submission to answer delivery.

**Target**: P95 latency < 2 seconds

**Measurement**: Track end-to-end request latency from API entry to response.

#### Latency (P99)
**Definition**: 99th percentile response time.

**Target**: P99 latency < 3 seconds

### Efficiency Metrics

#### Cache Hit Rate
**Definition**: Fraction of requests served from cache (embeddings, retrieval, or LLM response cache).

**Formula**: `Cache Hit Rate = (Cached requests) / (Total requests)`

**Target**: > 70% for repeated queries

#### Token Reduction
**Definition**: Percentage reduction in tokens sent to LLM via compression strategies.

**Formula**: `Token Reduction = (Original tokens - Compressed tokens) / Original tokens`

**Target**: ≥ 30% reduction with no degradation in Precision@1

## Dataset Requirements

### Document Collection
- **Size**: 50-100 sample documents
- **Formats**: Mix of PDFs and text files
- **Location**: `data/raw/`
- **Content**: Diverse topics suitable for knowledge base Q&A

### Data Manifest
Each document should have metadata tracked in `data/manifest.json`:
```json
{
  "doc_id": "unique_identifier",
  "file_name": "document.pdf",
  "file_path": "data/raw/document.pdf",
  "file_type": "pdf",
  "file_size": 12345,
  "upload_date": "2024-01-01T00:00:00Z",
  "title": "Document Title",
  "author": "Author Name",
  "tags": ["tag1", "tag2"],
  "status": "processed"
}
```

## Acceptance Criteria

1. ✅ **End-to-end query flow works**: query → retrieval → generation → response
2. ✅ **Observability**: all spans visible in Grafana, metrics collected
3. ✅ **Evaluation suite passes**: precision@1 >0.7, hallucination <5%
4. ✅ **Cache hit rate >70%** for repeated queries
5. ✅ **Load test**: 100 RPS with p95 latency <2s
6. ✅ **Frontend provides source citations** and feedback collection
7. ✅ **Cost tracking and optimization** in place

## Configuration Requirements

The system must support configuration via environment variables for:
- Vector database connection (Qdrant)
- LLM API keys and endpoints (OpenAI, Ollama)
- Redis connection for caching
- Chunking parameters (chunk_size, overlap)
- Retrieval parameters (top_k, similarity_threshold)
- Observability endpoints (Prometheus, OTLP)

