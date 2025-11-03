# RAG System with Observability

A production-ready RAG (Retrieval Augmented Generation) system with full observability, evaluation, and user feedback capabilities.

## Features

- 📄 **Document Ingestion**: PDF and text file processing with token-aware chunking
- 🔍 **Vector Search**: Dense retrieval with Qdrant, optional reranking and compression
- 🤖 **LLM Generation**: OpenAI API with local fallback (Ollama)
- 📊 **Observability**: OpenTelemetry tracing, Prometheus metrics, Grafana dashboards
- 🧪 **Evaluation**: Automated evaluation suite with precision@k, hallucination detection
- 💾 **Caching**: Redis caching for embeddings, retrieval, and LLM responses
- 🛡️ **Resilience**: Circuit breaker pattern, automatic fallbacks
- 🎨 **UI**: React frontend with chat interface and source citations (coming in Task 9)
- 🔒 **Privacy**: PII redaction, authentication, rate limiting

## Project Status

**Current Task**: Task 1 ✅ Complete - Project Spec and Metrics Setup

## Project Structure

```
rag_observability/
├── src/
│   ├── config/          # Configuration management ✅
│   ├── ingest/           # Document ingestion pipeline (Task 2)
│   ├── retrieval/       # Vector search & compression (Task 4)
│   ├── generator/        # LLM generation (Task 5)
│   ├── eval/            # Evaluation framework (Task 7)
│   ├── observability/   # OpenTelemetry, metrics (Task 6)
│   ├── cache/           # Redis caching (Task 8)
│   └── utils/           # Shared utilities
├── api/                 # FastAPI application (Task 9)
├── ui/                  # React frontend (Task 9)
├── tests/               # Test suite
├── docker/              # Docker configs
├── data/
│   ├── raw/             # Raw documents (PDFs, text files)
│   └── processed/      # Processed chunks
├── prompts/             # Versioned prompt templates
├── dashboards/          # Grafana dashboards
└── docs/                # Documentation
```

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for detailed setup instructions.

### Prerequisites

- Python 3.9+
- Docker Desktop (for services)
- OpenAI API key

### Setup

1. **Clone and install**:
```bash
git clone <your-repo-url>
cd rag_observability
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp env.template .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start services** (Task 3+):
```bash
docker-compose up -d
```

## Implementation Plan

Following a 10-task implementation plan:
- ✅ Task 1: Project Spec and Metrics Setup
- ⏳ Task 2: Chunking Pipeline
- ⏳ Task 3: Embeddings and Vector Index
- ⏳ Task 4: Retriever and Compressor
- ⏳ Task 5: Generator and Prompt Versioning
- ⏳ Task 6: Observability and Tracing
- ⏳ Task 7: Evaluation Harness and CI
- ⏳ Task 8: Resilience, Caching, and A/B Testing
- ⏳ Task 9: Provenance UI and Feedback Loop
- ⏳ Task 10: Privacy, Scaling, and Cost Optimization

See [plan.md](plan.md) for detailed task breakdown.

## Documentation

- [Project Specification](docs/spec.md) - Problem statement, metrics, acceptance criteria
- [Setup Guide](docs/SETUP.md) - Local setup instructions
- [Implementation Plan](plan.md) - Detailed task breakdown

## License

MIT

