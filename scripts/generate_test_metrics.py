#!/usr/bin/env python3
"""Generate test metrics for Grafana dashboard"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.observability.metrics import metrics_manager
from src.observability.instrumentation import record_token_usage, record_cost, record_precision_at_k

print("Generating test metrics...")

# Record some latency metrics
metrics_manager.record_latency("ingestion", 0.5)
metrics_manager.record_latency("embedding", 1.2)
metrics_manager.record_latency("retrieval", 0.3)
metrics_manager.record_latency("generation", 2.1)

# Record token usage
record_token_usage("openai", "prompt", 2500)
record_token_usage("openai", "completion", 150)
record_token_usage("openai", "prompt", 2400)
record_token_usage("openai", "completion", 200)

# Record costs
record_cost("openai", "gpt-3.5-turbo", 0.0040)
record_cost("openai", "gpt-3.5-turbo", 0.0041)

# Record precision at k
record_precision_at_k(1, 0.85)
record_precision_at_k(3, 0.75)
record_precision_at_k(5, 0.70)

# Record document count
metrics_manager.record_metric("rag_documents_total", 322.0)

print("✅ Test metrics generated!")
print("Check Prometheus: http://localhost:9090")
print("Check Grafana: http://localhost:3000")

