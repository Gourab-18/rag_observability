#!/usr/bin/env python3
# test script for observability (tracing and metrics)

import sys
import time
import logging
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.observability import (
    tracing_manager,
    metrics_manager,
    create_trace,
    get_current_trace_id,
    instrument_with_observability,
    record_token_usage,
    record_cost,
    record_precision_at_k,
    log_with_trace
)

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@instrument_with_observability("test_operation", {"component": "test"})
def test_operation(duration: float = 0.1):
    # simulated operation for testing
    time.sleep(duration)
    return {"result": "success"}


def main():
    print("=" * 60)
    print("Testing Observability System")
    print("=" * 60)
    
    # 1. Test tracing
    print("\n1. Testing Tracing...")
    query_id = "test_query_123"
    trace_id = create_trace(query_id)
    print(f"   Created trace_id: {trace_id}")
    print(f"   Current trace_id: {get_current_trace_id()}")
    
    # 2. Test metrics
    print("\n2. Testing Metrics...")
    metrics_manager.record_latency("test", 0.5)
    metrics_manager.record_tokens("openai", "prompt", 100)
    metrics_manager.record_tokens("openai", "completion", 50)
    metrics_manager.record_cost("openai", "gpt-3.5-turbo", 0.003)
    metrics_manager.record_precision_at_k(1, 0.85)
    metrics_manager.record_precision_at_k(3, 0.75)
    metrics_manager.record_precision_at_k(5, 0.70)
    print("   Recorded test metrics")
    
    # 3. Test instrumentation decorator
    print("\n3. Testing Instrumentation Decorator...")
    result = test_operation(0.2)
    print(f"   Operation result: {result}")
    
    # 4. Test structured logging
    print("\n4. Testing Structured Logging...")
    log_with_trace(logger, logging.INFO, "Test log message", 
                   query_id=query_id, stage="test")
    
    # 5. Start metrics server (if available)
    print("\n5. Starting Metrics Server...")
    try:
        metrics_manager.start_metrics_server()
        print(f"   Metrics server started on port 8000")
        print(f"   Access metrics at: http://localhost:8000/metrics")
    except Exception as e:
        print(f"   Could not start metrics server: {e}")
    
    print("\n" + "=" * 60)
    print("Observability Test Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start Prometheus: docker-compose up -d prometheus")
    print("2. Start Grafana: docker-compose up -d grafana")
    print("3. Access Grafana: http://localhost:3000 (admin/admin)")
    print("4. Access Prometheus: http://localhost:9090")
    print("5. View metrics: http://localhost:8000/metrics")


if __name__ == "__main__":
    main()

