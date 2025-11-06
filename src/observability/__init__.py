# observability module exports

from src.observability.tracing import (
    tracing_manager,
    create_trace,
    get_current_trace_id
)

from src.observability.metrics import (
    metrics_manager,
    record_metric
)

from src.observability.instrumentation import (
    instrument_span,
    instrument_latency,
    instrument_with_observability,
    ObservabilityContext,
    record_token_usage,
    record_cost,
    record_precision_at_k,
    record_cache_hit_rate,
    log_with_trace
)

__all__ = [
    # Tracing
    'tracing_manager',
    'create_trace',
    'get_current_trace_id',
    # Metrics
    'metrics_manager',
    'record_metric',
    # Instrumentation
    'instrument_span',
    'instrument_latency',
    'instrument_with_observability',
    'ObservabilityContext',
    'record_token_usage',
    'record_cost',
    'record_precision_at_k',
    'record_cache_hit_rate',
    'log_with_trace'
]

