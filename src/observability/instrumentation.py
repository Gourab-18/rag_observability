# instrumentation helpers for RAG pipeline components

from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import time
import logging

from src.observability.tracing import tracing_manager, create_trace, get_current_trace_id
from src.observability.metrics import metrics_manager

logger = logging.getLogger(__name__)


def instrument_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    # decorator to instrument a function with OpenTelemetry span
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # get trace_id from kwargs or current context
            trace_id = kwargs.get('trace_id') or get_current_trace_id()
            
            # create span attributes
            span_attrs = attributes.copy() if attributes else {}
            span_attrs['function'] = func.__name__
            
            with tracing_manager.create_span(name, attributes=span_attrs, trace_id=trace_id):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def instrument_latency(stage: str):
    # decorator to instrument a function with latency metrics
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_manager.record_latency(stage, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_manager.record_latency(stage, duration)
                metrics_manager.record_error(stage, type(e).__name__)
                raise
        
        return wrapper
    return decorator


def instrument_with_observability(stage: str, attributes: Optional[Dict[str, Any]] = None):
    # combined decorator for both tracing and metrics
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # get trace_id from kwargs or current context
            trace_id = kwargs.get('trace_id') or get_current_trace_id()
            
            # create span attributes
            span_attrs = attributes.copy() if attributes else {}
            span_attrs['function'] = func.__name__
            span_attrs['stage'] = stage
            
            start_time = time.time()
            
            # use ObservabilityContext instead of direct span creation
            with ObservabilityContext(stage, attributes=span_attrs, trace_id=trace_id):
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    metrics_manager.record_latency(stage, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    metrics_manager.record_latency(stage, duration)
                    metrics_manager.record_error(stage, type(e).__name__)
                    raise
        
        return wrapper
    return decorator


class ObservabilityContext:
    # context manager for observability in a code block
    
    def __init__(
        self,
        stage: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        self.stage = stage
        self.attributes = attributes or {}
        self.trace_id = trace_id or get_current_trace_id()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.span_context = tracing_manager.create_span(
            f"rag_{self.stage}",
            attributes=self.attributes,
            trace_id=self.trace_id
        )
        self.span = self.span_context.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        metrics_manager.record_latency(self.stage, duration)
        
        if exc_type:
            metrics_manager.record_error(self.stage, exc_type.__name__)
        
        self.span_context.__exit__(exc_type, exc_val, exc_tb)
        return False


def record_token_usage(provider: str, token_type: str, count: int):
    # convenience function to record token usage
    metrics_manager.record_tokens(provider, token_type, count)


def record_cost(provider: str, model: str, cost: float):
    # convenience function to record cost
    metrics_manager.record_cost(provider, model, cost)


def record_precision_at_k(k: int, precision: float):
    # convenience function to record precision at k
    metrics_manager.record_precision_at_k(k, precision)


def record_cache_hit_rate(cache_type: str, hit_rate: float):
    # convenience function to record cache hit rate
    metrics_manager.record_cache_hit(cache_type, hit_rate)


# structured logging helper
def log_with_trace(logger: logging.Logger, level: int, message: str, **kwargs):
    # log with trace_id correlation
    trace_id = get_current_trace_id()
    extra = {'trace_id': trace_id} if trace_id else {}
    extra.update(kwargs)
    
    # format as JSON-like structured log
    log_data = {
        'message': message,
        **extra
    }
    
    logger.log(level, str(log_data))

