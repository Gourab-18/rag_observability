# Prometheus metrics for RAG pipeline

from typing import Optional, Dict, List
import logging
import time

logger = logging.getLogger(__name__)

# Prometheus imports with optional installation
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    from prometheus_client.core import CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Metrics will be disabled. Install with: pip install prometheus-client")

from src.config import settings


class MetricsManager:
    # manages Prometheus metrics for RAG pipeline
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.registry = REGISTRY if PROMETHEUS_AVAILABLE else None
            self.metrics = {}
            self._setup_metrics()
            MetricsManager._initialized = True
    
    def _setup_metrics(self):
        # setup Prometheus metrics
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available. Metrics disabled.")
            return
        
        try:
            # request latency histogram (seconds)
            self.metrics['rag_request_latency_seconds'] = Histogram(
                'rag_request_latency_seconds',
                'RAG request latency in seconds',
                ['stage'],  # stage: ingestion, embedding, retrieval, generation
                registry=self.registry
            )
            
            # token usage counter (by provider)
            self.metrics['rag_token_usage_total'] = Counter(
                'rag_token_usage_total',
                'Total tokens used in RAG pipeline',
                ['provider', 'type'],  # provider: openai, ollama | type: prompt, completion
                registry=self.registry
            )
            
            # cost per query gauge (USD)
            self.metrics['rag_cost_per_query'] = Gauge(
                'rag_cost_per_query',
                'Cost per query in USD',
                ['provider', 'model'],  # provider: openai, ollama | model: gpt-3.5-turbo, etc.
                registry=self.registry
            )
            
            # cache hit rate gauge
            self.metrics['rag_cache_hit_rate'] = Gauge(
                'rag_cache_hit_rate',
                'Cache hit rate (0.0 to 1.0)',
                ['cache_type'],  # cache_type: embeddings, retrieval, llm
                registry=self.registry
            )
            
            # precision at k histogram
            self.metrics['rag_precision_at_k'] = Histogram(
                'rag_precision_at_k',
                'Precision at k for retrieval quality',
                ['k'],  # k: 1, 3, 5
                registry=self.registry
            )
            
            # error rate counter
            self.metrics['rag_errors_total'] = Counter(
                'rag_errors_total',
                'Total errors in RAG pipeline',
                ['stage', 'error_type'],  # stage: ingestion, embedding, retrieval, generation
                registry=self.registry
            )
            
            # document count gauge
            self.metrics['rag_documents_total'] = Gauge(
                'rag_documents_total',
                'Total number of documents in the knowledge base',
                registry=self.registry
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            self.metrics = {}
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        # record a metric value with optional labels
        if not PROMETHEUS_AVAILABLE or name not in self.metrics:
            return
        
        try:
            metric = self.metrics[name]
            if labels:
                labeled_metric = metric.labels(**labels)
                # Gauge uses set(), Histogram uses observe()
                if isinstance(metric, Gauge):
                    labeled_metric.set(value)
                else:
                    labeled_metric.observe(value)
            else:
                # Gauge uses set(), Histogram uses observe()
                if isinstance(metric, Gauge):
                    metric.set(value)
                else:
                    metric.observe(value)
        except Exception as e:
            logger.warning(f"Failed to record metric {name}: {e}")
    
    def record_latency(self, stage: str, duration: float):
        # record latency for a pipeline stage
        self.record_metric('rag_request_latency_seconds', duration, {'stage': stage})
    
    def record_tokens(self, provider: str, token_type: str, count: int):
        # record token usage
        if not PROMETHEUS_AVAILABLE or 'rag_token_usage_total' not in self.metrics:
            return
        
        try:
            self.metrics['rag_token_usage_total'].labels(
                provider=provider,
                type=token_type
            ).inc(count)
        except Exception as e:
            logger.warning(f"Failed to record tokens: {e}")
    
    def record_cost(self, provider: str, model: str, cost: float):
        # record cost per query
        self.record_metric('rag_cost_per_query', cost, {'provider': provider, 'model': model})
    
    def record_cache_hit(self, cache_type: str, hit_rate: float):
        # record cache hit rate
        self.record_metric('rag_cache_hit_rate', hit_rate, {'cache_type': cache_type})
    
    def record_precision_at_k(self, k: int, precision: float):
        # record precision at k
        self.record_metric('rag_precision_at_k', precision, {'k': str(k)})
    
    def record_error(self, stage: str, error_type: str):
        # record an error
        if not PROMETHEUS_AVAILABLE or 'rag_errors_total' not in self.metrics:
            return
        
        try:
            self.metrics['rag_errors_total'].labels(
                stage=stage,
                error_type=error_type
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record error: {e}")
    
    def start_metrics_server(self):
        # start Prometheus metrics HTTP server
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available. Cannot start metrics server.")
            return
        
        try:
            start_http_server(settings.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


# global metrics manager instance
metrics_manager = MetricsManager()


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    # convenience function to record a metric
    metrics_manager.record_metric(name, value, labels)

