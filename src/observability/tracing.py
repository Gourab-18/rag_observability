# OpenTelemetry tracing for RAG pipeline

from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# OpenTelemetry imports with optional installation
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not installed. Tracing will be disabled. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc")

from src.config import settings


class TracingManager:
    # manages OpenTelemetry tracing for RAG pipeline
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.tracer = None
            self._setup_tracer()
            TracingManager._initialized = True
    
    def _setup_tracer(self):
        # setup OpenTelemetry tracer
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Tracing disabled.")
            return
        
        try:
            # create resource
            resource = Resource.create({
                "service.name": "rag-pipeline",
                "service.version": "1.0.0"
            })
            
            # create tracer provider
            provider = TracerProvider(resource=resource)
            
            # add console exporter (for development)
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            
            # add OTLP exporter if endpoint is configured
            if settings.otlp_endpoint:
                try:
                    otlp_exporter = OTLPSpanExporter(
                        endpoint=settings.otlp_endpoint,
                        insecure=True  # set to False for HTTPS
                    )
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info(f"OTLP exporter configured: {settings.otlp_endpoint}")
                except Exception as e:
                    logger.warning(f"Failed to configure OTLP exporter: {e}")
            
            # set global tracer provider
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            self.tracer = None
    
    @contextmanager
    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        # create a span context manager
        if not self.tracer:
            # no-op if tracing is disabled
            yield None
            return
        
        from opentelemetry.trace import get_current_span
        with self.tracer.start_as_current_span(name) as span:
            # set attributes
            if attributes and span:
                for key, value in attributes.items():
                    if value is not None:
                        span.set_attribute(key, str(value))
            
            # set trace ID if provided
            if trace_id and span:
                span.set_attribute("trace_id", trace_id)
            
            try:
                yield span
                if span:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                if span:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                raise


# global tracing manager instance
tracing_manager = TracingManager()


def create_trace(query_id: str) -> str:
    # create a new trace for a query and return trace_id
    if not tracing_manager.tracer:
        # fallback to UUID if tracing is disabled
        import uuid
        return str(uuid.uuid4())
    
    from opentelemetry.trace import get_tracer
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query_id", query_id)
        trace_id = format(span.get_span_context().trace_id, '032x')
    return trace_id


def get_current_trace_id() -> Optional[str]:
    # get current trace ID from active span
    if not tracing_manager.tracer:
        return None
    
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None

