"""
OpenTelemetry distributed tracing setup.

When OTEL_ENABLED = true:
  - TracerProvider backed by BatchSpanProcessor → OTLP gRPC exporter
    (points to Jaeger / OTEL Collector at OTEL_EXPORTER_OTLP_ENDPOINT)
  - FastAPI requests are automatically instrumented

When OTEL_ENABLED = false:
  - A NoOp TracerProvider is registered so trace.get_current_span() always
    works and log processors can call it safely without branching
  - No spans are exported or stored

The service resource is always set so traces are tagged correctly.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import NoOpTracer

from app.core.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)

_provider: TracerProvider | None = None


def _build_resource() -> Resource:
    settings = get_settings()
    return Resource.create(
        {
            SERVICE_NAME: settings.otel.service_name,
            SERVICE_VERSION: "0.1.0",
            DEPLOYMENT_ENVIRONMENT: settings.app.env,
        }
    )


def configure_tracing() -> TracerProvider | None:
    """Initialise the global TracerProvider.

    Returns the configured provider (may be a no-op) or None when already
    called.  Safe to call multiple times — subsequent calls are no-ops.
    """
    global _provider
    settings = get_settings()

    if not settings.otel.enabled:
        # Register a no-op provider so span APIs never raise.
        from opentelemetry.sdk.trace import TracerProvider as _TP
        noop = _TP(resource=_build_resource())
        trace.set_tracer_provider(noop)
        _logger.info("tracing_disabled")
        return None

    if _provider is not None:
        return _provider

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        exporter: SpanExporter = OTLPSpanExporter(
            endpoint=settings.otel.exporter_otlp_endpoint,
            insecure=True,
        )
    except Exception as exc:
        _logger.warning("otlp_exporter_unavailable", error=str(exc))
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()

    resource = _build_resource()
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)

    _maybe_instrument_fastapi()

    _logger.info(
        "tracing_configured",
        endpoint=settings.otel.exporter_otlp_endpoint,
        service=settings.otel.service_name,
    )
    return _provider


def _maybe_instrument_fastapi() -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument()
        _logger.info("fastapi_instrumented")
    except Exception as exc:
        _logger.warning("fastapi_instrumentation_failed", error=str(exc))


def get_tracer(name: str) -> trace.Tracer:
    """Return an OTel Tracer bound to *name*.

    Works regardless of whether tracing is enabled — the global provider may
    be a no-op that discards spans immediately.
    """
    return trace.get_tracer(name)
