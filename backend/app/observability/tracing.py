from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)


def configure_tracing() -> TracerProvider | None:
    """Set up OpenTelemetry tracing. Returns None when OTEL is disabled."""
    settings = get_settings()
    if not settings.otel.enabled:
        return None

    raise NotImplementedError


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
