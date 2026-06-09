"""
OpenTelemetry metrics with Prometheus export.

Metrics are ALWAYS collected regardless of OTEL_ENABLED.  The Prometheus
reader registers metrics into the default prometheus_client registry, so
GET /metrics always returns current values.

Instruments
-----------
workflow_node_duration_seconds   Histogram  Time each node takes to execute
workflow_node_errors_total       Counter    Node failures by node & error type
workflow_sessions_total          Counter    Workflow sessions started
workflow_sessions_completed_total Counter   Sessions completed by route & outcome
workflow_approval_decisions_total Counter   Approval decisions by action
http_requests_total              Counter    HTTP requests by method, path, status
http_request_duration_seconds    Histogram  HTTP request latency
"""

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from app.core.config import get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)

_meter_provider: MeterProvider | None = None
_meter: metrics.Meter | None = None


# ── Metric instruments (created lazily) ───────────────────────────────────────

_node_duration: metrics.Histogram | None = None
_node_errors: metrics.Counter | None = None
_sessions_started: metrics.Counter | None = None
_sessions_completed: metrics.Counter | None = None
_approval_decisions: metrics.Counter | None = None
_http_requests: metrics.Counter | None = None
_http_duration: metrics.Histogram | None = None


def configure_metrics() -> MeterProvider:
    """Initialise the global MeterProvider with a Prometheus reader.

    Safe to call multiple times — subsequent calls return the cached provider.
    """
    global _meter_provider, _meter
    global _node_duration, _node_errors
    global _sessions_started, _sessions_completed, _approval_decisions
    global _http_requests, _http_duration

    if _meter_provider is not None:
        return _meter_provider

    settings = get_settings()
    resource = Resource.create({SERVICE_NAME: settings.otel.service_name})
    reader = PrometheusMetricReader()
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(_meter_provider)

    _meter = _meter_provider.get_meter("app.workflow", version="0.1.0")

    _node_duration = _meter.create_histogram(
        name="workflow_node_duration_seconds",
        description="Execution time of each workflow node in seconds",
        unit="s",
    )
    _node_errors = _meter.create_counter(
        name="workflow_node_errors_total",
        description="Number of workflow node failures",
    )
    _sessions_started = _meter.create_counter(
        name="workflow_sessions_total",
        description="Number of workflow sessions started",
    )
    _sessions_completed = _meter.create_counter(
        name="workflow_sessions_completed_total",
        description="Number of workflow sessions that reached a terminal state",
    )
    _approval_decisions = _meter.create_counter(
        name="workflow_approval_decisions_total",
        description="Number of human-approval decisions submitted",
    )
    _http_requests = _meter.create_counter(
        name="http_requests_total",
        description="Number of HTTP requests handled",
    )
    _http_duration = _meter.create_histogram(
        name="http_request_duration_seconds",
        description="HTTP request latency in seconds",
        unit="s",
    )

    _logger.info("metrics_configured", service=settings.otel.service_name)
    return _meter_provider


# ── Recording helpers ─────────────────────────────────────────────────────────


def record_node_duration(node: str, route: str, duration_s: float) -> None:
    if _node_duration is not None:
        _node_duration.record(duration_s, attributes={"node": node, "route": route})


def record_node_error(node: str, route: str, error_type: str) -> None:
    if _node_errors is not None:
        _node_errors.add(1, attributes={"node": node, "route": route, "error_type": error_type})


def record_session_started() -> None:
    if _sessions_started is not None:
        _sessions_started.add(1)


def record_session_completed(route: str, outcome: str) -> None:
    """outcome: completed | rejected | failed"""
    if _sessions_completed is not None:
        _sessions_completed.add(1, attributes={"route": route, "outcome": outcome})


def record_approval_decision(action: str) -> None:
    """action: approved | rejected"""
    if _approval_decisions is not None:
        _approval_decisions.add(1, attributes={"action": action})


def record_http_request(method: str, path: str, status_code: int, duration_s: float) -> None:
    attrs = {"method": method, "path": path, "status_code": str(status_code)}
    if _http_requests is not None:
        _http_requests.add(1, attributes=attrs)
    if _http_duration is not None:
        _http_duration.record(duration_s, attributes=attrs)
