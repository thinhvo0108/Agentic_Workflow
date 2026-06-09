# Observability

This document covers configuration, usage examples, and operational guidance for the
OpenTelemetry tracing, Prometheus metrics, and structured logging stack.

---

## Architecture overview

```
FastAPI app
  │
  ├── RequestTracingMiddleware          ← injects request_id, records HTTP metrics
  │
  ├── LangGraph nodes (observe_node)   ← OTel span + histogram + error counter
  │                                        + trace_id/span_id in structlog context vars
  │
  ├── /metrics (Prometheus scrape)     ← prometheus_client registry
  │
  └── OTLP gRPC exporter               ← sends spans to Jaeger / Tempo / Grafana Cloud
```

---

## Configuration

All observability settings live in `app/core/config.py` under the `OtelSettings` block.

| Environment variable         | Default                     | Description                             |
|------------------------------|-----------------------------|-----------------------------------------|
| `OTEL_ENABLED`               | `false`                     | Enable/disable OTel tracing             |
| `OTEL_ENDPOINT`              | `http://localhost:4317`     | OTLP gRPC collector endpoint            |
| `OTEL_SERVICE_NAME`          | `agentic-workflow`          | Service name reported to the collector  |
| `LOG_LEVEL`                  | `INFO`                      | Structlog minimum level                 |

When `OTEL_ENABLED=false` (the default), a NoOp tracer provider is registered so all
`get_tracer()` / `start_as_current_span()` calls are zero-cost no-ops. No network
connection is attempted.

### Minimal local setup

```bash
# .env
OTEL_ENABLED=true
OTEL_ENDPOINT=http://localhost:4317   # Jaeger all-in-one
OTEL_SERVICE_NAME=agentic-workflow
```

Run a local Jaeger instance:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \   # UI
  -p 4317:4317 \     # OTLP gRPC
  jaegertracing/all-in-one:latest
```

Open `http://localhost:16686` → select `agentic-workflow` → view traces.

---

## Tracing

### `configure_tracing()`

Called once at application startup (inside the FastAPI `lifespan`).

```python
from app.observability.tracing import configure_tracing
configure_tracing()   # reads settings automatically
```

- Creates a `TracerProvider` with `BatchSpanProcessor` + OTLP gRPC exporter.
- Instruments FastAPI automatically via `FastAPIInstrumentor` (adds spans for every
  HTTP request, attaches `http.method`, `http.route`, `http.status_code` attributes).
- Returns `None` and registers a NoOp provider when `OTEL_ENABLED=false`.

### `get_tracer(name)`

Returns a `trace.Tracer` scoped to the given instrument name.

```python
from app.observability.tracing import get_tracer

tracer = get_tracer("app.services.approval")

with tracer.start_as_current_span("approval.check") as span:
    span.set_attribute("session_id", session_id)
    # ... business logic
```

### Automatic node spans

Every LangGraph node is wrapped by `observe_node()` in `app/graph/workflow.py`.
Each node execution automatically produces:

- A span named `workflow.node.<name>` (e.g. `workflow.node.router`)
- Span attributes: `workflow.node`, `workflow.session_id`, `workflow.route`
- Exception recording (`span.record_exception`) on failure

---

## Metrics

### `configure_metrics()`

Called once at startup. Safe to call multiple times (idempotent).

```python
from app.observability.metrics import configure_metrics
configure_metrics()
```

Uses a `PrometheusMetricReader` — metrics are collected on-demand when `/metrics`
is scraped, with no push interval required.

### Available metrics

| Metric name                            | Type      | Labels                              |
|----------------------------------------|-----------|-------------------------------------|
| `workflow_node_duration_seconds`       | Histogram | `node`, `route`                     |
| `workflow_node_errors_total`           | Counter   | `node`, `route`, `error_type`       |
| `workflow_sessions_total`              | Counter   | —                                   |
| `workflow_sessions_completed_total`    | Counter   | `route`, `outcome`                  |
| `workflow_approval_decisions_total`    | Counter   | `action`                            |
| `http_requests_total`                  | Counter   | `method`, `path`, `status_code`     |
| `http_request_duration_seconds`        | Histogram | `method`, `path`                    |

`path` labels are normalised — UUID path segments are replaced with `{id}` to prevent
unbounded label cardinality.

### `/metrics` endpoint

```
GET /metrics
```

Returns Prometheus text format, suitable for `prometheus.yml` scrape config:

```yaml
scrape_configs:
  - job_name: agentic-workflow
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```

### Recording helpers

Call these from application code (outside of observe_node, which calls them automatically):

```python
from app.observability.metrics import (
    record_session_started,
    record_session_completed,
    record_approval_decision,
)

# When a new session is created
record_session_started()

# When a session finishes (route="research", outcome="completed"|"failed"|"rejected")
record_session_completed(route="research", outcome="completed")

# When an operator approves or rejects
record_approval_decision(action="approved")
```

---

## Structured logging

### `configure_logging(log_level)`

Sets up structlog with:
- JSON rendering (one JSON object per line)
- Automatic `timestamp`, `level`, `logger` fields
- `_inject_otel_context` processor that adds `trace_id` and `span_id` when a
  recording OTel span is active

```python
from app.core.logging import configure_logging
configure_logging("INFO")
```

### Using the logger

```python
from app.core.logging import get_logger

_logger = get_logger(__name__)

_logger.info("retrieval_complete", doc_count=10, session_id=session_id)
_logger.warning("low_confidence", score=0.42)
_logger.error("node_failed", node="reranker", error=str(exc))
```

### Log output example

When a span is active (e.g. inside `observe_node`):

```json
{
  "timestamp": "2026-06-09T10:23:45.123456Z",
  "level": "info",
  "logger": "app.graph.nodes.router",
  "event": "route_decided",
  "route": "research",
  "session_id": "sess-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "request_id": "9b8a3e12-f1c4-4d2e-8a0b-1234567890ab"
}
```

`trace_id` links this log line directly to the Jaeger trace for the same request.
`request_id` is injected by `RequestTracingMiddleware` for each HTTP request.

### `_inject_otel_context` processor

This is the structlog processor that links logs to traces. It is added automatically
by `configure_logging()`. You do not need to call it manually.

It uses `setdefault` so that pre-bound values (e.g. from a `contextvars.bind_contextvars`
call) are never overwritten.

---

## Middleware

`RequestTracingMiddleware` (added in `create_app()`) does three things per request:

1. Generates a unique `request_id` (UUID v4)
2. Binds `request_id` to structlog context vars (all log lines for this request include it)
3. Returns `X-Request-ID` in the response headers for client-side correlation
4. Records `http_requests_total` and `http_request_duration_seconds` Prometheus metrics

---

## Grafana dashboard (example queries)

### Node error rate by node

```promql
rate(workflow_node_errors_total[5m])
```

### p99 node latency

```promql
histogram_quantile(0.99, rate(workflow_node_duration_seconds_bucket[5m]))
```

### Session throughput

```promql
rate(workflow_sessions_total[1m])
```

### HTTP request rate by route

```promql
rate(http_requests_total[1m])
```

### Approval decision breakdown

```promql
sum by (action) (rate(workflow_approval_decisions_total[5m]))
```

---

## Adding telemetry to a new node

Any future LangGraph node gets automatic telemetry by wrapping it with `observe_node()`
in `workflow.py`:

```python
from app.observability.node_telemetry import observe_node
from app.graph.nodes.my_new_node import my_new_node_func

graph.add_node("my_new_node", observe_node("my_new_node", my_new_node_func))
```

That is the only change required — no modifications to the node function itself.

### Adding a custom span inside a node

```python
from app.observability.tracing import get_tracer

_tracer = get_tracer(__name__)

async def my_new_node_func(state: AppState) -> dict:
    with _tracer.start_as_current_span("my_new_node.expensive_step") as span:
        span.set_attribute("item_count", len(state["documents"]))
        # ... logic
```

Child spans appear nested under the parent `workflow.node.my_new_node` span in Jaeger.
