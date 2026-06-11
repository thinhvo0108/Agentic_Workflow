"""
Structured JSON logging via structlog.

Every log record automatically includes:
  service     — "agentic-workflow" (constant)
  trace_id    — hex-encoded OTel trace ID when a span is active
  span_id     — hex-encoded OTel span ID when a span is active
  request_id  — UUID bound by RequestTracingMiddleware (via contextvars)

Log lines are emitted as JSON to stdout, which is consumed by log
aggregators such as Loki, Datadog, or CloudWatch Logs.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# ── Processors ────────────────────────────────────────────────────────────────


def _add_service(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    event_dict["service"] = "agentic-workflow"
    return event_dict


def _inject_otel_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add OTel trace_id and span_id when a span is active.

    Uses setdefault so context-var bindings (e.g. from node_telemetry) take
    precedence over the span-derived values, but the processor still acts as
    a reliable fallback for all non-node code running inside a span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict.setdefault("trace_id", format(ctx.trace_id, "032x"))
            event_dict.setdefault("span_id", format(ctx.span_id, "016x"))
    except Exception:
        pass  # OTel not installed / not configured — skip silently
    return event_dict


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output.

    Call once during application startup.  Safe to call multiple times —
    structlog stores configuration globally.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_service,
        _inject_otel_context,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
