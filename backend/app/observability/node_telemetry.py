"""
Node telemetry — wraps a LangGraph node function with:

  • An OTel span  (workflow.node.<name>)
  • Execution timing recorded in workflow_node_duration_seconds
  • Error counting in workflow_node_errors_total
  • trace_id / span_id bound to structlog context vars for the duration
    of the node so every log emitted inside the node carries them

Usage in workflow.py
--------------------
    from app.observability.node_telemetry import observe_node

    graph.add_node("router",  observe_node("router",  router_node))
    graph.add_node("retriever", observe_node("retriever", retriever_node))
    ...

The observe_node() wrapper is transparent: it preserves the original
function's signature and name so LangGraph introspection still works.
"""

import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.core.logging import get_logger
from app.graph.state import AppState
from app.observability.metrics import record_node_duration, record_node_error
from app.observability.tracing import get_tracer

_logger = get_logger(__name__)

_TRACER_NAME = "app.workflow.nodes"


def observe_node(
    name: str,
    func: Callable[[AppState], Coroutine[Any, Any, dict]],
) -> Callable[[AppState], Coroutine[Any, Any, dict]]:
    """Return *func* wrapped with OTel span, timing, and error metrics.

    The wrapper is transparent:
    - ``__name__`` and ``__doc__`` are preserved via functools.wraps
    - Exceptions propagate unchanged after being recorded
    """

    @functools.wraps(func)
    async def _observed(state: AppState) -> dict:
        tracer = get_tracer(_TRACER_NAME)
        session_id = state.get("session_id", "")
        route = state.get("route") or "unknown"

        with tracer.start_as_current_span(f"workflow.node.{name}") as span:
            span.set_attribute("workflow.node", name)
            span.set_attribute("workflow.session_id", session_id)
            span.set_attribute("workflow.route", route)

            # Inject trace context into structlog so every log emitted by the
            # node body automatically includes trace_id and span_id.
            ctx = span.get_span_context()
            if ctx.is_valid:
                structlog.contextvars.bind_contextvars(
                    trace_id=format(ctx.trace_id, "032x"),
                    span_id=format(ctx.span_id, "016x"),
                    node=name,
                )

            start = time.perf_counter()
            try:
                result = await func(state)
                duration = time.perf_counter() - start
                record_node_duration(name, route, duration)
                _logger.debug(
                    "node_completed",
                    node=name,
                    session_id=session_id,
                    duration_ms=round(duration * 1000, 2),
                )
                return result

            except Exception as exc:
                duration = time.perf_counter() - start
                record_node_duration(name, route, duration)
                record_node_error(name, route, type(exc).__name__)
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                _logger.error(
                    "node_exception",
                    node=name,
                    session_id=session_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise

            finally:
                if ctx.is_valid:
                    structlog.contextvars.unbind_contextvars(
                        "trace_id", "span_id", "node"
                    )

    return _observed
