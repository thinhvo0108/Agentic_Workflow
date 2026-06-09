"""
Request tracing middleware.

For every HTTP request:
  1. Generates a unique request_id (UUID4) and attaches it to the response
     as X-Request-ID so clients can correlate log lines with a request.
  2. Binds request_id to structlog context vars so every log emitted during
     the request automatically includes it without explicit passing.
  3. Measures request duration and records it via the metrics module.
  4. Clears structlog context vars after the response is sent so they do
     not leak into the next request on the same event-loop task.
"""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger
from app.observability.metrics import record_http_request

_logger = get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware that attaches a request_id to logs and measures latency."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.perf_counter() - start
            status_code = response.status_code if response is not None else 500

            # Normalise path: replace UUIDs/IDs with placeholders so the
            # cardinality of the 'path' label stays bounded.
            path = _normalise_path(request.url.path)
            record_http_request(
                method=request.method,
                path=path,
                status_code=status_code,
                duration_s=duration,
            )
            _logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )

            if response is not None:
                response.headers[_REQUEST_ID_HEADER] = request_id

            structlog.contextvars.unbind_contextvars("request_id")


def _normalise_path(path: str) -> str:
    """Replace UUID-like path segments with {id} to cap Prometheus cardinality."""
    import re
    return re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "{id}",
        path,
        flags=re.IGNORECASE,
    )
