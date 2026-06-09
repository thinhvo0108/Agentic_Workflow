from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.api.routes import router
from app.core.config import get_settings
from app.core.exceptions import AgenticWorkflowError, ApprovalError
from app.core.logging import configure_logging, get_logger
from app.observability.metrics import configure_metrics
from app.observability.middleware import RequestTracingMiddleware
from app.observability.tracing import configure_tracing

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_metrics()
    configure_tracing()

    from app.api.dependencies import init_workflow
    init_workflow()

    _logger.info("startup", env=settings.app.env, port=settings.app.port)
    yield
    _logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Agentic Workflow API",
        description="Production-style multi-agent AI workflow with RAG and human-in-the-loop",
        version="0.1.0",
        debug=settings.app.debug,
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app.env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Prometheus metrics endpoint ────────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False, tags=["ops"])
    def metrics_endpoint() -> Response:
        """Expose Prometheus metrics for scraping by Prometheus / Grafana."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── Health ─────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        return {"status": "ok"}

    # ── Exception handlers ─────────────────────────────────────────────────────
    @app.exception_handler(ApprovalError)
    async def approval_error_handler(request: Request, exc: ApprovalError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": exc.message},
        )

    @app.exception_handler(AgenticWorkflowError)
    async def agentic_error_handler(request: Request, exc: AgenticWorkflowError) -> JSONResponse:
        _logger.error("unhandled_application_error", error=str(exc), details=exc.details)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.message, "detail": str(exc.details) or None},
        )

    return app


app = create_app()
