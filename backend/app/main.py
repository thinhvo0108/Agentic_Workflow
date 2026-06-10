from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.routes import router
from app.checkpoints.repository import CheckpointRepository
from app.core.config import get_settings
from app.core.exceptions import AgenticWorkflowError, ApprovalError
from app.core.logging import configure_logging, get_logger
from app.graph.nodes.checkpoint import set_repository
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

    # asyncpg pool → CheckpointRepository (custom audit records table)
    pool = await asyncpg.create_pool(
        settings.postgres.sync_dsn,
        min_size=2,
        max_size=settings.postgres.pool_size,
    )
    repo = CheckpointRepository(pool=pool)
    await repo.setup()
    set_repository(repo)

    # Pre-warm the reranker model so the first query doesn't pay the load cost.
    # bge-reranker-large is ~1 GB and takes 30-60 s on CPU; loading it here
    # keeps the class-level cache warm after every uvicorn --reload cycle.
    try:
        from app.rag.reranker import RerankerService
        _logger.info("prewarming_reranker")
        await RerankerService().warm_up()
        _logger.info("reranker_ready")
    except Exception as exc:
        _logger.warning("reranker_prewarm_failed", error=str(exc))

    # AsyncPostgresSaver → LangGraph graph state persistence (survives restarts)
    async with AsyncPostgresSaver.from_conn_string(settings.postgres.sync_dsn) as checkpointer:
        await checkpointer.setup()
        init_workflow(checkpointer)
        _logger.info("startup", env=settings.app.env, port=settings.app.port)
        yield

    await pool.close()
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
