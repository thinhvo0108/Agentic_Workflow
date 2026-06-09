"""
PostgresCheckpointStore — manages two distinct PostgreSQL checkpoint backends.

Layer 1  (LangGraph built-in)
    AsyncPostgresSaver from langgraph-checkpoint-postgres.
    Uses psycopg3 and automatically persists the full LangGraph state after
    every node transition.  Required for interrupt_before / resume to work.

Layer 2  (Application repository)
    CheckpointRepository backed by asyncpg.
    Writes curated, human-readable audit records to workflow_checkpoints so
    operators can query checkpoint history without parsing LangGraph's internal
    schema.

Lifecycle
---------
Call await store.setup() once at application startup.
Call await store.teardown() on shutdown to close both pools cleanly.

Dependency injection
--------------------
Both pools can be injected for testing or replaced in integration harnesses.
"""

from typing import Any

import asyncpg
from psycopg_pool import AsyncConnectionPool

from app.checkpoints.repository import CheckpointRepository
from app.core.config import get_settings
from app.core.exceptions import CheckpointError
from app.core.logging import get_logger

_logger = get_logger(__name__)


class PostgresCheckpointStore:
    """Manages LangGraph and application-level PostgreSQL checkpointing.

    Parameters
    ----------
    asyncpg_pool:
        Optional pre-built asyncpg pool for the repository layer.
    psycopg_pool:
        Optional pre-built psycopg3 pool for the LangGraph checkpointer layer.
    """

    def __init__(
        self,
        asyncpg_pool: asyncpg.Pool | None = None,
        psycopg_pool: AsyncConnectionPool | None = None,
    ) -> None:
        self._settings = get_settings()
        self._asyncpg_pool: asyncpg.Pool | None = asyncpg_pool
        self._psycopg_pool: AsyncConnectionPool | None = psycopg_pool
        self._repository: CheckpointRepository | None = None
        self._langgraph_saver: Any | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Open connection pools and bootstrap schemas.

        Creates the asyncpg pool (application repository) and the psycopg3
        pool (LangGraph checkpointer) if they were not injected.  Then runs
        setup() on both layers to create their respective tables.
        """
        await self._setup_asyncpg_layer()
        await self._setup_psycopg_layer()
        _logger.info("checkpoint_store_ready")

    async def teardown(self) -> None:
        """Close both connection pools."""
        if self._asyncpg_pool:
            await self._asyncpg_pool.close()
            _logger.info("asyncpg_pool_closed")
        if self._psycopg_pool:
            await self._psycopg_pool.close()
            _logger.info("psycopg_pool_closed")

    # ── Repository accessor ────────────────────────────────────────────────────

    def repository(self) -> CheckpointRepository:
        """Return the CheckpointRepository.

        Raises
        ------
        CheckpointError
            If setup() has not been called yet.
        """
        if self._repository is None:
            raise CheckpointError(
                "CheckpointRepository is not initialised — call setup() first"
            )
        return self._repository

    # ── LangGraph checkpointer ─────────────────────────────────────────────────

    def as_langgraph_checkpointer(self) -> Any:
        """Return the AsyncPostgresSaver for LangGraph's built-in checkpointing.

        Raises
        ------
        CheckpointError
            If setup() has not been called yet.
        """
        if self._langgraph_saver is None:
            raise CheckpointError(
                "LangGraph checkpointer is not initialised — call setup() first"
            )
        return self._langgraph_saver

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _setup_asyncpg_layer(self) -> None:
        pg = self._settings.postgres
        if self._asyncpg_pool is None:
            try:
                self._asyncpg_pool = await asyncpg.create_pool(
                    host=pg.host,
                    port=pg.port,
                    database=pg.db,
                    user=pg.user,
                    password=pg.password,
                    min_size=2,
                    max_size=pg.pool_size,
                )
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to create asyncpg pool: {exc}"
                ) from exc

        self._repository = CheckpointRepository(self._asyncpg_pool)
        await self._repository.setup()
        _logger.info("asyncpg_layer_ready")

    async def _setup_psycopg_layer(self) -> None:
        pg = self._settings.postgres
        conninfo = (
            f"host={pg.host} port={pg.port} dbname={pg.db} "
            f"user={pg.user} password={pg.password}"
        )
        if self._psycopg_pool is None:
            try:
                self._psycopg_pool = AsyncConnectionPool(
                    conninfo=conninfo,
                    min_size=1,
                    max_size=pg.pool_size,
                    open=False,
                )
                await self._psycopg_pool.open()
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to create psycopg3 pool: {exc}"
                ) from exc

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            self._langgraph_saver = AsyncPostgresSaver(self._psycopg_pool)
            await self._langgraph_saver.setup()
        except Exception as exc:
            raise CheckpointError(
                f"Failed to initialise LangGraph AsyncPostgresSaver: {exc}"
            ) from exc

        _logger.info("psycopg_layer_ready")
