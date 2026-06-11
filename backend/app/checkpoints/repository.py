"""
CheckpointRepository — repository pattern over the workflow_checkpoints table.

Uses asyncpg directly (not SQLAlchemy) to keep the query surface explicit and
give full control over JSONB serialisation.  Every public method acquires a
connection from the pool, runs one parameterised query, and releases it.

The pool is injected via the constructor so tests can supply an in-process
connection pool against a real test database without touching application
settings.

DDL
---
The setup() method issues CREATE TABLE / INDEX IF NOT EXISTS so the schema can
be bootstrapped without Alembic in development and test environments.  In
production the DDL should be managed via migrations; setup() is idempotent.
"""

import json

import asyncpg

from app.checkpoints.models import CheckpointRecord, CheckpointStage
from app.core.exceptions import CheckpointError
from app.core.logging import get_logger

_logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    id                    BIGSERIAL    PRIMARY KEY,
    session_id            TEXT         NOT NULL,
    stage                 TEXT         NOT NULL,
    query                 TEXT         NOT NULL,
    route                 TEXT,
    retrieved_doc_count   INTEGER      NOT NULL DEFAULT 0,
    reranked_doc_count    INTEGER      NOT NULL DEFAULT 0,
    has_draft             BOOLEAN      NOT NULL DEFAULT FALSE,
    has_structured_output BOOLEAN      NOT NULL DEFAULT FALSE,
    approval_status       TEXT,
    error_count           INTEGER      NOT NULL DEFAULT 0,
    state_snapshot        JSONB        NOT NULL DEFAULT '{}',
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_wcp_session_id
    ON workflow_checkpoints (session_id);
CREATE INDEX IF NOT EXISTS idx_wcp_session_stage
    ON workflow_checkpoints (session_id, stage);
"""

_INSERT_SQL = """
INSERT INTO workflow_checkpoints (
    session_id, stage, query, route,
    retrieved_doc_count, reranked_doc_count,
    has_draft, has_structured_output, approval_status,
    error_count, state_snapshot
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
RETURNING id, created_at
"""

_SELECT_BY_SESSION_SQL = """
SELECT id, session_id, stage, query, route,
       retrieved_doc_count, reranked_doc_count,
       has_draft, has_structured_output, approval_status,
       error_count, state_snapshot, created_at
FROM   workflow_checkpoints
WHERE  session_id = $1
ORDER  BY created_at ASC
"""

_SELECT_LATEST_SQL = """
SELECT id, session_id, stage, query, route,
       retrieved_doc_count, reranked_doc_count,
       has_draft, has_structured_output, approval_status,
       error_count, state_snapshot, created_at
FROM   workflow_checkpoints
WHERE  session_id = $1
ORDER  BY created_at DESC
LIMIT  1
"""

_SELECT_BY_STAGE_SQL = """
SELECT id, session_id, stage, query, route,
       retrieved_doc_count, reranked_doc_count,
       has_draft, has_structured_output, approval_status,
       error_count, state_snapshot, created_at
FROM   workflow_checkpoints
WHERE  session_id = $1 AND stage = $2
ORDER  BY created_at DESC
LIMIT  1
"""

_COUNT_SQL = "SELECT COUNT(*) FROM workflow_checkpoints WHERE session_id = $1"

_DELETE_SQL = "DELETE FROM workflow_checkpoints WHERE session_id = $1"


def _row_to_record(row: asyncpg.Record) -> CheckpointRecord:
    """Convert an asyncpg Record to a CheckpointRecord."""
    snapshot = row["state_snapshot"]
    # asyncpg returns JSONB columns as dicts already.
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    return CheckpointRecord(
        id=row["id"],
        session_id=row["session_id"],
        stage=CheckpointStage(row["stage"]),
        query=row["query"],
        route=row["route"],
        retrieved_doc_count=row["retrieved_doc_count"],
        reranked_doc_count=row["reranked_doc_count"],
        has_draft=row["has_draft"],
        has_structured_output=row["has_structured_output"],
        approval_status=row["approval_status"],
        error_count=row["error_count"],
        state_snapshot=snapshot or {},
        created_at=row["created_at"],
    )


class CheckpointRepository:
    """Async repository for workflow_checkpoints audit records.

    Parameters
    ----------
    pool:
        An asyncpg connection pool.  The caller owns the pool lifecycle.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Create the workflow_checkpoints table and indexes if absent.

        Idempotent — safe to call on every startup.
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(_DDL)
        except Exception as exc:
            raise CheckpointError(f"Failed to bootstrap checkpoint schema: {exc}") from exc
        _logger.info("checkpoint_schema_ready")

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save(self, record: CheckpointRecord) -> CheckpointRecord:
        """Persist *record* and return it with id and created_at populated.

        Raises
        ------
        CheckpointError
            If the INSERT fails.
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    _INSERT_SQL,
                    record.session_id,
                    str(record.stage),
                    record.query,
                    record.route,
                    record.retrieved_doc_count,
                    record.reranked_doc_count,
                    record.has_draft,
                    record.has_structured_output,
                    record.approval_status,
                    record.error_count,
                    json.dumps(record.state_snapshot),
                )
        except Exception as exc:
            raise CheckpointError(
                f"Failed to save checkpoint for session '{record.session_id}': {exc}"
            ) from exc

        return record.model_copy(update={"id": row["id"], "created_at": row["created_at"]})

    # ── Read ──────────────────────────────────────────────────────────────────

    async def list_by_session(self, session_id: str) -> list[CheckpointRecord]:
        """Return all checkpoints for *session_id*, ordered oldest-first."""
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(_SELECT_BY_SESSION_SQL, session_id)
        except Exception as exc:
            raise CheckpointError(
                f"Failed to list checkpoints for session '{session_id}': {exc}"
            ) from exc
        return [_row_to_record(r) for r in rows]

    async def get_latest(self, session_id: str) -> CheckpointRecord | None:
        """Return the most recent checkpoint for *session_id*, or None."""
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(_SELECT_LATEST_SQL, session_id)
        except Exception as exc:
            raise CheckpointError(
                f"Failed to get latest checkpoint for session '{session_id}': {exc}"
            ) from exc
        return _row_to_record(row) if row else None

    async def get_by_stage(
        self, session_id: str, stage: CheckpointStage
    ) -> CheckpointRecord | None:
        """Return the most recent checkpoint at *stage* for *session_id*, or None."""
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(_SELECT_BY_STAGE_SQL, session_id, str(stage))
        except Exception as exc:
            raise CheckpointError(
                f"Failed to get '{stage}' checkpoint for session '{session_id}': {exc}"
            ) from exc
        return _row_to_record(row) if row else None

    async def count(self, session_id: str) -> int:
        """Return the number of checkpoint records for *session_id*."""
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval(_COUNT_SQL, session_id)  # type: ignore[no-any-return]
        except Exception as exc:
            raise CheckpointError(
                f"Failed to count checkpoints for session '{session_id}': {exc}"
            ) from exc

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_session(self, session_id: str) -> int:
        """Delete all checkpoints for *session_id*.

        Returns the number of rows deleted.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(_DELETE_SQL, session_id)
            # asyncpg returns e.g. "DELETE 3"
            return int(result.split()[-1])
        except Exception as exc:
            raise CheckpointError(
                f"Failed to delete checkpoints for session '{session_id}': {exc}"
            ) from exc
