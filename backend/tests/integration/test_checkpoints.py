"""
Integration tests for the checkpoint repository and checkpoint_node.

These tests hit a real PostgreSQL instance.  If the database is unreachable
the entire module is automatically skipped — no patching or mocking.

Configuration
-------------
Set POSTGRES_TEST_DSN to override the default connection string, e.g.

    export POSTGRES_TEST_DSN="postgresql://postgres:postgres@localhost:5432/agentic_workflow_test"

or start the development stack with:

    docker compose up -d postgres

The fixture creates the workflow_checkpoints table, runs each test, and
truncates the table after every test so cases are fully isolated.

Test groups
-----------
TestCheckpointRecord         — Pydantic model validation (pure, no DB)
TestCheckpointStageInference — _determine_stage() and helpers (pure, no DB)
TestCheckpointRepositorySetup   — schema bootstrapping
TestCheckpointRepositorySave    — INSERT + RETURNING round-trip
TestCheckpointRepositoryList    — list_by_session ordering and isolation
TestCheckpointRepositoryGetLatest — get_latest behaviour
TestCheckpointRepositoryGetByStage — stage-targeted reads
TestCheckpointRepositoryCount   — COUNT queries
TestCheckpointRepositoryDelete  — DELETE + row count
TestCheckpointNode              — node → repository → DB round-trip
"""

import json
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Literal

import asyncpg
import pytest
from pydantic import ValidationError

from app.checkpoints.models import CheckpointRecord, CheckpointStage
from app.checkpoints.repository import CheckpointRepository
from app.graph.nodes.checkpoint import (
    _build_record,
    _build_snapshot,
    _determine_stage,
    checkpoint_node,
    set_repository,
)
from app.graph.state import RouteDecision, initial_state

# ── Connection fixtures ────────────────────────────────────────────────────────

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/agentic_workflow_test"


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    return os.getenv("POSTGRES_TEST_DSN", _DEFAULT_DSN)


@pytest.fixture
async def pg_pool(pg_dsn: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """Function-scoped pool — each test runs in its own event loop context."""
    try:
        pool = await asyncpg.create_pool(pg_dsn, min_size=1, max_size=3)
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available at {pg_dsn!r}: {exc}")
    yield pool
    await pool.close()


@pytest.fixture
async def repo(pg_pool: asyncpg.Pool) -> AsyncGenerator[CheckpointRepository, None]:
    r = CheckpointRepository(pg_pool)
    await r.setup()
    yield r
    async with pg_pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE workflow_checkpoints RESTART IDENTITY")


# ── Test factories ─────────────────────────────────────────────────────────────


def _record(
    session_id: str = "sess-001",
    stage: CheckpointStage = CheckpointStage.GENERATION,
    query: str = "How do transformers work?",
    route: str | None = "research",
    retrieved: int = 10,
    reranked: int = 3,
    has_draft: bool = True,
    has_so: bool = True,
    approval: str | None = "pending",
    errors: int = 0,
    snapshot: dict | None = None,
) -> CheckpointRecord:
    return CheckpointRecord(
        session_id=session_id,
        stage=stage,
        query=query,
        route=route,
        retrieved_doc_count=retrieved,
        reranked_doc_count=reranked,
        has_draft=has_draft,
        has_structured_output=has_so,
        approval_status=approval,
        error_count=errors,
        state_snapshot=snapshot or {"key": "value"},
    )


def _state(
    session_id: str = "sess-node",
    query: str = "test query",
    route: RouteDecision | None = None,
    with_retrieved: bool = False,
    with_reranked: bool = False,
    with_draft: bool = False,
    with_so: bool = False,
):
    s = initial_state(session_id=session_id, query=query)
    if route is not None:
        s["route"] = route
    if with_retrieved:
        s["retrieved_documents"] = [
            {"id": f"r{i}", "content": "c", "source": "s", "metadata": {}, "score": 0.8}
            for i in range(3)
        ]
    if with_reranked:
        s["reranked_documents"] = [
            {
                "id": f"rk{i}", "content": "c", "source": "s", "metadata": {},
                "retrieval_score": 0.8, "rerank_score": 0.9,
            }
            for i in range(2)
        ]
    if with_draft:
        s["draft_response"] = '{"summary": "xxxxxxxxxx", "answer": "xxxxxxxxxxxxxxxxxxxx", "citations": []}'
    if with_so:
        s["structured_output"] = {
            "summary": "Ten chars ok.",
            "answer": "This is a twenty character answer.",
            "citations": [],
        }
    return s


# ── Pure tests: CheckpointRecord ──────────────────────────────────────────────


class TestCheckpointRecord:
    def test_valid_record(self):
        r = _record()
        assert r.stage == CheckpointStage.GENERATION
        assert r.id is None
        assert r.created_at is None

    def test_rejects_negative_retrieved_doc_count(self):
        with pytest.raises(ValidationError):
            CheckpointRecord(
                session_id="s", stage=CheckpointStage.ROUTING,
                query="q", retrieved_doc_count=-1, state_snapshot={},
            )

    def test_rejects_negative_error_count(self):
        with pytest.raises(ValidationError):
            CheckpointRecord(
                session_id="s", stage=CheckpointStage.ROUTING,
                query="q", error_count=-1, state_snapshot={},
            )

    def test_all_stages_accepted(self):
        for stage in CheckpointStage:
            assert _record(stage=stage).stage == stage

    def test_state_snapshot_defaults_to_empty_dict(self):
        r = CheckpointRecord(
            session_id="s", stage=CheckpointStage.ROUTING, query="q",
        )
        assert r.state_snapshot == {}


# ── Pure tests: stage inference ───────────────────────────────────────────────


class TestCheckpointStageInference:
    def test_generation_stage_when_structured_output_present(self):
        assert _determine_stage(_state(with_so=True)) == CheckpointStage.GENERATION

    def test_generation_stage_when_draft_present(self):
        assert _determine_stage(_state(with_draft=True)) == CheckpointStage.GENERATION

    def test_reranking_stage_when_reranked_docs_present(self):
        assert _determine_stage(_state(with_reranked=True)) == CheckpointStage.RERANKING

    def test_retrieval_stage_when_retrieved_docs_present(self):
        assert _determine_stage(_state(with_retrieved=True)) == CheckpointStage.RETRIEVAL

    def test_routing_stage_when_only_route_present(self):
        assert _determine_stage(_state(route="research")) == CheckpointStage.ROUTING

    def test_routing_stage_for_bare_state(self):
        assert _determine_stage(_state()) == CheckpointStage.ROUTING

    def test_build_record_counts_retrieved_documents(self):
        assert _build_record(_state(with_retrieved=True)).retrieved_doc_count == 3

    def test_build_record_counts_reranked_documents(self):
        assert _build_record(_state(with_reranked=True)).reranked_doc_count == 2

    def test_build_record_sets_has_draft_flag(self):
        assert _build_record(_state(with_draft=True)).has_draft is True

    def test_build_record_sets_has_structured_output_flag(self):
        assert _build_record(_state(with_so=True)).has_structured_output is True

    def test_build_snapshot_includes_session_id(self):
        snap = _build_snapshot(_state(session_id="snap-sess"))
        assert snap["session_id"] == "snap-sess"

    def test_build_snapshot_includes_retrieved_doc_ids(self):
        snap = _build_snapshot(_state(with_retrieved=True))
        assert snap["retrieved_doc_ids"] == ["r0", "r1", "r2"]

    def test_build_snapshot_includes_structured_output_summary(self):
        snap = _build_snapshot(_state(with_so=True))
        assert "structured_output_summary" in snap
        assert "citation_count" in snap["structured_output_summary"]

    def test_build_snapshot_is_json_serialisable(self):
        snap = _build_snapshot(_state(with_so=True, with_retrieved=True, with_reranked=True))
        json.dumps(snap)  # must not raise


# ── Integration: setup ────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositorySetup:
    async def test_setup_creates_table(self, pg_pool: asyncpg.Pool, repo: CheckpointRepository):
        async with pg_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'workflow_checkpoints'"
            )
        assert count == 1

    async def test_setup_is_idempotent(self, repo: CheckpointRepository):
        await repo.setup()
        await repo.setup()  # second call must not raise


# ── Integration: save ─────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositorySave:
    async def test_save_returns_record_with_id(self, repo: CheckpointRepository):
        saved = await repo.save(_record())
        assert saved.id is not None
        assert saved.id > 0

    async def test_save_populates_created_at(self, repo: CheckpointRepository):
        saved = await repo.save(_record())
        assert saved.created_at is not None
        # Normalise to UTC-aware regardless of asyncpg codec settings.
        created_at = saved.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        # PostgreSQL NOW() is the transaction-start time, so it may be a few
        # milliseconds before Python captures datetime.now().  We just verify
        # the timestamp is recent (within the last 10 seconds).
        age_seconds = abs((datetime.now(UTC) - created_at).total_seconds())
        assert age_seconds < 10, f"created_at is not recent: {created_at}"

    async def test_save_persists_session_id(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="my-session"))
        fetched = await repo.get_latest("my-session")
        assert fetched is not None
        assert fetched.session_id == "my-session"

    async def test_save_persists_stage(self, repo: CheckpointRepository):
        saved = await repo.save(_record(stage=CheckpointStage.RETRIEVAL))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.stage == CheckpointStage.RETRIEVAL

    async def test_save_persists_route(self, repo: CheckpointRepository):
        saved = await repo.save(_record(route="support"))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.route == "support"

    async def test_save_persists_null_route(self, repo: CheckpointRepository):
        saved = await repo.save(_record(route=None))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.route is None

    async def test_save_persists_doc_counts(self, repo: CheckpointRepository):
        saved = await repo.save(_record(retrieved=7, reranked=3))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.retrieved_doc_count == 7
        assert fetched.reranked_doc_count == 3

    async def test_save_persists_boolean_fields(self, repo: CheckpointRepository):
        saved = await repo.save(_record(has_draft=True, has_so=False))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.has_draft is True
        assert fetched.has_structured_output is False

    async def test_save_persists_jsonb_snapshot(self, repo: CheckpointRepository):
        snap = {"query": "test", "route": "research", "step": 5, "nested": {"k": "v"}}
        saved = await repo.save(_record(snapshot=snap))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.state_snapshot == snap

    async def test_save_persists_error_count(self, repo: CheckpointRepository):
        saved = await repo.save(_record(errors=3))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.error_count == 3

    async def test_save_persists_approval_status(self, repo: CheckpointRepository):
        saved = await repo.save(_record(approval="approved"))
        fetched = await repo.get_latest(saved.session_id)
        assert fetched is not None
        assert fetched.approval_status == "approved"


# ── Integration: list_by_session ─────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositoryList:
    async def test_list_returns_all_records_for_session(self, repo: CheckpointRepository):
        for stage in [CheckpointStage.ROUTING, CheckpointStage.RETRIEVAL, CheckpointStage.GENERATION]:
            await repo.save(_record(session_id="list-sess", stage=stage))
        assert len(await repo.list_by_session("list-sess")) == 3

    async def test_list_returns_empty_for_unknown_session(self, repo: CheckpointRepository):
        assert await repo.list_by_session("no-such-session") == []

    async def test_list_ordered_oldest_first(self, repo: CheckpointRepository):
        for stage in [CheckpointStage.ROUTING, CheckpointStage.RETRIEVAL, CheckpointStage.GENERATION]:
            await repo.save(_record(session_id="order-sess", stage=stage))
        results = await repo.list_by_session("order-sess")
        assert [r.stage for r in results] == [
            CheckpointStage.ROUTING,
            CheckpointStage.RETRIEVAL,
            CheckpointStage.GENERATION,
        ]

    async def test_list_does_not_return_other_sessions(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="sess-a"))
        await repo.save(_record(session_id="sess-b"))
        results = await repo.list_by_session("sess-a")
        assert all(r.session_id == "sess-a" for r in results)
        assert len(results) == 1


# ── Integration: get_latest ───────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositoryGetLatest:
    async def test_get_latest_returns_most_recent(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="latest-sess", stage=CheckpointStage.ROUTING))
        await repo.save(_record(session_id="latest-sess", stage=CheckpointStage.GENERATION))
        latest = await repo.get_latest("latest-sess")
        assert latest is not None
        assert latest.stage == CheckpointStage.GENERATION

    async def test_get_latest_returns_none_for_unknown_session(self, repo: CheckpointRepository):
        assert await repo.get_latest("ghost-session") is None

    async def test_get_latest_single_record(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="single-sess"))
        latest = await repo.get_latest("single-sess")
        assert latest is not None
        assert latest.session_id == "single-sess"


# ── Integration: get_by_stage ─────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositoryGetByStage:
    async def test_get_by_stage_returns_matching_record(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="stage-sess", stage=CheckpointStage.ROUTING))
        await repo.save(_record(session_id="stage-sess", stage=CheckpointStage.RETRIEVAL))
        result = await repo.get_by_stage("stage-sess", CheckpointStage.ROUTING)
        assert result is not None
        assert result.stage == CheckpointStage.ROUTING

    async def test_get_by_stage_returns_none_when_stage_absent(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="stage-sess2", stage=CheckpointStage.ROUTING))
        assert await repo.get_by_stage("stage-sess2", CheckpointStage.APPROVAL) is None

    async def test_get_by_stage_returns_latest_when_duplicate_stages(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="dup-sess", stage=CheckpointStage.APPROVAL, errors=0))
        await repo.save(_record(session_id="dup-sess", stage=CheckpointStage.APPROVAL, errors=1))
        result = await repo.get_by_stage("dup-sess", CheckpointStage.APPROVAL)
        assert result is not None
        assert result.error_count == 1  # most recent row

    async def test_all_five_pipeline_stages_roundtrip(self, repo: CheckpointRepository):
        """Demonstrate persisting each of the five required pipeline stages."""
        stages = [
            CheckpointStage.ROUTING,
            CheckpointStage.RETRIEVAL,
            CheckpointStage.RERANKING,
            CheckpointStage.GENERATION,
            CheckpointStage.APPROVAL,
        ]
        for stage in stages:
            await repo.save(_record(session_id="all-stages", stage=stage))
        for stage in stages:
            found = await repo.get_by_stage("all-stages", stage)
            assert found is not None, f"Stage {stage} not found"
            assert found.stage == stage


# ── Integration: count ────────────────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositoryCount:
    async def test_count_returns_zero_for_unknown_session(self, repo: CheckpointRepository):
        assert await repo.count("none") == 0

    async def test_count_returns_correct_number(self, repo: CheckpointRepository):
        for _ in range(4):
            await repo.save(_record(session_id="count-sess"))
        assert await repo.count("count-sess") == 4

    async def test_count_is_session_scoped(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="c-a"))
        await repo.save(_record(session_id="c-a"))
        await repo.save(_record(session_id="c-b"))
        assert await repo.count("c-a") == 2
        assert await repo.count("c-b") == 1


# ── Integration: delete_session ───────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointRepositoryDelete:
    async def test_delete_returns_row_count(self, repo: CheckpointRepository):
        for _ in range(3):
            await repo.save(_record(session_id="del-sess"))
        assert await repo.delete_session("del-sess") == 3

    async def test_delete_removes_all_session_records(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="del-clean"))
        await repo.delete_session("del-clean")
        assert await repo.count("del-clean") == 0

    async def test_delete_does_not_affect_other_sessions(self, repo: CheckpointRepository):
        await repo.save(_record(session_id="keep-me"))
        await repo.save(_record(session_id="delete-me"))
        await repo.delete_session("delete-me")
        assert await repo.count("keep-me") == 1

    async def test_delete_unknown_session_returns_zero(self, repo: CheckpointRepository):
        assert await repo.delete_session("no-such-session") == 0


# ── Integration: checkpoint_node ─────────────────────────────────────────────


@pytest.mark.integration
class TestCheckpointNode:
    @pytest.fixture(autouse=True)
    def inject_repo(self, repo: CheckpointRepository):
        set_repository(repo)
        yield
        set_repository(None)

    async def test_node_saves_record_to_database(self, repo: CheckpointRepository):
        state = _state(
            session_id="node-sess-01",
            route="research",
            with_retrieved=True,
            with_reranked=True,
            with_so=True,
        )
        update = await checkpoint_node(state)
        assert "errors" not in update
        assert len(await repo.list_by_session("node-sess-01")) == 1

    async def test_node_persists_correct_stage(self, repo: CheckpointRepository):
        await checkpoint_node(_state(session_id="node-sess-02", with_so=True))
        latest = await repo.get_latest("node-sess-02")
        assert latest is not None
        assert latest.stage == CheckpointStage.GENERATION

    async def test_node_persists_doc_counts(self, repo: CheckpointRepository):
        state = _state(session_id="node-sess-03", with_retrieved=True, with_reranked=True)
        await checkpoint_node(state)
        latest = await repo.get_latest("node-sess-03")
        assert latest is not None
        assert latest.retrieved_doc_count == 3
        assert latest.reranked_doc_count == 2

    async def test_node_sets_current_node(self, repo: CheckpointRepository):
        update = await checkpoint_node(_state(session_id="node-sess-04", with_so=True))
        assert update["current_node"] == "checkpoint"

    async def test_node_increments_step_count(self, repo: CheckpointRepository):
        s = _state(session_id="node-sess-05")
        s["step_count"] = 4
        update = await checkpoint_node(s)
        assert update["step_count"] == 5

    async def test_node_records_error_when_repository_not_set(self):
        set_repository(None)
        update = await checkpoint_node(_state(session_id="node-sess-06"))
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "checkpoint"

    async def test_routing_stage_checkpoint(self, repo: CheckpointRepository):
        state = _state(session_id="stage-routing", route="support")
        await checkpoint_node(state)
        rec = await repo.get_by_stage("stage-routing", CheckpointStage.ROUTING)
        assert rec is not None
        assert rec.route == "support"

    async def test_five_stages_via_sequential_node_calls(self, repo: CheckpointRepository):
        """Persist all five required pipeline stages using sequential node calls."""
        sid = "five-stages"

        # 1. Routing
        await checkpoint_node(_state(session_id=sid, route="research"))
        # 2. Retrieval
        await checkpoint_node(_state(session_id=sid, route="research", with_retrieved=True))
        # 3. Reranking
        await checkpoint_node(_state(session_id=sid, route="research",
                                     with_retrieved=True, with_reranked=True))
        # 4. Generation
        await checkpoint_node(_state(session_id=sid, route="research",
                                     with_retrieved=True, with_reranked=True, with_so=True))
        # 5. Approval — saved directly via the repository (approval node not yet implemented)
        await repo.save(CheckpointRecord(
            session_id=sid, stage=CheckpointStage.APPROVAL, query="test query",
            route="research", has_structured_output=True,
            approval_status="approved", state_snapshot={},
        ))

        assert await repo.count(sid) == 5

        gen = await repo.get_by_stage(sid, CheckpointStage.GENERATION)
        assert gen is not None

        app_rec = await repo.get_by_stage(sid, CheckpointStage.APPROVAL)
        assert app_rec is not None
        assert app_rec.approval_status == "approved"
