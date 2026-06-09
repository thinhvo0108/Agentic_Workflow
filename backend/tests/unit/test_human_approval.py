"""
Unit tests for the human-approval workflow.

No real LangGraph runtime or Ollama required — everything is mocked.

Test groups
-----------
TestHumanApprovalNode   — node state contract (approved / rejected / pending guard)
TestFinalResponseNode   — node state contract (assembles FinalResponse, error paths)
TestDeriveStatus        — _derive_status() pure function covering all six states
TestApprovalService     — get_status, get_state, submit_decision, get_final_response
TestApprovalRoutes      — FastAPI endpoints via TestClient (submit, status, result, approve)
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import ApprovalError
from app.graph.nodes.final_response import final_response_node
from app.graph.nodes.human_approval import human_approval_node
from app.graph.state import (
    Citation,
    FinalResponse,
    StructuredOutput,
    initial_state,
)
from app.services.approval_service import ApprovalService, _derive_status


# ── Factories ──────────────────────────────────────────────────────────────────


def _state(
    session_id: str = "sess-ha",
    approval_status: str = "pending",
    with_structured_output: bool = False,
    route: str = "research",
    step: int = 3,
):
    s = initial_state(session_id=session_id, query="test query")
    s["approval_status"] = approval_status  # type: ignore[assignment]
    s["step_count"] = step
    s["route"] = route  # type: ignore[assignment]
    if with_structured_output:
        s["structured_output"] = StructuredOutput(
            summary="This is a summary.",
            answer="This is the detailed answer to the query.",
            citations=[
                Citation(
                    document_id="doc-1",
                    source="kb.txt",
                    excerpt="A relevant excerpt from the document.",
                    rerank_score=0.92,
                )
            ],
        )
    return s


def _snapshot(next_nodes: tuple = (), values: dict | None = None) -> MagicMock:
    snap = MagicMock()
    snap.next = next_nodes
    snap.values = values or {"session_id": "sess", "errors": []}
    return snap


def _mock_workflow(
    snapshot: MagicMock | None = None,
    update_state_effect=None,
    invoke_effect=None,
) -> MagicMock:
    wf = MagicMock()
    wf.aget_state = AsyncMock(return_value=snapshot or _snapshot())
    wf.aupdate_state = AsyncMock(side_effect=update_state_effect)
    wf.ainvoke = AsyncMock(side_effect=invoke_effect)
    return wf


# ── human_approval_node ───────────────────────────────────────────────────────


class TestHumanApprovalNode:
    @pytest.mark.asyncio
    async def test_sets_current_node_on_approved(self):
        update = await human_approval_node(_state(approval_status="approved"))
        assert update["current_node"] == "human_approval"

    @pytest.mark.asyncio
    async def test_sets_current_node_on_rejected(self):
        update = await human_approval_node(_state(approval_status="rejected"))
        assert update["current_node"] == "human_approval"

    @pytest.mark.asyncio
    async def test_increments_step_count(self):
        update = await human_approval_node(_state(approval_status="approved", step=7))
        assert update["step_count"] == 8

    @pytest.mark.asyncio
    async def test_no_errors_key_on_approved(self):
        update = await human_approval_node(_state(approval_status="approved"))
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_no_errors_key_on_rejected(self):
        update = await human_approval_node(_state(approval_status="rejected"))
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_records_error_when_still_pending(self):
        """If the node runs before a decision is submitted, it must error cleanly."""
        update = await human_approval_node(_state(approval_status="pending"))
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "human_approval"
        assert "pending" in update["errors"][0]["message"]

    @pytest.mark.asyncio
    async def test_reads_approval_record_from_state(self):
        s = _state(approval_status="approved")
        s["approval_record"] = {
            "reviewer_id": "alice",
            "action": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }
        update = await human_approval_node(s)
        assert update["current_node"] == "human_approval"


# ── final_response_node ───────────────────────────────────────────────────────


class TestFinalResponseNode:
    @pytest.mark.asyncio
    async def test_assembles_final_response(self):
        update = await final_response_node(_state(with_structured_output=True))
        assert "final_response" in update
        fr = update["final_response"]
        assert fr["session_id"] == "sess-ha"
        assert fr["summary"] == "This is a summary."

    @pytest.mark.asyncio
    async def test_sets_approval_status_to_approved(self):
        update = await final_response_node(_state(with_structured_output=True))
        assert update["final_response"]["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_preserves_route(self):
        update = await final_response_node(_state(with_structured_output=True, route="support"))
        assert update["final_response"]["route"] == "support"

    @pytest.mark.asyncio
    async def test_preserves_citations(self):
        update = await final_response_node(_state(with_structured_output=True))
        citations = update["final_response"]["citations"]
        assert len(citations) == 1
        assert citations[0]["document_id"] == "doc-1"
        assert citations[0]["rerank_score"] == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_created_at_is_iso_string(self):
        update = await final_response_node(_state(with_structured_output=True))
        ts = update["final_response"]["created_at"]
        assert "T" in ts  # ISO-8601 marker

    @pytest.mark.asyncio
    async def test_sets_current_node(self):
        update = await final_response_node(_state(with_structured_output=True))
        assert update["current_node"] == "final_response"

    @pytest.mark.asyncio
    async def test_increments_step_count(self):
        update = await final_response_node(_state(with_structured_output=True, step=5))
        assert update["step_count"] == 6

    @pytest.mark.asyncio
    async def test_records_error_when_no_structured_output(self):
        update = await final_response_node(_state(with_structured_output=False))
        assert len(update["errors"]) == 1
        assert update["errors"][0]["node"] == "final_response"
        assert "final_response" not in update

    @pytest.mark.asyncio
    async def test_no_errors_key_on_success(self):
        update = await final_response_node(_state(with_structured_output=True))
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_empty_citations_list_preserved(self):
        s = _state()
        s["structured_output"] = StructuredOutput(
            summary="a" * 10, answer="b" * 20, citations=[]
        )
        update = await final_response_node(s)
        assert update["final_response"]["citations"] == []


# ── _derive_status ────────────────────────────────────────────────────────────


class TestDeriveStatus:
    def test_none_snapshot_is_not_found(self):
        assert _derive_status(None) == "not_found"

    def test_empty_values_is_not_found(self):
        snap = MagicMock()
        snap.values = {}
        snap.next = ()
        assert _derive_status(snap) == "not_found"

    def test_human_approval_in_next_is_awaiting_approval(self):
        snap = _snapshot(next_nodes=("human_approval",), values={"session_id": "s", "errors": []})
        assert _derive_status(snap) == "awaiting_approval"

    def test_non_empty_next_other_than_approval_is_running(self):
        snap = _snapshot(next_nodes=("generator",), values={"session_id": "s", "errors": []})
        assert _derive_status(snap) == "running"

    def test_completed_with_final_response(self):
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [], "final_response": {"answer": "ok"}},
        )
        assert _derive_status(snap) == "completed"

    def test_rejected_when_approval_status_rejected(self):
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [], "approval_status": "rejected"},
        )
        assert _derive_status(snap) == "rejected"

    def test_failed_when_errors_present_and_no_final_response(self):
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [{"node": "router", "message": "fail"}]},
        )
        assert _derive_status(snap) == "failed"

    def test_completed_when_no_errors_and_no_final_response(self):
        snap = _snapshot(next_nodes=(), values={"session_id": "s", "errors": []})
        assert _derive_status(snap) == "completed"


# ── ApprovalService ───────────────────────────────────────────────────────────


class TestApprovalService:
    @pytest.mark.asyncio
    async def test_get_status_awaiting_approval(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_status("s") == "awaiting_approval"

    @pytest.mark.asyncio
    async def test_get_status_completed(self):
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [], "final_response": {"answer": "ok"}},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_status("s") == "completed"

    @pytest.mark.asyncio
    async def test_get_status_rejected(self):
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [], "approval_status": "rejected"},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_status("s") == "rejected"

    @pytest.mark.asyncio
    async def test_get_status_not_found_for_missing_session(self):
        wf = _mock_workflow(snapshot=None)
        wf.aget_state = AsyncMock(return_value=MagicMock(values=None, next=()))
        svc = ApprovalService(workflow=wf)
        assert await svc.get_status("missing") == "not_found"

    @pytest.mark.asyncio
    async def test_get_state_returns_state_values(self):
        values = {"session_id": "s", "query": "q", "errors": []}
        snap = _snapshot(next_nodes=(), values=values)
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        result = await svc.get_state("s")
        assert result == values

    @pytest.mark.asyncio
    async def test_get_state_returns_none_for_unknown_session(self):
        wf = MagicMock()
        wf.aget_state = AsyncMock(return_value=MagicMock(values=None, next=()))
        svc = ApprovalService(workflow=wf)
        assert await svc.get_state("ghost") is None

    @pytest.mark.asyncio
    async def test_get_current_node_returns_first_next_node(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_current_node("s") == "human_approval"

    @pytest.mark.asyncio
    async def test_get_current_node_returns_none_when_complete(self):
        snap = _snapshot(next_nodes=(), values={"session_id": "s", "errors": []})
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_current_node("s") is None

    @pytest.mark.asyncio
    async def test_submit_decision_calls_update_and_invoke(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        wf = _mock_workflow(snapshot=snap)
        svc = ApprovalService(workflow=wf)
        await svc.submit_decision("s", "approved", "alice")
        wf.aupdate_state.assert_called_once()
        wf.ainvoke.assert_called_once_with(None, {"configurable": {"thread_id": "s"}})

    @pytest.mark.asyncio
    async def test_submit_decision_injects_approval_status(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        wf = _mock_workflow(snapshot=snap)
        svc = ApprovalService(workflow=wf)
        await svc.submit_decision("s", "rejected", "bob", comment="needs revision")
        call_args = wf.aupdate_state.call_args
        state_update = call_args[0][1]  # second positional arg
        assert state_update["approval_status"] == "rejected"
        assert state_update["approval_record"]["reviewer_id"] == "bob"
        assert state_update["approval_record"]["comment"] == "needs revision"

    @pytest.mark.asyncio
    async def test_submit_decision_raises_when_not_awaiting_approval(self):
        snap = _snapshot(
            next_nodes=("generator",),  # not awaiting approval
            values={"session_id": "s", "errors": []},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        with pytest.raises(ApprovalError, match="not awaiting approval"):
            await svc.submit_decision("s", "approved", "alice")

    @pytest.mark.asyncio
    async def test_submit_decision_raises_for_unknown_session(self):
        wf = MagicMock()
        wf.aget_state = AsyncMock(return_value=MagicMock(values=None, next=()))
        svc = ApprovalService(workflow=wf)
        with pytest.raises(ApprovalError, match="not found"):
            await svc.submit_decision("ghost", "approved", "alice")

    @pytest.mark.asyncio
    async def test_submit_decision_raises_for_invalid_action(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        with pytest.raises(ApprovalError, match="Invalid action"):
            await svc.submit_decision("s", "maybe", "alice")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_get_final_response_returns_value_when_complete(self):
        fr = {"session_id": "s", "summary": "s" * 10, "answer": "a" * 20,
              "citations": [], "route": "research", "approval_status": "approved",
              "created_at": "2026-01-01T00:00:00Z"}
        snap = _snapshot(
            next_nodes=(),
            values={"session_id": "s", "errors": [], "final_response": fr},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        result = await svc.get_final_response("s")
        assert result == fr

    @pytest.mark.asyncio
    async def test_get_final_response_returns_none_when_not_complete(self):
        snap = _snapshot(
            next_nodes=("human_approval",),
            values={"session_id": "s", "errors": []},
        )
        svc = ApprovalService(workflow=_mock_workflow(snapshot=snap))
        assert await svc.get_final_response("s") is None


# ── API routes ────────────────────────────────────────────────────────────────


def _make_approval_svc(
    status: str = "awaiting_approval",
    state: dict | None = None,
    final_response: dict | None = None,
) -> ApprovalService:
    svc = MagicMock(spec=ApprovalService)
    svc.get_status = AsyncMock(return_value=status)
    svc.get_state = AsyncMock(return_value=state or {"session_id": "s", "errors": []})
    svc.get_current_node = AsyncMock(return_value="human_approval" if status == "awaiting_approval" else None)
    svc.get_final_response = AsyncMock(return_value=final_response)
    svc.submit_decision = AsyncMock()
    return svc


def _make_test_client(svc: ApprovalService) -> TestClient:
    from app.api.dependencies import get_approval_service, get_workflow
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_approval_service] = lambda: svc
    application.dependency_overrides[get_workflow] = lambda: MagicMock()
    return TestClient(application, raise_server_exceptions=False)


class TestApprovalRoutes:
    def test_submit_workflow_returns_202(self):
        svc = _make_approval_svc(status="running")
        client = _make_test_client(svc)
        resp = client.post("/api/v1/workflow", json={"query": "How does BERT work?"})
        assert resp.status_code == 202

    def test_submit_workflow_returns_session_id(self):
        svc = _make_approval_svc(status="running")
        client = _make_test_client(svc)
        resp = client.post("/api/v1/workflow", json={"query": "test query"})
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_submit_workflow_returns_running_status(self):
        svc = _make_approval_svc(status="running")
        client = _make_test_client(svc)
        resp = client.post("/api/v1/workflow", json={"query": "test query"})
        assert resp.json()["status"] == "running"

    def test_submit_workflow_accepts_custom_session_id(self):
        svc = _make_approval_svc(status="running")
        client = _make_test_client(svc)
        resp = client.post(
            "/api/v1/workflow",
            json={"query": "q", "session_id": "my-custom-id"},
        )
        assert resp.json()["session_id"] == "my-custom-id"

    def test_get_status_awaiting_approval(self):
        svc = _make_approval_svc(status="awaiting_approval")
        client = _make_test_client(svc)
        resp = client.get("/api/v1/workflow/sess-001")
        assert resp.status_code == 200
        assert resp.json()["status"] == "awaiting_approval"

    def test_get_status_includes_session_id(self):
        svc = _make_approval_svc(status="running")
        client = _make_test_client(svc)
        resp = client.get("/api/v1/workflow/sess-123")
        assert resp.json()["session_id"] == "sess-123"

    def test_get_status_completed(self):
        svc = _make_approval_svc(status="completed")
        client = _make_test_client(svc)
        resp = client.get("/api/v1/workflow/sess-ok")
        assert resp.json()["status"] == "completed"

    def test_get_result_returns_200_when_complete(self):
        fr = {
            "session_id": "s", "summary": "Good summary here.",
            "answer": "Detailed answer to the query.",
            "citations": [], "route": "research",
            "approval_status": "approved", "created_at": "2026-01-01T00:00:00Z",
        }
        svc = _make_approval_svc(status="completed", final_response=fr)
        client = _make_test_client(svc)
        resp = client.get("/api/v1/workflow/s/result")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"] == "Good summary here."
        assert body["route"] == "research"

    def test_get_result_returns_404_when_not_complete(self):
        svc = _make_approval_svc(status="awaiting_approval", final_response=None)
        client = _make_test_client(svc)
        resp = client.get("/api/v1/workflow/pending-sess/result")
        assert resp.status_code == 404

    def test_submit_approval_returns_200(self):
        svc = _make_approval_svc()
        client = _make_test_client(svc)
        resp = client.post(
            "/api/v1/workflow/sess-001/approve",
            json={
                "session_id": "sess-001",
                "action": "approved",
                "reviewer_id": "alice",
            },
        )
        assert resp.status_code == 200

    def test_submit_approval_response_contains_action(self):
        svc = _make_approval_svc()
        client = _make_test_client(svc)
        resp = client.post(
            "/api/v1/workflow/sess-001/approve",
            json={"session_id": "sess-001", "action": "rejected", "reviewer_id": "bob"},
        )
        assert resp.json()["action"] == "rejected"
        assert resp.json()["reviewer_id"] == "bob"

    def test_submit_approval_calls_service_with_correct_args(self):
        svc = _make_approval_svc()
        client = _make_test_client(svc)
        client.post(
            "/api/v1/workflow/sess-001/approve",
            json={
                "session_id": "sess-001",
                "action": "approved",
                "reviewer_id": "charlie",
                "comment": "Looks good",
            },
        )
        svc.submit_decision.assert_called_once_with(
            session_id="sess-001",
            action="approved",
            reviewer_id="charlie",
            comment="Looks good",
        )

    def test_submit_approval_returns_409_when_not_awaiting(self):
        svc = _make_approval_svc()
        svc.submit_decision = AsyncMock(
            side_effect=ApprovalError("not awaiting approval")
        )
        client = _make_test_client(svc)
        resp = client.post(
            "/api/v1/workflow/running-sess/approve",
            json={"session_id": "running-sess", "action": "approved", "reviewer_id": "x"},
        )
        assert resp.status_code == 409

    def test_submit_approval_returns_422_when_session_id_mismatch(self):
        svc = _make_approval_svc()
        client = _make_test_client(svc)
        resp = client.post(
            "/api/v1/workflow/sess-001/approve",
            json={"session_id": "different-id", "action": "approved", "reviewer_id": "x"},
        )
        assert resp.status_code == 422
