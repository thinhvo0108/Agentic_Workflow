"""
Checkpoint node — persists an application-level audit record to PostgreSQL.

This runs after the structured-output node and before human-approval.
At that point every upstream field (route, retrieved_documents,
reranked_documents, draft_response, structured_output) is available, so the
snapshot covers all five pipeline stages in a single record.

The LangGraph built-in checkpointer (AsyncPostgresSaver) independently saves
the full raw state after every node transition; this node writes a curated,
human-readable row to workflow_checkpoints that is easy to query and audit.

Repository injection
--------------------
The module-level _repository can be overridden via set_repository() so
integration tests can supply a real pool without touching application settings.
"""

import json
from typing import Any

from app.checkpoints.models import CheckpointRecord, CheckpointStage
from app.checkpoints.repository import CheckpointRepository
from app.core.exceptions import CheckpointError
from app.core.logging import get_logger
from app.graph.state import AppState, make_error

_logger = get_logger(__name__)

_NODE = "checkpoint"

# Module-level repository singleton.  Set via set_repository() before the node
# is invoked (during app startup or in integration tests).
_repository: CheckpointRepository | None = None


def set_repository(repo: CheckpointRepository | None) -> None:
    """Override the module-level repository.  Call from app startup or tests."""
    global _repository
    _repository = repo


def _get_repository() -> CheckpointRepository:
    if _repository is None:
        raise CheckpointError(
            "CheckpointRepository is not set — call set_repository() at startup"
        )
    return _repository


# ── State → checkpoint helpers ─────────────────────────────────────────────────


def _determine_stage(state: AppState) -> CheckpointStage:
    """Infer the most advanced completed pipeline stage from state."""
    if state.get("structured_output") is not None or state.get("draft_response"):
        return CheckpointStage.GENERATION
    if state.get("reranked_documents"):
        return CheckpointStage.RERANKING
    if state.get("retrieved_documents"):
        return CheckpointStage.RETRIEVAL
    if state.get("route") is not None:
        return CheckpointStage.ROUTING
    return CheckpointStage.ROUTING


def _build_snapshot(state: AppState) -> dict[str, Any]:
    """Build a compact, JSON-serialisable state summary.

    Large fields (document content, full draft) are summarised rather than
    stored verbatim to keep row sizes manageable.
    """
    so = state.get("structured_output")
    snapshot: dict[str, Any] = {
        "session_id": state["session_id"],
        "query": state["query"],
        "route": state.get("route"),
        "step_count": state.get("step_count", 0),
        "retrieved_doc_ids": [
            d["id"] for d in (state.get("retrieved_documents") or [])
        ],
        "reranked_doc_ids": [
            d["id"] for d in (state.get("reranked_documents") or [])
        ],
        "has_draft": bool(state.get("draft_response")),
        "has_structured_output": so is not None,
        "approval_status": state.get("approval_status"),
        "errors": state.get("errors") or [],
    }
    if so is not None:
        snapshot["structured_output_summary"] = {
            "summary": (so.get("summary") or "")[:200],
            "citation_count": len(so.get("citations") or []),
        }
    return snapshot


def _build_record(state: AppState) -> CheckpointRecord:
    retrieved = state.get("retrieved_documents") or []
    reranked = state.get("reranked_documents") or []
    errors = state.get("errors") or []
    return CheckpointRecord(
        session_id=state["session_id"],
        stage=_determine_stage(state),
        query=state["query"],
        route=state.get("route"),
        retrieved_doc_count=len(retrieved),
        reranked_doc_count=len(reranked),
        has_draft=bool(state.get("draft_response")),
        has_structured_output=state.get("structured_output") is not None,
        approval_status=state.get("approval_status"),
        error_count=len(errors),
        state_snapshot=_build_snapshot(state),
    )


# ── Node ───────────────────────────────────────────────────────────────────────


async def checkpoint_node(state: AppState) -> dict:
    """Persist an audit checkpoint to PostgreSQL.

    Reads
    -----
    All populated state fields are summarised into the checkpoint record.

    Writes
    ------
    current_node : str
    step_count   : incremented by 1
    errors       : appended on failure
    """
    _logger.info(
        "checkpoint_node_start",
        session_id=state["session_id"],
        stage=str(_determine_stage(state)),
    )

    step = state.get("step_count", 0) + 1

    try:
        repo = _get_repository()
        record = _build_record(state)
        saved = await repo.save(record)
    except CheckpointError as exc:
        _logger.error(
            "checkpoint_node_failed",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, str(exc))],
        }
    except Exception as exc:
        _logger.error(
            "checkpoint_node_unexpected",
            session_id=state["session_id"],
            error=str(exc),
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, f"Unexpected error: {exc}")],
        }

    _logger.info(
        "checkpoint_node_done",
        session_id=state["session_id"],
        checkpoint_id=saved.id,
        stage=str(saved.stage),
    )
    return {
        "current_node": _NODE,
        "step_count": step,
    }
