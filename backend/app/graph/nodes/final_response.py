from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.graph.state import (
    AppState,
    Citation,
    ConfidenceScores,
    FinalResponse,
    WorkflowMetrics,
    make_error,
)
from app.services.confidence import score_overall

_logger = get_logger(__name__)

_NODE = "final_response"


def _compute_metrics(state: AppState, completed_at: datetime) -> WorkflowMetrics:
    """Derive observability metrics from accumulated state at workflow completion."""
    started_at_str = state.get("started_at") or completed_at.isoformat()
    try:
        started = datetime.fromisoformat(started_at_str)
        latency_ms = round((completed_at - started).total_seconds() * 1000, 1)
    except (ValueError, TypeError):
        latency_ms = 0.0

    step_count = max(state.get("step_count", 0), 1)
    error_count = len(state.get("errors") or [])

    gnd = state.get("groundedness")
    hallucination_rate: float | None = None
    if gnd is not None:
        hallucination_rate = round(1.0 - float(gnd.get("groundedness_score", 0.0)), 4)

    judge = state.get("judge_result")
    judge_score: float | None = float(judge["overall_score"]) if judge else None

    cp = state.get("context_precision")
    context_precision_score: float | None = float(cp["context_precision_score"]) if cp else None

    return WorkflowMetrics(
        started_at=started_at_str,
        completed_at=completed_at.isoformat(),
        latency_ms=latency_ms,
        total_tokens=state.get("total_tokens", 0),
        error_count=error_count,
        error_rate=round(error_count / step_count, 4),
        hallucination_rate=hallucination_rate,
        judge_score=judge_score,
        context_precision_score=context_precision_score,
        step_count=step_count,
    )


async def final_response_node(state: AppState) -> dict[str, Any]:
    """Assemble the approved FinalResponse from structured_output.

    Reads
    -----
    session_id, structured_output, route, approval_status

    Writes
    ------
    final_response : FinalResponse
    current_node   : str
    step_count     : incremented by 1
    errors         : appended when structured_output is absent
    """
    _logger.info("final_response_node_start", session_id=state["session_id"])

    now = datetime.now(UTC)
    step = state.get("step_count", 0) + 1
    so = state.get("structured_output")

    if so is None:
        _logger.error("final_response_no_structured_output", session_id=state["session_id"])
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, "structured_output not found in state")],
        }

    citations: list[Citation] = [
        Citation(
            document_id=c.get("document_id", ""),
            source=c.get("source", ""),
            excerpt=c.get("excerpt", ""),
            rerank_score=float(c.get("rerank_score", 0.0)),
        )
        for c in (so.get("citations") or [])
    ]

    router_conf = state.get("router_confidence") or 0.0
    retrieval_conf = state.get("retrieval_confidence") or 0.0
    answer_conf = state.get("answer_confidence") or 0.0
    confidence = ConfidenceScores(
        router=router_conf,
        retrieval=retrieval_conf,
        answer=answer_conf,
        overall=score_overall(router_conf, retrieval_conf, answer_conf),
    )

    record: dict[str, Any] = dict(state.get("approval_record") or {})
    metrics = _compute_metrics(state, now)
    response = FinalResponse(
        session_id=state["session_id"],
        summary=so.get("summary", ""),
        answer=so.get("answer", ""),
        citations=citations,
        route=state.get("route") or "research",
        approval_status="approved",
        auto_approved=bool(state.get("auto_approved", False)),
        reviewer_id=record.get("reviewer_id") or None,
        reviewer_comment=record.get("comment") or None,
        created_at=now.isoformat(),
        confidence=confidence,
        groundedness=state.get("groundedness"),
        context_precision=state.get("context_precision"),
        judge_result=state.get("judge_result"),
        metrics=metrics,
    )

    _logger.info(
        "final_response_node_done",
        session_id=state["session_id"],
        citation_count=len(citations),
        confidence_overall=confidence["overall"],
        latency_ms=metrics["latency_ms"],
        total_tokens=metrics["total_tokens"],
        error_rate=metrics["error_rate"],
        hallucination_rate=metrics["hallucination_rate"],
    )
    return {
        "final_response": response,
        "current_node": _NODE,
        "step_count": step,
    }
