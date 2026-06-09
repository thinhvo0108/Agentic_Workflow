from datetime import UTC, datetime

from app.core.logging import get_logger
from app.graph.state import AppState, Citation, ConfidenceScores, FinalResponse, make_error
from app.services.confidence import score_overall

# GroundednessResult imported at usage for type narrowing only

_logger = get_logger(__name__)

_NODE = "final_response"


async def final_response_node(state: AppState) -> dict:
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

    response = FinalResponse(
        session_id=state["session_id"],
        summary=so.get("summary", ""),
        answer=so.get("answer", ""),
        citations=citations,
        route=state.get("route") or "research",
        approval_status="approved",
        created_at=datetime.now(UTC).isoformat(),
        confidence=confidence,
        groundedness=state.get("groundedness"),
    )

    _logger.info(
        "final_response_node_done",
        session_id=state["session_id"],
        citation_count=len(citations),
        confidence_overall=confidence["overall"],
    )
    return {
        "final_response": response,
        "current_node": _NODE,
        "step_count": step,
    }
