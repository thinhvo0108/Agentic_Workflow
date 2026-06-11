"""
Auto-approval gate node.

Runs after the checkpoint node, before human_approval.

If the overall confidence score meets or exceeds AUTO_APPROVE_THRESHOLD the
workflow is approved automatically and routes directly to final_response —
bypassing the human_approval interrupt entirely.

If the score is below the threshold the node writes nothing to approval_status
so the conditional edge routes to human_approval, where LangGraph's
interrupt_before pauses execution for a manual decision.
"""

from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.graph.state import ApprovalRecord, AppState
from app.services.confidence import score_overall

_logger = get_logger(__name__)
_NODE = "auto_approval_gate"

AUTO_APPROVE_THRESHOLD = 0.70


async def auto_approval_gate_node(state: AppState) -> dict[str, Any]:
    """Auto-approve if overall confidence >= 70%; otherwise defer to human review."""
    router_conf = float(state.get("router_confidence") or 0.0)
    retrieval_conf = float(state.get("retrieval_confidence") or 0.0)
    answer_conf = float(state.get("answer_confidence") or 0.0)
    overall = score_overall(router_conf, retrieval_conf, answer_conf)
    step = state.get("step_count", 0) + 1
    auto_approved = overall >= AUTO_APPROVE_THRESHOLD

    _logger.info(
        "auto_approval_gate",
        session_id=state["session_id"],
        overall_confidence=overall,
        threshold=AUTO_APPROVE_THRESHOLD,
        auto_approved=auto_approved,
    )

    if auto_approved:
        record: ApprovalRecord = {
            "reviewer_id": "system",
            "action": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
            "comment": f"Auto-approved: confidence {overall:.0%} ≥ {AUTO_APPROVE_THRESHOLD:.0%} threshold",
        }
        return {
            "approval_status": "approved",
            "approval_record": record,
            "auto_approved": True,
            "current_node": _NODE,
            "step_count": step,
        }

    # Below threshold — pass through; human_approval will handle it.
    return {
        "auto_approved": False,
        "current_node": _NODE,
        "step_count": step,
    }
