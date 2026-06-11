"""
Auto-approval gate node.

Runs after the checkpoint node, before human_approval.

Auto-approves only when BOTH conditions are satisfied (judge-as-veto pattern):
  1. overall_confidence >= AUTO_APPROVE_THRESHOLD  (pipeline signal)
  2. judge_result.overall_score >= JUDGE_THRESHOLD  (LLM quality signal)

If judge_result is absent (judge node failed), the gate falls back to
confidence-only — graceful degradation preserves the existing behaviour.

When either condition fails the node leaves approval_status untouched so the
conditional edge routes to web_search → human_approval.
"""

from datetime import UTC, datetime
from typing import Any

from app.core.logging import get_logger
from app.graph.state import ApprovalRecord, AppState
from app.services.confidence import score_overall

_logger = get_logger(__name__)
_NODE = "auto_approval_gate"

AUTO_APPROVE_THRESHOLD = 0.70
JUDGE_THRESHOLD = 0.60


async def auto_approval_gate_node(state: AppState) -> dict[str, Any]:
    """Auto-approve when confidence AND judge score both meet their thresholds."""
    router_conf = float(state.get("router_confidence") or 0.0)
    retrieval_conf = float(state.get("retrieval_confidence") or 0.0)
    answer_conf = float(state.get("answer_confidence") or 0.0)
    overall = score_overall(router_conf, retrieval_conf, answer_conf)

    # Judge-as-veto: if judge ran, it must also pass its threshold.
    judge_result = state.get("judge_result")
    judge_score: float | None = judge_result["overall_score"] if judge_result else None
    judge_passed = judge_score is None or judge_score >= JUDGE_THRESHOLD

    step = state.get("step_count", 0) + 1
    auto_approved = overall >= AUTO_APPROVE_THRESHOLD and judge_passed

    _logger.info(
        "auto_approval_gate",
        session_id=state["session_id"],
        overall_confidence=overall,
        confidence_threshold=AUTO_APPROVE_THRESHOLD,
        confidence_passed=overall >= AUTO_APPROVE_THRESHOLD,
        judge_score=judge_score,
        judge_threshold=JUDGE_THRESHOLD,
        judge_passed=judge_passed,
        auto_approved=auto_approved,
    )

    if auto_approved:
        comment = f"Auto-approved: confidence {overall:.0%} ≥ {AUTO_APPROVE_THRESHOLD:.0%}" + (
            f", judge {judge_score:.0%} ≥ {JUDGE_THRESHOLD:.0%}" if judge_score is not None else ""
        )
        record: ApprovalRecord = {
            "reviewer_id": "system",
            "action": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
            "comment": comment,
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
