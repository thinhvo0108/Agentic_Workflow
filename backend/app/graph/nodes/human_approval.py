"""
Human approval node — executes AFTER the reviewer submits a decision.

LangGraph lifecycle
-------------------
The graph is compiled with interrupt_before=["human_approval"], so execution
PAUSES before this node body runs.  The ApprovalService resumes the graph by:
  1. Calling graph.aupdate_state(config, {"approval_status": ..., "approval_record": ...})
  2. Calling graph.ainvoke(None, config)

At that point this node body executes.  The state already has the reviewer's
decision, so this node only needs to log it and advance tracking fields.

The conditional edge (_approval_decision) runs after the node and routes:
  "approved"  → final_response
  "rejected"  → END
"""

from app.core.logging import get_logger
from app.graph.state import AppState, make_error

_logger = get_logger(__name__)

_NODE = "human_approval"


async def human_approval_node(state: AppState) -> dict:
    """Record the reviewer's decision and yield control to the conditional edge.

    Reads
    -----
    approval_status : "pending" | "approved" | "rejected"
    approval_record : ApprovalRecord | None

    Writes
    ------
    current_node : str
    step_count   : incremented by 1
    errors       : appended when approval_status is still "pending" after resume
                   (indicates the node was called before a decision was submitted)
    """
    status = state.get("approval_status", "pending")
    record = state.get("approval_record")
    step = state.get("step_count", 0) + 1

    _logger.info(
        "human_approval_decision",
        session_id=state["session_id"],
        status=status,
        reviewer_id=record.get("reviewer_id") if record else None,
    )

    if status == "pending":
        # Resumed without a decision — this should not happen in normal flow.
        _logger.warning(
            "human_approval_no_decision",
            session_id=state["session_id"],
        )
        return {
            "current_node": _NODE,
            "step_count": step,
            "errors": [make_error(_NODE, "Node executed with approval_status still pending")],
        }

    return {
        "current_node": _NODE,
        "step_count": step,
    }
