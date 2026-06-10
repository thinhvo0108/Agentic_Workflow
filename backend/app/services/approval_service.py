"""
ApprovalService — orchestrates the human-in-the-loop approval lifecycle.

Workflow integration
--------------------
The LangGraph graph is compiled with interrupt_before=["human_approval"].
When the workflow reaches that point, it persists state to the checkpointer
and returns.  This service:

  1. Exposes get_status() so the API can tell callers whether a session is
     running, awaiting approval, completed, rejected, or failed.

  2. Exposes submit_decision() which:
       a. Validates the session is actually awaiting approval.
       b. Injects approval_status + approval_record into state via aupdate_state().
       c. Resumes the workflow with ainvoke(None, config).
       d. After resumption the human_approval_node body runs, the conditional
          edge routes to final_response (approved) or END (rejected), and the
          graph completes within the same ainvoke call.

  3. Exposes get_state() to give the API layer raw access to state values.

Dependency injection
--------------------
The workflow (CompiledStateGraph) is injected so the service can be tested
with a mock graph without touching the LangGraph runtime.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph

from app.core.exceptions import ApprovalError
from app.core.logging import get_logger
from app.graph.state import ApprovalRecord, ApprovalStatus, FinalResponse

_logger = get_logger(__name__)

WorkflowStatus = Literal[
    "running", "awaiting_approval", "completed", "rejected", "failed", "not_found"
]


def _derive_status(snapshot: Any | None) -> WorkflowStatus:
    """Map a LangGraph StateSnapshot to a human-readable workflow status."""
    if snapshot is None or not snapshot.values:
        return "not_found"

    next_nodes: tuple = snapshot.next or ()
    if "human_approval" in next_nodes:
        return "awaiting_approval"

    if next_nodes:
        return "running"

    # Graph has finished — inspect final state to classify the outcome.
    values: dict = snapshot.values
    if values.get("final_response") is not None:
        return "completed"
    if values.get("approval_status") == "rejected":
        return "rejected"
    if values.get("errors"):
        return "failed"
    # step_count=0 means the initial checkpoint was written but no node has run
    # yet (AsyncPostgresSaver commits the initial state before starting).
    if not values.get("step_count"):
        return "running"
    return "failed"


class ApprovalService:
    """Manages the human-approval lifecycle for in-progress workflows.

    Parameters
    ----------
    workflow:
        A compiled LangGraph graph.  Inject a mock in tests.
    """

    def __init__(self, workflow: CompiledStateGraph) -> None:
        self._workflow = workflow

    # ── Status ─────────────────────────────────────────────────────────────────

    async def get_status(self, session_id: str) -> WorkflowStatus:
        """Return the current status of a workflow session."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._workflow.aget_state(config)
        return _derive_status(snapshot)

    async def get_state(self, session_id: str) -> dict | None:
        """Return raw state values for a session, or None if not found."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._workflow.aget_state(config)
        if snapshot is None or not snapshot.values:
            return None
        return dict(snapshot.values)

    async def get_current_node(self, session_id: str) -> str | None:
        """Return the name of the next pending node, or None."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._workflow.aget_state(config)
        if snapshot is None:
            return None
        next_nodes = snapshot.next or ()
        return next_nodes[0] if next_nodes else None

    # ── Decision ───────────────────────────────────────────────────────────────

    async def submit_decision(
        self,
        session_id: str,
        action: ApprovalStatus,
        reviewer_id: str,
        comment: str | None = None,
    ) -> None:
        """Inject an approval decision and resume the workflow.

        Parameters
        ----------
        session_id:
            The workflow session to approve or reject.
        action:
            "approved" or "rejected".
        reviewer_id:
            Identity of the reviewer submitting the decision.
        comment:
            Optional free-text comment.

        Raises
        ------
        ApprovalError
            If the session does not exist or is not awaiting approval.
        """
        if action not in ("approved", "rejected"):
            raise ApprovalError(
                f"Invalid action {action!r} — must be 'approved' or 'rejected'"
            )

        config = {"configurable": {"thread_id": session_id}}
        snapshot = await self._workflow.aget_state(config)

        if snapshot is None or not snapshot.values:
            raise ApprovalError(f"Session '{session_id}' not found")

        next_nodes: tuple = snapshot.next or ()
        if "human_approval" not in next_nodes:
            raise ApprovalError(
                f"Session '{session_id}' is not awaiting approval "
                f"(current next nodes: {list(next_nodes)})"
            )

        record: ApprovalRecord = {
            "reviewer_id": reviewer_id,
            "action": action,
            "decided_at": datetime.now(UTC).isoformat(),
        }
        if comment is not None:
            record["comment"] = comment

        # Inject decision into state, then resume.
        await self._workflow.aupdate_state(
            config,
            {"approval_status": action, "approval_record": record},
        )
        await self._workflow.ainvoke(None, config)

        _logger.info(
            "approval_submitted",
            session_id=session_id,
            action=action,
            reviewer_id=reviewer_id,
        )

    # ── Result ─────────────────────────────────────────────────────────────────

    async def get_final_response(self, session_id: str) -> FinalResponse | None:
        """Return the FinalResponse for a completed session, or None."""
        state = await self.get_state(session_id)
        if state is None:
            return None
        return state.get("final_response")
