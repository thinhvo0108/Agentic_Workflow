from app.graph.state import AppState, ApprovalRecord


async def human_approval_node(state: AppState) -> dict:
    """Gate node that pauses the graph until a reviewer acts.

    Because the graph is compiled with interrupt_before=["human_approval"],
    LangGraph persists state and returns control to the caller before this
    node body runs.  The approval service resumes the graph by injecting
    approval_status and approval_record into state via graph.update_state().

    Reads
    -----
    approval_status : ApprovalStatus  ("pending" | "approved" | "rejected")
    approval_record : ApprovalRecord | None

    Writes
    ------
    current_node : str
    errors : list  (appended on failure)
    """
    raise NotImplementedError
